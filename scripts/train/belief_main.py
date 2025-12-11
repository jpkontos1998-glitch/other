import sys
from dataclasses import dataclass, field
import yaml
import os
import torch.distributed as dist

import pyrallis
import torch

from pyengine.belief.belief import Belief, BeliefConfig
from pyengine.networks.lstm_belief_transformer import LSTMBeliefTransformer
from pyengine.networks.temporal_belief_transformer import TemporalBeliefTransformer
from pyengine.networks.belief_transformer import BeliefTransformer
from pyengine import utils
from pyengine.utils.constants import N_PLAYER
from pyengine.utils.loading import load_rl_model, load_arrangement_model, load_env


class DDPWrapper(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@dataclass
class MainConfig(utils.RunConfig):
    seed: int = 1
    num_envs: int = 1024
    belief: BeliefConfig = field(default_factory=lambda: BeliefConfig())
    move_memory: int = 86
    max_num_moves: int = 4000
    max_num_moves_between_attacks: int = 100
    rl_model_path: str = ""
    val_model_path: str = ""
    use_wb: int = 0
    save_dir: str = "exps/belief/run1"


def setup():
    if cfg.is_distributed:
        # initialize the process group for distributed training
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        torch.cuda.set_device(0)


def run_training(cfg):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    setup()

    if cfg.is_distributed:
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        global_rank = 0
        local_rank = 0

    # print to both std and a file
    if global_rank == 0:
        sys.stdout = utils.Logger(cfg.log_path, print_to_stdout=True)
    else:
        ext_idx = cfg.log_path.find(".log")
        sys.stdout = utils.Logger(
            cfg.log_path[:ext_idx] + "_rank" + str(global_rank) + ".log", print_to_stdout=False
        )

    utils.set_seed_everywhere(cfg.seed + global_rank)

    pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
    print(utils.wrap_ruler("train_info"))
    with open(cfg.cfg_path, "r") as f:
        print(f.read(), end="")
    cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

    env = load_env(
        cfg.rl_model_path,
        num_envs=cfg.num_envs,
        traj_len_per_player=cfg.max_num_moves // N_PLAYER + cfg.belief.num_row,
        max_num_moves=cfg.max_num_moves,
        max_num_moves_between_attacks=cfg.max_num_moves_between_attacks,
        cuda_device=local_rank,
        quiet=1,
    )
    print(utils.wrap_ruler(""))

    rl_model, arr = load_rl_model(cfg.rl_model_path, rank=local_rank)
    init_model = load_arrangement_model(
        cfg.rl_model_path.replace("model", "init_model"), rank=local_rank
    )
    if cfg.is_distributed:
        rl_model = DDPWrapper(rl_model, device_ids=[local_rank])
        init_model = DDPWrapper(init_model, device_ids=[local_rank])

    if cfg.belief.use_temporal_model:
        belief_model = TemporalBeliefTransformer(cfg.max_num_moves, cfg.belief.temporal_belief).to(
            local_rank
        )
    elif cfg.belief.use_lstm_model:
        belief_model = LSTMBeliefTransformer(cfg.max_num_moves, cfg.belief.lstm_belief).to(
            local_rank
        )
    else:
        belief_model = BeliefTransformer(cfg.belief.belief_transformer).to(local_rank)

    if cfg.is_distributed:
        belief_model = DDPWrapper(belief_model, device_ids=[local_rank])

    if global_rank == 0:
        print(belief_model)
        utils.count_parameters(belief_model)

    stats = utils.MultiCounter(
        cfg.save_dir,
        use_wandb=bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
        rank=global_rank,
    )
    saver = utils.Saver(cfg.save_dir, "belief")

    learner = Belief(
        belief_model,
        rl_model,
        init_model,
        env,
        stats,
        saver,
        cfg.belief,
        rank=local_rank,
        saver_enabled=global_rank == 0,
    )
    learner.learn()


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        cfg.is_distributed = True
        os.environ["OMP_NUM_THREADS"] = "4"
    else:
        cfg.is_distributed = False
    run_training(cfg)
