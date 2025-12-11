from dataclasses import dataclass, fields, is_dataclass, field
import os
import io
import pprint
import random
import pickle
import sys
import yaml
from typing import Optional

import pyrallis
import torch
import torch.distributed as dist

from pyengine.arrangement.utils import filter_terminal
from pyengine.core.env import Stratego
from pyengine.core.evaluation import EvaluationManager
from pyengine.core.rl import RL, RLConfig, RLResumeConfig
from pyengine.networks.move_transformer import MoveTransformer
from pyengine.networks.arrangement_transformer import ArrangementTransformer
from pyengine.utils.types import InitApproach
from pyengine.utils.constants import N_PLAYER
from pyengine import utils

pystratego = utils.get_pystratego()


class DDPWrapper(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@dataclass
class ResumeConfig:
    max_num_moves_between_attacks: Optional[int] = None
    eval_against: Optional[list[str]] = None
    use_wb: Optional[bool] = None


@dataclass
class MainConfig(utils.RunConfig):
    seed: int = 1
    num_envs: int = 1600
    full_info: bool = False
    barrage: bool = False
    rl: RLConfig = field(default_factory=lambda: RLConfig())
    move_memory: int = 86
    max_num_moves: int = 4000
    max_num_moves_between_attacks: int = 100
    eval_against: list[str] = field(default_factory=lambda: [])
    init_approach: InitApproach = InitApproach.NN
    random_inits: int = 1_000_000
    save_dir: str = ""
    use_wb: bool = False
    resume: ResumeConfig = field(default_factory=lambda: ResumeConfig())

    def __post_init__(self):
        if self.save_dir == "":
            self.save_dir = "exps/tmp"

        if len(self.eval_against):
            print("Will eval against:")
            pprint.pprint(self.eval_against)


def setup(cfg: MainConfig):
    if cfg.is_distributed:
        # initialize the process group for distributed training
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        torch.cuda.set_device(0)


def construct_envs(cfg):
    if cfg.init_approach == InitApproach.UNIFORM:
        if cfg.barrage:
            board_variant = pystratego.BoardVariant.BARRAGE
        else:
            board_variant = pystratego.BoardVariant.CLASSIC

        gen = pystratego.PieceArrangementGenerator(board_variant)
        ids = [random.randint(0, gen.num_possible_arrangements()) for _ in range(cfg.random_inits)]
        with open(os.path.join(cfg.save_dir, "arrangement_ids.pkl"), "wb") as f:
            pickle.dump(ids, f)
        arrangements = filter_terminal(gen.generate_string_arrangements(ids))
        custom_arrs = [arrangements, arrangements]
    else:
        custom_arrs = None

    if cfg.is_distributed:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0

    train_env = Stratego(
        num_envs=cfg.num_envs,
        traj_len_per_player=cfg.max_num_moves // N_PLAYER + cfg.rl.train_every_per_player,
        full_info=bool(cfg.full_info),
        barrage=bool(cfg.barrage),
        custom_arrs=custom_arrs,
        cuda_device=rank,
        quiet=1,
        move_memory=cfg.move_memory,
        max_num_moves=cfg.max_num_moves,
        max_num_moves_between_attacks=cfg.max_num_moves_between_attacks,
    )
    test_env = Stratego(
        num_envs=cfg.num_envs // 2,
        traj_len_per_player=2,
        full_info=bool(cfg.full_info),
        barrage=bool(cfg.barrage),
        verbose=False,
        custom_arrs=custom_arrs,
        cuda_device=rank,
        move_memory=cfg.move_memory,
    )
    return train_env, test_env


def construct_model(train_env, cfg: MainConfig):
    if cfg.is_distributed:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0

    if cfg.barrage:
        piece_counts = torch.tensor(
            utils.init_helpers.COUNTERS["barrage"] + [0, 32], device=f"cuda:{rank}"
        )
    else:
        piece_counts = torch.tensor(
            utils.init_helpers.COUNTERS["classic"] + [0, 0], device=f"cuda:{rank}"
        )
    rl_model = MoveTransformer(
        piece_counts,
        cfg.rl.move_transformer,
    ).to(rank)
    if cfg.is_distributed:
        rl_model = DDPWrapper(rl_model, device_ids=[rank])

    print("put rl model on device ", rank)
    print(rl_model)
    utils.count_parameters(rl_model)

    if cfg.init_approach == InitApproach.NN:
        print("putting arr model on device ", rank)
        arr_model = ArrangementTransformer(
            piece_counts,
            cfg=cfg.rl.arr_transformer,
        ).to(rank)
        if cfg.is_distributed:
            arr_model = DDPWrapper(arr_model, device_ids=[rank])
        print(arr_model)
        utils.count_parameters(arr_model)
    else:
        arr_model = None
    return rl_model, arr_model


def filter_unknown_fields(cls, cfg, path=""):
    result = {}
    field_map = {f.name: f for f in fields(cls)}
    for k, v in cfg.items():
        if k not in field_map:
            print(f"Warning: Ignoring unknown field: {path + k}")
            continue

        field_type = field_map[k].type
        # If the field itself is a dataclass and the value is a dict, filter recursively.
        if is_dataclass(field_type) and isinstance(v, dict):
            result[k] = filter_unknown_fields(field_type, v, path + k + ".")
        else:
            result[k] = v

    return result


def safe_load_config(cls, path):
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    filtered_cfg = filter_unknown_fields(cls, raw_cfg)
    filtered_yaml = yaml.dump(filtered_cfg)
    stream = io.StringIO(filtered_yaml)
    return pyrallis.load(cls, stream)


def run_training(cfg):
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    setup(cfg)

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

    train_env, test_env = construct_envs(cfg)
    print(utils.wrap_ruler(""))

    rl_model, arr_model = construct_model(train_env, cfg)

    stats = utils.MultiCounter(
        cfg.save_dir,
        use_wandb=bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
        rank=global_rank,
    )

    eval_manager = EvaluationManager(test_env)
    for path in cfg.eval_against:
        eval_manager.add_checkpoint(path)

    agent = RL(
        rl_model,
        train_env,
        stats,
        cfg.save_dir,
        eval_manager,
        cfg.rl,
        rank=local_rank,
        arr_model=arr_model,
        saver_enabled=global_rank == 0,
    )
    agent.learn()


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    if cfg.rl.resume_from:
        resume_cfg_path = os.path.join(os.path.dirname(cfg.rl.resume_from), "cfg.yaml")
        print(f"Resume run, loading config from {resume_cfg_path}")
        resume_cfg = safe_load_config(MainConfig, resume_cfg_path)
        resume_cfg.rl.resume_from = cfg.rl.resume_from
        resume_cfg.save_dir = cfg.save_dir
        # Overwrite main config with active resume cfg parameters
        for field_name in fields(ResumeConfig):
            cfg_value = getattr(cfg.resume, field_name.name)
            if cfg_value is not None:
                setattr(resume_cfg, field_name.name, cfg_value)
        # Overwrite config with active resume cfg parameters
        for field_name in fields(RLResumeConfig):
            cfg_value = getattr(cfg.rl.resume, field_name.name)
            if cfg_value is not None:
                setattr(resume_cfg.rl, field_name.name, cfg_value)
        cfg = resume_cfg

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        cfg.is_distributed = True
        os.environ["OMP_NUM_THREADS"] = "4"
    else:
        cfg.is_distributed = False

    run_training(cfg)
