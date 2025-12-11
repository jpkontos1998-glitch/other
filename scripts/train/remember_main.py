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

from pyengine.core.env import Stratego
from pyengine.core.remember import Remember, RememberConfig
from pyengine.networks.rl_planar import RLPlanar, RLPlanarConfig
from pyengine.networks.rl_temporal import RLTemporal
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


@dataclass
class MainConfig(utils.RunConfig):
    seed: int = 1
    num_envs: int = 1600
    full_info: bool = False
    barrage: bool = False
    remember: RememberConfig = field(default_factory=lambda: RememberConfig())
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
        arrangements = gen.generate_string_arrangements(ids)
        custom_inits = [arrangements, arrangements]
    else:
        custom_inits = None

    rank = 0

    train_env = Stratego(
        num_envs=cfg.num_envs,
        traj_len_per_player=cfg.max_num_moves // N_PLAYER + cfg.remember.train_every_per_player,
        full_info=bool(cfg.full_info),
        barrage=bool(cfg.barrage),
        custom_inits=custom_inits,
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
        custom_inits=custom_inits,
        cuda_device=rank,
        move_memory=cfg.move_memory,
    )
    return train_env, test_env


def construct_model(train_env, cfg: MainConfig):
    rank = 0

    if cfg.barrage:
        piece_counts = torch.tensor(
            utils.init_helpers.COUNTERS["barrage"] + [0, 32], device=f"cuda:{rank}"
        )
    else:
        piece_counts = torch.tensor(
            utils.init_helpers.COUNTERS["classic"] + [0, 0], device=f"cuda:{rank}"
        )
    if cfg.remember.use_temporal_model:
        model = RLTemporal(
            piece_counts,
            cfg.max_num_moves,
            cfg.remember.temporal_transformer,
        ).to(rank)
    else:
        model = RLPlanar(
            piece_counts,
            cfg.remember.rl_transformer,
        ).to(rank)

    print("put model on device ", rank)
    print(model)
    utils.count_parameters(model)

    if cfg.init_approach == InitApproach.NN:
        print("putting init policy on device ", rank)
        init_policy = ArrangementTransformer(
            piece_counts,
            cfg=cfg.remember.init_transformer,
        ).to(rank)
        print(init_policy)
        utils.count_parameters(init_policy)
    else:
        init_policy = None
    return model, init_policy


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

    model, arr = utils.loading.load_rl_model(cfg.remember.resume_from)
    train_env.change_reset_behavior_to_random_initial_arrangement(arr)

    stats = utils.MultiCounter(
        cfg.save_dir,
        use_wandb=bool(cfg.use_wb),
        wb_exp_name=cfg.wb_exp,
        wb_run_name=cfg.wb_run,
        wb_group_name=cfg.wb_group,
        config=cfg_dict,
        rank=global_rank,
    )

    agent = Remember(
        model,
        train_env,
        stats,
        cfg.save_dir,
        cfg.remember,
        rank=local_rank,
    )
    agent.learn()


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=MainConfig)
    run_training(cfg)
