import time
from argparse import ArgumentParser

import torch

from pyengine.core.search import SearchBot
from pyengine.networks.legacy_rl import TransformerRL, TransformerRLConfig
from pyengine.networks.legacy_belief import ARBelief, ARBeliefConfig
from pyengine.networks.temporal_belief_transformer import (
    TemporalBeliefTransformer,
    TemporalBeliefConfig,
)
from pyengine.core.env import Stratego
from pyengine import utils

parser = ArgumentParser()
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--max_num_samples", type=int, default=200)
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--use_temporal_belief", type=int, default=0)
args = parser.parse_args()

if args.dtype == "float32":
    dtype = torch.float32
elif args.dtype == "bfloat16":
    dtype = torch.bfloat16
else:
    raise ValueError(f"Invalid dtype: {args.dtype}")

policy = TransformerRL(
    torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
    TransformerRLConfig(),
).to("cuda")
env = Stratego(num_envs=1, traj_len_per_player=100)
if args.use_temporal_belief:
    belief_model = TemporalBeliefTransformer(env.conf.max_num_moves, TemporalBeliefConfig()).to(
        "cuda"
    )
    # belief_model = load_belief_model("/scratch/ev2237/log_dir/stratego/07_15_25/large_temp/use_temporal_modeltrue_lr5e-5_rl_model_pathscratchev2237log_dirstratego05_30_25greene_long8_resumeuse_wb1_eval_every2000__7d398f2/belief2430.pthm")
else:
    belief_model = ARBelief(num_piece_type=14, cfg=ARBeliefConfig()).to("cuda")
search_env = Stratego(num_envs=args.num_envs, traj_len_per_player=100, quiet=1)
search_bot = SearchBot(
    policy,
    search_env,
    depth=args.depth,
    stepsize=10,
    temperature=1e-3,
    td_lambda=1.0,
    max_num_samples=args.max_num_samples,
    dtype=dtype,
    belief_model=belief_model,
)
action_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
times = []
for i in range(30):
    torch.cuda.synchronize()
    start = time.time()
    even_states, odd_states = env.snapshot_env_history(env.current_step, 0)
    env_state = even_states if env.current_player == 0 else odd_states
    search_bot(
        env_state,
        env.current_infostate_tensor,
        env.current_piece_ids,
        env.current_legal_action_mask,
        env.current_num_moves,
        env.current_unknown_piece_position_onehot,
        env.current_unknown_piece_counts,
        env.current_unknown_piece_has_moved,
    )
    torch.cuda.synchronize()
    if i > 10:
        times.append(time.time() - start)
    env.sample_random_legal_action(action_tensor)
    env.apply_actions(action_tensor)

print(f"mean search time: {sum(times) / len(times)}")

print(search_bot.stopwatch.summary())
