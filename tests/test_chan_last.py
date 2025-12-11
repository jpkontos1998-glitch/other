import torch

from pyengine.networks.legacy_rl import TransformerRL, TransformerRLConfig
from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego
from pyengine.utils.constants import N_ACTION
from pyengine import utils

pystratego = get_pystratego()

env = Stratego(
    num_envs=128,
    traj_len_per_player=100,
)


def benchmark_chan_first():
    obs_size = env.obs_shape
    net = TransformerRL(
        torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
        TransformerRLConfig(depth=1, embed_dim_per_head_over8=1),
    ).to("cuda")

    input = torch.rand(128, *obs_size).cuda()
    legal_actions = (torch.rand(128, N_ACTION) * 2).long().cuda()
    torch.cuda.synchronize()

    t = time.time()
    for _ in range(200):
        y = net(input, env.current_piece_ids, legal_actions, env.current_num_moves)["action"]
        y = y.detach()
        torch.cuda.synchronize()

    t = time.time() - t
    print(f"[channel first] time spent: {t:.2f}")


def benchmark_chan_last():
    obs_size = env.obs_shape
    net = TransformerRL(
        torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 32], device="cuda"),
        TransformerRLConfig(depth=1, embed_dim_per_head_over8=1),
    ).to("cuda")
    net = net.to(memory_format=torch.channels_last)

    input = torch.rand(128, *obs_size).to("cuda", memory_format=torch.channels_last)
    # input = input.permute([0, 2, 3, 1]).contiguous()
    legal_actions = (torch.rand(128, N_ACTION) * 2).long().cuda()
    torch.cuda.synchronize()

    t = time.time()
    for _ in range(200):
        y = net(input, env.current_piece_ids, legal_actions, env.current_num_moves)["action"]
        # print(y.size())
        y = y.detach()
        torch.cuda.synchronize()

    t = time.time() - t
    print(f"[channel last] time spent: {t:.2f}")


if __name__ == "__main__":
    import time

    print("warm up")
    benchmark_chan_last()
    benchmark_chan_first()

    print("benchmark")
    benchmark_chan_last()
    benchmark_chan_first()
