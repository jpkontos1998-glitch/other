import unittest

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()


class UnknownPieceCountTest(unittest.TestCase):
    def test_unknown_piece_count(self):
        num_envs = 100
        env = Stratego(num_envs=num_envs, traj_len_per_player=100, full_info=False, barrage=False)
        action_tensor = torch.zeros(num_envs, device="cuda", dtype=torch.int32)
        for _ in range(1000):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            unknown_counts1 = env.current_unknown_piece_type_onehot.cumsum(dim=1)[:, -1, :12]
            unknown_counts2 = env.current_unknown_piece_counts
            self.assertTrue(torch.all(unknown_counts1 == unknown_counts2))


if __name__ == "__main__":
    unittest.main()
