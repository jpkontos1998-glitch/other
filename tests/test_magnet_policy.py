import torch
import unittest

import numpy as np

from pyengine.core.env import Stratego
from pyengine.utils.constants import N_ACTION
from pyengine.utils import get_pystratego
from pyengine.utils.helper import get_weighted_uniform_policy

pystratego = get_pystratego()


def print_board(env):
    print(
        "\n".join(
            [
                env.board_strs(env.current_step())[0]
                .replace("@", " ")
                .replace(".", " ")[i * 20 : (i + 1) * 20]
                for i in range(10)
            ]
        )
    )


movable_pieces = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


class MagnetTests(unittest.TestCase):
    def test_case(self):
        string_arrangement = "KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"
        valid_origin = [s in movable_pieces for s in string_arrangement] + 60 * [False]
        env = Stratego(
            num_envs=1,
            traj_len_per_player=200,
            custom_inits=(
                [string_arrangement],
                [string_arrangement],
            ),
        )
        legal_action_mask = env.current_legal_action_mask.flatten()
        legal_moves_per_origin = [0 for _ in range(100)]
        # Sanity check
        for i in range(N_ACTION):
            if legal_action_mask[i]:
                assert valid_origin[i % 100]
                legal_moves_per_origin[i % 100] += 1
        # print_board(env)
        # Computed by inspection
        reduced_num_legal_moves = [2, 2, 3, 3, 14, 14]
        num_movable_pieces = len(reduced_num_legal_moves)
        num_legal_moves = []
        for o in valid_origin:
            if o:
                num_legal_moves.append(reduced_num_legal_moves.pop(0))
            else:
                num_legal_moves.append(0)
        # for i in range(4):
        #     print(num_legal_moves[i * 10 : (i + 1) * 10])
        assert legal_moves_per_origin == num_legal_moves
        mass_per_piece = 1 / num_movable_pieces
        mass_per_action_by_origin = [mass_per_piece / o if o > 0 else 0 for o in num_legal_moves]
        # for i in range(4):
        #     print(mass_per_action_by_origin[i * 10 : (i + 1) * 10])
        my_magnet_policy = []
        for i in range(N_ACTION):
            if legal_action_mask[i]:
                my_magnet_policy.append(mass_per_action_by_origin[i % 100])
            else:
                my_magnet_policy.append(0)
        assert np.isclose(sum(my_magnet_policy), 1)
        weighted_uniform_policy = get_weighted_uniform_policy(legal_action_mask.unsqueeze(0))
        self.assertTrue(
            torch.allclose(torch.tensor(my_magnet_policy, device="cuda"), weighted_uniform_policy)
        )

    def test_sequence(self):
        string_arrangement = "KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"
        env = Stratego(
            num_envs=1,
            traj_len_per_player=200,
            custom_inits=(
                [string_arrangement],
                [string_arrangement],
            ),
        )
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for _ in range(1000):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

            legal_action_mask = env.current_legal_action_mask.flatten()
            legal_moves_per_origin = [0 for _ in range(100)]
            for i in range(N_ACTION):
                if legal_action_mask[i]:
                    legal_moves_per_origin[i % 100] += 1
            mass_per_piece = 1 / sum([m > 0 for m in legal_moves_per_origin])
            mass_per_action_by_origin = [
                mass_per_piece / o if o > 0 else 0 for o in legal_moves_per_origin
            ]
            my_magnet_policy = []
            for i in range(N_ACTION):
                if legal_action_mask[i]:
                    my_magnet_policy.append(mass_per_action_by_origin[i % 100])
                else:
                    my_magnet_policy.append(0)
            assert np.isclose(sum(my_magnet_policy), 1)
            weighted_uniform_policy = get_weighted_uniform_policy(legal_action_mask.unsqueeze(0))
            self.assertTrue(
                torch.allclose(
                    torch.tensor(my_magnet_policy, device="cuda"), weighted_uniform_policy
                )
            )


if __name__ == "__main__":
    unittest.main()
