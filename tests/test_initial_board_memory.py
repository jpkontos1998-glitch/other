import unittest
import random

import torch

from pyengine.core.env import Stratego
from pyengine.utils import set_seed_everywhere, get_pystratego

pystratego = get_pystratego()


class MemoryTest(unittest.TestCase):
    def test_string(self):
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=50,
            traj_len_per_player=8,
            full_info=False,
            barrage=True,
        )
        initial_boards = env.current_board_strs
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for i in range(10_000):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            is_terminal = env.current_is_terminal
            if is_terminal.any():
                for j in range(env.num_envs):
                    if is_terminal[j]:
                        self.assertTrue(initial_boards[j] == env.current_zero_board_strs[j])
            num_moves_since_reset = env.current_num_moves_since_reset
            if (num_moves_since_reset == 0).any():
                for j in range(env.num_envs):
                    if num_moves_since_reset[j] == 0:
                        initial_boards[j] = env.current_board_strs[j]

    def test_tensor(self):
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=50,
            traj_len_per_player=8,
            full_info=False,
            barrage=True,
        )
        initial_boards = env.current_state.boards
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for i in range(10_000):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            env.current_infostate_tensor
            is_terminal = env.current_is_terminal
            if is_terminal.any():
                for j in range(env.num_envs):
                    if is_terminal[j]:
                        self.assertTrue((initial_boards[j] == env.current_zero_boards[j]).all())
            num_moves_since_reset = env.current_num_moves_since_reset
            if (num_moves_since_reset == 0).any():
                for j in range(env.num_envs):
                    if num_moves_since_reset[j] == 0:
                        initial_boards[j] = env.current_state.boards[j]
                        pieces = env.current_zero_boards[j, 0:1600:16].cpu()
                        ids = env.current_zero_boards[j, 1:1600:16].cpu()
                        self.assertEqual(env.current_acting_player, 0)
                        for cell in range(100):
                            relative_cell = cell if cell < 40 else 99 - cell
                            self.assertTrue(
                                ids[cell] == relative_cell or pieces[cell] % 16 in (12, 13)
                            )  # EMPTY

    def test_zero_arrangement(self):
        set_seed_everywhere(0)
        piece_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40]
        c = random.randint(1, 30)
        piece_counter[random.randint(0, 9)] += c
        piece_counter[-1] -= c
        bs = pystratego.PieceArrangementGenerator(piece_counter)
        red_ids = [111]
        blue_ids = [222]
        arrangements_red = bs.generate_string_arrangements(red_ids)
        arrangements_blue = bs.generate_string_arrangements(blue_ids)
        red_tensor = pystratego.util.arrangement_tensor_from_strings(arrangements_red).to("cuda")
        blue_tensor = pystratego.util.arrangement_tensor_from_strings(arrangements_blue).to("cuda")
        arrangement_pairs = [arrangements_red, arrangements_blue]
        env = Stratego(
            num_envs=50,
            traj_len_per_player=8,
            full_info=False,
            barrage=True,
        )
        env.change_reset_behavior_to_random_initial_arrangement(arrangement_pairs)
        env.reset()
        red_tensor2, blue_tensor2 = env.current_zero_arrangements
        self.assertTrue((red_tensor == red_tensor2).all())
        self.assertTrue((blue_tensor == blue_tensor2).all())


if __name__ == "__main__":
    unittest.main()
