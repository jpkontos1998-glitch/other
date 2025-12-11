import unittest
import random

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego, set_seed_everywhere

pystratego = get_pystratego()

red_chars = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "B"]
blue_chars = ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "N"]
red2blue = {red_chars[i]: blue_chars[i] for i in range(len(red_chars))}
blue2red = {blue_chars[i]: red_chars[i] for i in range(len(blue_chars))}


class ResetTest(unittest.TestCase):
    def test_uniform_histogram(self):
        set_seed_everywhere(0)
        piece_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40]
        c = random.randint(1, 30)
        piece_counter[random.randint(0, 9)] += c
        piece_counter[-1] -= c
        bs = pystratego.PieceArrangementGenerator(piece_counter)
        red_ids = [100]
        blue_ids = [0]
        arrangements_red = bs.generate_string_arrangements(red_ids)
        arrangements_blue = bs.generate_string_arrangements(blue_ids)
        arrangement_pairs = [arrangements_red, arrangements_blue]
        num_envs = 1

        move_memory = 32
        num_row = 2 * move_memory
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=num_row,
            move_memory=move_memory,
            max_num_moves=20000,
        )
        env.change_reset_behavior_to_random_initial_arrangement(arrangement_pairs)
        env.reset()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        # Wait until game has terminated
        while True:
            if (env.env.get_terminated_since(env.current_step) > 0).any():
                break
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        # Change reset behavior
        new_red_ids = [101]
        new_blue_ids = [1]
        new_arrangements_red = bs.generate_string_arrangements(new_red_ids)
        new_arrangements_blue = bs.generate_string_arrangements(new_blue_ids)
        new_arrangement_pairs = [new_arrangements_red, new_arrangements_blue]
        env.change_reset_behavior(new_arrangement_pairs)

        # Wait until reset occurs
        while True:
            if (env.env.get_terminated_since(env.current_step) == 0).any():
                break
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        # Test initial arrangement distribution is correct
        board_str = env.board_strs(env.current_step)
        r_arr = board_str[0].upper().replace("@", "")[:40]
        b_arr_tmp = board_str[0].upper().replace("@", "")[-40:]
        b_arr = "".join(reversed([blue2red[b] if b in blue2red else b for b in b_arr_tmp]))
        self.assertEqual(r_arr, new_arrangements_red[0])
        self.assertEqual(b_arr, new_arrangements_blue[0])
        self.assertNotEqual(r_arr, arrangements_red[0])
        self.assertNotEqual(b_arr, arrangements_blue[0])

    def test_old_assertion(self):
        move_memory = 32
        num_row = 2 * move_memory
        env = Stratego(
            num_envs=64, traj_len_per_player=num_row, move_memory=move_memory, max_num_moves=20000
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                [
                    "AAMALKBCAEAADDAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "ALAKDAMBECDAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "ALMCEDKBDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "LEBMKCADDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "AMLDAABKEDCAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                ],
                [
                    "CBDMADLKAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "DKLADMCBEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "MKEAADDLAAABCAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "BMLAKCADAAADEAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    "MAAEBDLKADACAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                ],
            ]
        )
        env.reset()

    def test_p1_reset_state(self):
        num_envs = 2
        for i in range(10):
            env = Stratego(num_envs=num_envs, traj_len_per_player=8)
            action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            env_state = env.current_state
            env.change_reset_behavior_to_env_state(env_state)
            player_at_termination = None
            while True:
                if env.current_is_terminal[0] and env.current_player == 0:
                    player_at_termination = 0
                    break
                if env.current_is_terminal[0] and env.current_player == 0:
                    player_at_termination = 1
                    break
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
            running_player = player_at_termination
            for j in range(5):
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
                running_player = 1 - running_player
                if (env_state.boards[0] == env.current_state.boards[0]).all():
                    break
            else:
                self.fail("Custom state was not reached")
            self.assertEqual(running_player, 1)


if __name__ == "__main__":
    unittest.main()
