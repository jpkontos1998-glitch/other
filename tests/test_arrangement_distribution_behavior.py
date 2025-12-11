import unittest
import random

import torch

from pyengine.utils import get_pystratego, set_seed_everywhere

pystratego = get_pystratego()

red_chars = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "B"]
blue_chars = ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "N"]
red2blue = {red_chars[i]: blue_chars[i] for i in range(len(red_chars))}
blue2red = {blue_chars[i]: red_chars[i] for i in range(len(blue_chars))}


class DistributionTest(unittest.TestCase):
    def test_uniform_histogram(self):
        set_seed_everywhere(0)
        piece_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40]
        c = random.randint(1, 30)
        piece_counter[random.randint(0, 9)] += c
        piece_counter[-1] -= c
        bs = pystratego.PieceArrangementGenerator(piece_counter)
        red_ids = [100, 101]
        blue_ids = [0, 1, 2]
        arrangements_red = bs.generate_string_arrangements(red_ids)
        arrangements_blue = bs.generate_string_arrangements(blue_ids)
        arrangement_pairs = [arrangements_red, arrangements_blue]
        num_envs = 1024

        num_row = 210
        env = pystratego.StrategoRolloutBuffer(
            num_row,
            num_envs,
            reset_behavior=pystratego.ResetBehavior.RANDOM_CUSTOM_INITIAL_ARRANGEMENT,
            move_memory=32,
            max_num_moves=20000,
            initial_arrangements=arrangement_pairs,
        )
        # Test initial arrangement distribution is correct
        red_counters = [0, 0]
        blue_counters = [0, 0, 0]
        for i in range(num_envs):
            board_str = env.board_strs(env.current_step())
            r_arr = board_str[i].upper().replace("@", "")[:40]
            b_arr_tmp = board_str[i].upper().replace("@", "")[-40:]
            b_arr = "".join(reversed([blue2red[b] if b in blue2red else b for b in b_arr_tmp]))
            red_counters[arrangements_red.index(r_arr)] += 1 / num_envs
            blue_counters[arrangements_blue.index(b_arr)] += 1 / num_envs
        self.assertTrue(all([0.4 < p < 0.6 for p in red_counters]))
        self.assertTrue(all([0.23 < p < 0.43 for p in blue_counters]))

        # Test distribution remains correct after reset
        env.reset()
        red_counters = [0, 0]
        blue_counters = [0, 0, 0]
        for i in range(num_envs):
            board_str = env.board_strs(env.current_step())
            r_arr = board_str[i].upper().replace("@", "")[:40]
            b_arr_tmp = board_str[i].upper().replace("@", "")[-40:]
            b_arr = "".join(reversed([blue2red[b] if b in blue2red else b for b in b_arr_tmp]))
            red_counters[arrangements_red.index(r_arr)] += 1 / num_envs
            blue_counters[arrangements_blue.index(b_arr)] += 1 / num_envs
        self.assertTrue(all([0.4 < p < 0.6 for p in red_counters]))
        self.assertTrue(all([0.23 < p < 0.43 for p in blue_counters]))

    def test_non_uniform_histogram(self):
        set_seed_everywhere(0)
        piece_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40]
        c = random.randint(1, 30)
        piece_counter[random.randint(0, 9)] += c
        piece_counter[-1] -= c
        bs = pystratego.PieceArrangementGenerator(piece_counter)
        red_ids = [0, 1]
        blue_ids = [101, 102, 103]
        arrangements_red = bs.generate_string_arrangements(red_ids)
        arrangements_blue = bs.generate_string_arrangements(blue_ids)
        arrangement_pairs = (arrangements_red, arrangements_blue)
        num_envs = 1024

        num_row = 210
        env = pystratego.StrategoRolloutBuffer(
            num_row,
            num_envs,
            reset_behavior=pystratego.ResetBehavior.RANDOM_CUSTOM_INITIAL_ARRANGEMENT,
            move_memory=32,
            max_num_moves=20000,
            initial_arrangements=arrangement_pairs,
        )
        env.change_reset_behavior(
            arrangement_pairs,
            (torch.tensor([0.1, 0.9], device="cuda"), torch.tensor([0, 0.3, 0.7], device="cuda")),
        )
        # Test that there was no implicit reset
        red_counters = [0, 0]
        blue_counters = [0, 0, 0]
        for i in range(num_envs):
            board_str = env.board_strs(env.current_step())
            r_arr = board_str[i].upper().replace("@", "")[:40]
            b_arr_tmp = board_str[i].upper().replace("@", "")[-40:]
            b_arr = "".join(reversed([blue2red[b] if b in blue2red else b for b in b_arr_tmp]))
            red_counters[arrangements_red.index(r_arr)] += 1 / num_envs
            blue_counters[arrangements_blue.index(b_arr)] += 1 / num_envs
        self.assertTrue(all([0.4 < p < 0.6 for p in red_counters]))
        self.assertTrue(all([0.23 < p < 0.43 for p in blue_counters]))

        # Test that the distribution is correct after reset
        env.reset()
        red_counters = [0, 0]
        blue_counters = [0, 0, 0]
        for i in range(num_envs):
            board_str = env.board_strs(env.current_step())
            r_arr = board_str[i].upper().replace("@", "")[:40]
            b_arr_tmp = board_str[i].upper().replace("@", "")[-40:]
            b_arr = "".join(reversed([blue2red[b] if b in blue2red else b for b in b_arr_tmp]))
            red_counters[arrangements_red.index(r_arr)] += 1 / num_envs
            blue_counters[arrangements_blue.index(b_arr)] += 1 / num_envs
        self.assertTrue(0 < red_counters[0] < 0.2)
        self.assertTrue(blue_counters[0] == 0)
        self.assertTrue(0.2 < blue_counters[1] < 0.4)


if __name__ == "__main__":
    unittest.main()
