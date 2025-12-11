import unittest
import os

import torch
import msgpack

from pyengine.core.env import Stratego
from pyengine.utils import set_seed_everywhere


class TestSaveGames(unittest.TestCase):
    def test_save_games1(self):
        set_seed_everywhere(0)
        num_envs = 1
        env = Stratego(num_envs, 100, max_num_moves=10)
        my_data = []
        cur_my_data = [[env.current_zero_board_strs[i], []] for i in range(num_envs)]

        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        fn = os.path.join(tmp_dir, "test_save_games.msgpack")
        env.save_games(fn)

        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for _ in range(1000):
            env.sample_random_legal_action(action_tensor)
            for i in range(num_envs):
                cur_my_data[i][-1].append(action_tensor[i].item())
            env.apply_actions(action_tensor)
            for i in range(num_envs):
                if env.current_has_just_reset[i]:
                    my_data.append(cur_my_data[i])
                    cur_my_data[i] = [env.current_zero_board_strs[i], []]
        env.stop_saving_games()
        self.assertTrue(os.path.exists(fn))
        with open(fn, "rb") as f:
            data = msgpack.load(f)
        os.remove(fn)
        os.rmdir(tmp_dir)
        my_data = [(tuple(game[0]), tuple(game[1])) for game in my_data]
        data = [(tuple(game[0]), tuple(game[1])) for game in data]
        for g, g_ in zip(my_data, data):
            assert g[0] == g_[0]
        self.assertEqual(data, my_data)

    def test_save_games2(self):
        set_seed_everywhere(0)
        num_envs = 2
        env = Stratego(num_envs, 100, max_num_moves=10)
        my_data = []
        cur_my_data = [[env.current_zero_board_strs[i], []] for i in range(num_envs)]

        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        fn = os.path.join(tmp_dir, "test_save_games.msgpack")
        env.save_games(fn)

        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for _ in range(20):
            env.sample_random_legal_action(action_tensor)
            for i in range(num_envs):
                cur_my_data[i][-1].append(action_tensor[i].item())
            env.apply_actions(action_tensor)
            for i in range(num_envs):
                if env.current_has_just_reset[i]:
                    my_data.append(cur_my_data[i])
                    cur_my_data[i] = [env.current_zero_board_strs[i], []]
        env.stop_saving_games()
        self.assertTrue(os.path.exists(fn))
        with open(fn, "rb") as f:
            data = msgpack.load(f)
        os.remove(fn)
        os.rmdir(tmp_dir)
        my_data = [(tuple(game[0]), tuple(game[1])) for game in my_data]
        data = [(tuple(game[0]), tuple(game[1])) for game in data]
        self.assertEqual(set(data), set(my_data))

    def test_save_games3(self):
        set_seed_everywhere(0)
        num_envs = 256
        env = Stratego(num_envs, 100, max_num_moves=200)
        my_data = []
        cur_my_data = [[env.current_zero_board_strs[i], []] for i in range(num_envs)]

        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        fn = os.path.join(tmp_dir, "test_save_games.msgpack")
        env.save_games(fn)

        num_steps = 1000
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for t in range(num_steps):
            env.sample_random_legal_action(action_tensor)
            for i in range(num_envs):
                cur_my_data[i][-1].append(action_tensor[i].item())
            env.apply_actions(action_tensor)
            for i in range(num_envs):
                if env.current_has_just_reset[i]:
                    my_data.append(cur_my_data[i])
                    cur_my_data[i] = [env.current_zero_board_strs[i], []]
        env.stop_saving_games()
        self.assertTrue(os.path.exists(fn))
        with open(fn, "rb") as f:
            data = msgpack.load(f)
        os.remove(fn)
        os.rmdir(tmp_dir)
        my_data = set([(tuple(game[0]), tuple(game[1])) for game in my_data])
        data = set([(tuple(game[0]), tuple(game[1])) for game in data])
        self.assertEqual(data, my_data)


if __name__ == "__main__":
    unittest.main()
