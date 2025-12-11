import unittest
import os

import torch
import msgpack

from pyengine.core.env import Stratego
from pyengine.utils import set_seed_everywhere


class ActionCarousel:
    def __init__(self, actions: list[list[int]], num_envs: int):
        assert len(actions) % num_envs == 0
        packed_actions = [[] for _ in range(num_envs)]
        for i, action_list in enumerate(actions):
            packed_actions[i % num_envs] += action_list
        self.packed_actions = packed_actions
        self.action_lens = [len(actions) for actions in packed_actions]
        self.num_envs = num_envs

    def get_action_tensor(self, t: int):
        action_tensor = []
        for i in range(self.num_envs):
            action_list = self.packed_actions[i]
            action_index = t % self.action_lens[i]
            action_tensor.append(action_list[action_index])
        return torch.tensor(action_tensor, dtype=torch.int32, device="cuda")


class StringCarousel:
    def __init__(self, strings: list[list[str]], num_envs: int):
        assert len(strings) % num_envs == 0
        packed_strings = [[] for _ in range(num_envs)]
        for i, strings_list in enumerate(strings):
            packed_strings[i % num_envs] += strings_list
        self.packed_strings = packed_strings
        self.strings_lens = [len(s) for s in packed_strings]
        self.num_envs = num_envs

    def get_strings(self, t: int):
        strings = []
        for i in range(self.num_envs):
            string_list = self.packed_strings[i]
            index = t % self.strings_lens[i]
            strings.append(string_list[index])
        return strings


class DataCarousel:
    def __init__(self, data: list[list[torch.tensor]], num_envs: int):
        assert len(data) % num_envs == 0
        data = [torch.stack(d) for d in data]
        packed_data = [[] for _ in range(num_envs)]
        for i, d in enumerate(data):
            packed_data[i % num_envs].append(d)
        for i in range(num_envs):
            packed_data[i] = torch.cat(packed_data[i], dim=0)
        self.packed_data = packed_data
        self.data_lens = [len(d) for d in packed_data]
        self.num_envs = num_envs

    def get_data_tensor(self, t: int):
        data = []
        for i in range(self.num_envs):
            data_i = self.packed_data[i]
            data_index = t % self.data_lens[i]
            data.append(data_i[data_index])
        return torch.stack(data)


class TestReplayGames(unittest.TestCase):
    def test_replay_games(self):
        set_seed_everywhere(0)
        num_envs = 20
        env = Stratego(num_envs=num_envs, traj_len_per_player=100, barrage=True)
        my_data = []
        cur_my_data = [
            [
                env.current_zero_board_strs[i],
                [],  # actions
                [],  # board strings
                [],  # info states
                [],  # legacy placeholder
                [],  # num moves
                [],  # piece ids
                [],  # unknown piece position
                [],  # unknown piece type
                [],  # unknown piece count
                [],  # unknown piece has moved
            ]
            for i in range(num_envs)
        ]

        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        fn = os.path.join(tmp_dir, "test_save_games.msgpack")
        if os.path.exists(fn):
            os.remove(fn)
        env.save_games(fn)

        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for _ in range(1005):
            env.sample_random_legal_action(action_tensor)
            for i in range(num_envs):
                cur_my_data[i][1].append(action_tensor[i].item())
                cur_my_data[i][2].append(env.current_board_strs[i])
                cur_my_data[i][3].append(env.current_infostate_tensor[i].cpu())
                cur_my_data[i][5].append(env.current_num_moves[i].cpu())
                cur_my_data[i][6].append(env.current_piece_ids[i].cpu())
                cur_my_data[i][7].append(env.current_unknown_piece_position_onehot[i].cpu())
                cur_my_data[i][8].append(env.current_unknown_piece_type_onehot[i].cpu())
                cur_my_data[i][9].append(env.current_unknown_piece_counts[i])
                cur_my_data[i][10].append(env.current_unknown_piece_has_moved[i])
            env.apply_actions(action_tensor)
            for i in range(num_envs):
                if env.current_has_just_reset[i]:
                    my_data.append(cur_my_data[i])
                    cur_my_data[i] = [
                        env.current_zero_board_strs[i],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                        [],
                    ]
        env.stop_saving_games()
        self.assertTrue(os.path.exists(fn))
        with open(fn, "rb") as f:
            data = msgpack.load(f)
        os.remove(fn)
        os.rmdir(tmp_dir)
        my_data.sort(key=lambda x: (tuple(x[0]), tuple(x[1])))
        data.sort(key=lambda x: (tuple(x[0]), tuple(x[1])))
        data = data[: num_envs * (len(data) // num_envs)]
        my_data = my_data[: num_envs * (len(my_data) // num_envs)]
        for i in range(len(data)):
            assert tuple(data[i][0]) == tuple(my_data[i][0])
        init_boards = [game[0][::2] for game in data]
        env.change_reset_behavior_to_replay(init_boards)
        env.reset()
        action_carousel = ActionCarousel([game[1] for game in data], num_envs)
        string_carousel = StringCarousel([game[2] for game in my_data], num_envs)
        infostate_carousel = DataCarousel([game[3] for game in my_data], num_envs)
        num_moves_carousel = DataCarousel([game[5] for game in my_data], num_envs)
        piece_ids_carousel = DataCarousel([game[6] for game in my_data], num_envs)
        unknown_piece_position_carousel = DataCarousel([game[7] for game in my_data], num_envs)
        unknown_piece_type_carousel = DataCarousel([game[8] for game in my_data], num_envs)
        unknown_piece_count_carousel = DataCarousel([game[9] for game in my_data], num_envs)
        unknown_piece_has_moved_carousel = DataCarousel([game[10] for game in my_data], num_envs)
        for t in range(2000):
            self.assertEqual(tuple(env.current_board_strs), tuple(string_carousel.get_strings(t)))
            self.assertTrue(
                torch.allclose(
                    env.current_infostate_tensor.cpu(), infostate_carousel.get_data_tensor(t)
                )
            )
            self.assertTrue(
                torch.allclose(env.current_num_moves.cpu(), num_moves_carousel.get_data_tensor(t))
            )
            self.assertTrue(
                torch.allclose(env.current_piece_ids.cpu(), piece_ids_carousel.get_data_tensor(t))
            )
            self.assertTrue(
                torch.allclose(
                    env.current_unknown_piece_position_onehot.cpu(),
                    unknown_piece_position_carousel.get_data_tensor(t),
                )
            )
            self.assertTrue(
                torch.allclose(
                    env.current_unknown_piece_type_onehot.cpu(),
                    unknown_piece_type_carousel.get_data_tensor(t),
                )
            )
            self.assertTrue(
                torch.allclose(
                    env.current_unknown_piece_counts,
                    unknown_piece_count_carousel.get_data_tensor(t),
                )
            )
            self.assertTrue(
                torch.allclose(
                    env.current_unknown_piece_has_moved,
                    unknown_piece_has_moved_carousel.get_data_tensor(t),
                )
            )
            action_tensor = action_carousel.get_action_tensor(t)
            env.apply_actions(action_tensor)


if __name__ == "__main__":
    unittest.main()
