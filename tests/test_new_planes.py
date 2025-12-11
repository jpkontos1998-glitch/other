import unittest

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego, set_seed_everywhere

pystratego = get_pystratego()

not_pl1_letters = ["c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "b", "a"]
not_pl0_letters = ["o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "n", "a"]

our_type_letters_pl0 = ["c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "b"] + [
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "B",
]
our_type_letters_pl1 = ["o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "n"] + [
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "N",
]

hidden_pl1 = ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "N"]
hidden_pl0 = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "B"]


class NewPlaneTests(unittest.TestCase):
    def test_their_visible_types_pl0(self):
        """Cell value is 1 / (piece.type + 1) if piece is visible else -1"""
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "their_visible_types" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl0_their_vis = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s.islower() and s not in not_pl1_letters)
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10)
            piece_types = env.current_piece_type_onehot.int().argmax(dim=-1)
            if env.current_player == 1:
                piece_types = piece_types.flip(-1, -2)
            their_vis = piece_types * mask.int()
            # Convert to backend representation
            their_vis = 1 / (their_vis.float() + 1)
            their_vis[~mask] = -1
            pl0_their_vis = torch.cat((pl0_their_vis, their_vis.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl0_their_vis = torch.zeros_like(pl0_their_vis)

            # # Test equality
            if env.current_player == 0:
                their_vis_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(their_vis_backend, pl0_their_vis))

    def test_their_visible_types_pl1(self):
        """Cell value is 1 / (piece.type + 1) if piece is visible else -1"""
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "their_visible_types" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl1_their_vis = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s.islower() and s not in not_pl0_letters)
            # We have to flip to get player 1's perspective
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10).flip(-1, -2)
            piece_types = env.current_piece_type_onehot.int().argmax(dim=-1)
            if env.current_player == 0:
                piece_types = piece_types.flip(-1, -2)
            their_vis = piece_types * mask.int()
            # Convert to backend representation
            their_vis = 1 / (their_vis.float() + 1)
            their_vis[~mask] = -1
            pl1_their_vis = torch.cat((pl1_their_vis, their_vis.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl1_their_vis = torch.zeros_like(pl1_their_vis)

            # # Test equality
            if env.current_player == 1:
                their_vis_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(their_vis_backend, pl1_their_vis))

    def test_our_types_pl0(self):
        """Cell value is 1 / (piece.type + 1) if piece is ours else -1"""
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "our_types" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl0_our_vis = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        players = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s in our_type_letters_pl0)
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10)
            piece_types = env.current_piece_type_onehot.int().argmax(dim=-1)
            if env.current_player == 1:
                piece_types = piece_types.flip(-1, -2)
            our_vis = piece_types * mask.int()
            # Convert to backend representation
            our_vis = 1 / (our_vis.float() + 1)
            our_vis[~mask] = -1
            pl0_our_vis = torch.cat((pl0_our_vis, our_vis.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]
            players.append(env.current_player)

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl0_our_vis = torch.zeros_like(pl0_our_vis)

            # # Test equality
            if env.current_player == 0:
                our_vis_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(our_vis_backend, pl0_our_vis))

    def test_our_types_pl1(self):
        """Cell value is 1 / (piece.type + 1) if piece is ours else -1"""
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "our_types" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl1_our_vis = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        players = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s in our_type_letters_pl1)
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10).flip(-1, -2)
            piece_types = env.current_piece_type_onehot.int().argmax(dim=-1)
            if env.current_player == 0:
                piece_types = piece_types.flip(-1, -2)
            our_vis = piece_types * mask.int()
            # Convert to backend representation
            our_vis = 1 / (our_vis.float() + 1)
            our_vis[~mask] = -1
            pl1_our_vis = torch.cat((pl1_our_vis, our_vis.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]
            players.append(env.current_player)

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl1_our_vis = torch.zeros_like(pl1_our_vis)

            # # Test equality
            if env.current_player == 1:
                our_vis_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(our_vis_backend, pl1_our_vis))

    def test_their_hidden_pl0(self):
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "their_hidden[" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl0_their_hidden = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        players = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s in hidden_pl1)
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10)
            pl0_their_hidden = torch.cat((pl0_their_hidden, mask.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]
            players.append(env.current_player)

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl0_their_hidden = torch.zeros_like(pl0_their_hidden)

            # # Test equality
            if env.current_player == 0:
                their_hidden_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(their_hidden_backend, pl0_their_hidden))

    def test_their_hidden_pl1(self):
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "their_hidden[" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl1_their_hidden = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        players = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s in hidden_pl0)
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10).flip(-1, -2)
            pl1_their_hidden = torch.cat((pl1_their_hidden, mask.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]
            players.append(env.current_player)

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl1_their_hidden = torch.zeros_like(pl1_their_hidden)

            # # Test equality
            if env.current_player == 1:
                their_hidden_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(their_hidden_backend, pl1_their_hidden))

    def test_our_hidden_pl0(self):
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "our_hidden[" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl0_our_hidden = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        players = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s in hidden_pl0)
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10)
            pl0_our_hidden = torch.cat((pl0_our_hidden, mask.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]
            players.append(env.current_player)

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl0_our_hidden = torch.zeros_like(pl0_our_hidden)

            # Test equality
            if env.current_player == 0:
                our_hidden_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(our_hidden_backend, pl0_our_hidden))

    def test_our_hidden_pl1(self):
        set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        plane_idx = []
        for i, description in enumerate(env.env.INFOSTATE_CHANNEL_DESCRIPTION):
            if "our_hidden[" in description:
                plane_idx.append(i)
        plane_idx = torch.tensor(plane_idx, device="cuda")
        pl1_our_hidden = torch.zeros((1, plane_idx.shape[0], 10, 10), device="cuda")
        boards = 30 * [None]
        players = 30 * [None]
        for i in range(1000):
            mask = []
            string = env.current_board_strs[0][::2]
            for s in string:
                mask.append(s in hidden_pl1)
            mask = torch.tensor(mask, device="cuda").view(1, 10, 10).flip(-1, -2)
            pl1_our_hidden = torch.cat((pl1_our_hidden, mask.unsqueeze(1)), dim=1)[:, 1:]
            boards.append(env.current_board_strs_pretty[0])
            boards = boards[1:]
            players.append(env.current_player)

            # Take action
            a = env.sample_random_legal_action()
            env.apply_actions(a)
            while env.current_is_terminal:
                a = env.sample_random_legal_action()
                env.apply_actions(a)
                pl1_our_hidden = torch.zeros_like(pl1_our_hidden)

            # # Test equality
            if env.current_player == 1:
                our_hidden_backend = env.current_infostate_tensor[:, plane_idx]
                self.assertTrue(torch.allclose(our_hidden_backend, pl1_our_hidden))

    def test_inception(self):
        num_envs = 8
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=100,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )
        actions = []
        data = []
        num_moves = []
        num_levels = 500
        snap = env.snapshot_state(env.current_step)
        snaps = []
        env.change_reset_behavior_to_env_state(snap)
        next_is_terminal = []
        for i in range(num_levels):
            snaps.append(env.current_state)
            a = env.sample_random_legal_action()
            actions.append(a)
            data.append(env.current_infostate_tensor)
            num_moves.append(env.current_num_moves)
            set_seed_everywhere(i)
            env.apply_actions(a)
            next_is_terminal.append(env.current_is_terminal)
        inception_break = False
        max_inception = 0
        cur_inception = 0
        for i in range(num_levels):
            if snap.terminated_since.max() == 0 and not inception_break:
                env.change_reset_behavior_to_env_state(snap)
                self.assertTrue(torch.allclose(num_moves[i], env.current_num_moves))
                self.assertTrue(torch.allclose(data[i], env.current_infostate_tensor))
                set_seed_everywhere(i)
                env.apply_actions(actions[i])
                snap = env.current_state
                cur_inception += 1
            elif snaps[i].terminated_since.max() == 0 and inception_break:
                env.change_reset_behavior_to_env_state(snaps[i])
                self.assertTrue(torch.allclose(num_moves[i], env.current_num_moves))
                self.assertTrue(torch.allclose(data[i], env.current_infostate_tensor))
                set_seed_everywhere(i)
                env.apply_actions(actions[i])
                snap = env.current_state
                inception_break = False
                max_inception = max(max_inception, cur_inception)
                cur_inception = 0
            else:
                inception_break = True
                a = env.sample_random_legal_action()
                actions.append(a)


if __name__ == "__main__":
    unittest.main()
