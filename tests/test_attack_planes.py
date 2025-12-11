import unittest
import os

import torch

from pyengine.core.env import Stratego
from pyengine import utils


pystratego = utils.get_pystratego()


cwd = os.path.dirname(__file__)
continuous_chase_games = sorted(os.listdir(os.path.join(cwd, "continuous_chase_games_new")))
violations = {"initial_boards": [], "action_sequences": [], "fn": []}

for game_file in continuous_chase_games[:1]:  # Limit to first game for faster testing
    with open(os.path.join(cwd, "continuous_chase_games_new", game_file), "r") as f:
        game_data = f.readlines()
        init_board = game_data[0].strip()
        assert len(init_board) == 100
        violations["initial_boards"].append(init_board)
        violations["action_sequences"].append([int(game_data[i]) for i in range(1, len(game_data))])
        violations["fn"].append(game_file)

letters_pl0 = ["c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "b"]
letters_pl1 = ["o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "n"]
flag_or_bomb_letters = ["m", "b", "y", "n"]
letter_to_index = {
    **{c: i for i, c in enumerate(letters_pl0)},
    **{c: i for i, c in enumerate(letters_pl1)},
}


def get_opp_adj_pieces(board_str, cell, player):
    my_pieces = letters_pl0 if player == 0 else letters_pl1
    pieces = []
    if cell < 90:
        pieces.append(board_str[cell + 10])
    if cell > 10:
        pieces.append(board_str[cell - 10])
    if (cell % 10) < 9:
        pieces.append(board_str[cell + 1])
    if (cell % 10) > 0:
        pieces.append(board_str[cell - 1])
    return [p for p in pieces if p.lower() not in (["a", "_"] + my_pieces)]


def is_adjacent(src, dst):
    is_to_right = (src == dst + 1) and (dst % 10 < 9)
    is_to_left = (src == dst - 1) and (dst % 10 > 0)
    is_above = (src == dst + 10) and (dst < 90)
    is_below = (src == dst - 10) and (dst > 9)
    return is_to_right or is_to_left or is_above or is_below


class MyThreatTracker:
    def __init__(self, player):
        self.info = {i: torch.zeros(11, device="cuda") for i in range(40)}
        self.player = player

    def update(self, board_str, piece_ids, abs_move):
        board_str = board_str[0][::2]
        if self.player == 1:
            # Flip absolute coordinates for player 1
            board_str = board_str[::-1]
        piece_ids = piece_ids[0].flatten()
        src, dst = abs_move[0]
        if self.player == 1:
            # Flip absolute coordinates for player 1
            src = 99 - src
            dst = 99 - dst

        piece_id = piece_ids[src].item()

        opp_adj_pieces = get_opp_adj_pieces(board_str, dst, self.player)
        for p in opp_adj_pieces:
            if p.isupper():
                self.info[piece_id][10] = 1  # 10 is hidden piece index
            elif p in flag_or_bomb_letters:
                continue  # we don't track threats to immovable pieces
            else:
                self.info[piece_id][letter_to_index[p.lower()]] = 1


class MyEvadeTracker:
    def __init__(self, player):
        self.info = {i: torch.zeros(11, device="cuda") for i in range(40)}
        self.player = player
        self.opp_letters = letters_pl1 if player == 0 else letters_pl0

    def update(self, board_str, piece_ids, abs_move, last_abs_move):
        if last_abs_move is None:
            return

        board_str = board_str[0][::2]
        if self.player == 1:
            # Flip absolute coordinates for player 1
            board_str = board_str[::-1]
        piece_ids = piece_ids[0].flatten()
        src, dst = abs_move[0]
        last_src, last_dst = last_abs_move[0]
        if self.player == 1:
            # Flip absolute coordinates for player 1
            src = 99 - src
            dst = 99 - dst
            last_src = 99 - last_src
            last_dst = 99 - last_dst

        piece_id = piece_ids[src].item()

        if is_adjacent(last_dst, src):  # opponent moved adjacent to us
            if board_str[last_dst].lower() in self.opp_letters:  # opponent still alive
                if board_str[last_dst].isupper():  # opponent is hidden
                    self.info[piece_id][10] = 1  # 10 is hidden piece index
                else:
                    self.info[piece_id][letter_to_index[board_str[last_dst].lower()]] = 1


class MyActivelyAdjacentTracker:
    def __init__(self, player):
        self.info = {i: torch.zeros(11, device="cuda") for i in range(40)}
        self.player = player
        self.opp_letters = letters_pl1 if player == 0 else letters_pl0

    def update(self, board_str, piece_ids, abs_move):
        board_str = board_str[0][::2]
        if self.player == 1:
            # Flip absolute coordinates for player 1
            board_str = board_str[::-1]
        piece_ids = piece_ids[0].flatten()
        src, dst = abs_move[0]
        if self.player == 1:
            # Flip absolute coordinates for player 1
            src = 99 - src
            dst = 99 - dst

        moving_piece_id = piece_ids[src].item()
        cell = torch.argwhere(piece_ids == moving_piece_id).item()
        above = cell - 10 if cell >= 10 else None
        below = cell + 10 if cell < 90 else None
        left = cell - 1 if cell % 10 > 0 else None
        right = cell + 1 if cell % 10 < 9 else None
        adj_cells = [above, below, left, right]
        for adj_cell in adj_cells:
            if adj_cell is None:
                continue
            if adj_cell == dst:  # moving piece
                continue
            cell_val = board_str[adj_cell]
            if cell_val.lower() in self.opp_letters:
                if cell_val.isupper():  # opponent is hidden
                    self.info[moving_piece_id][10] = 1  # 10 is hidden piece index
                elif cell_val in self.opp_letters[:10]:  # check is movable
                    self.info[moving_piece_id][letter_to_index[cell_val.lower()]] = 1

        for piece_id in piece_ids:
            if piece_id >= 40:  # not our piece
                continue
            if piece_id == moving_piece_id:  # special case for moving piece
                continue
            cell = torch.argwhere(piece_ids == piece_id)
            opp_adj_pieces = get_opp_adj_pieces(board_str, cell.item(), self.player)
            for p in opp_adj_pieces:
                if p.isupper():
                    self.info[piece_id.item()][10] = 1  # 10 is hidden piece index
                elif p in self.opp_letters[:10]:  # check is movable
                    self.info[piece_id.item()][letter_to_index[p.lower()]] = 1


class AttackPlanesTest(unittest.TestCase):
    def test_we_threatened(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("we_threatened_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("we_threatened_unknown") + 1
        self.assertEqual(s, 43)
        self.assertEqual(e, 54)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            trackers = [MyThreatTracker(0), MyThreatTracker(1)]
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            for a in action_seq[:-1]:
                we_threat_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in trackers[env.current_acting_player].info:
                    cell = torch.argwhere(env.current_piece_ids[0].flatten() == piece_id)
                    if cell.numel() == 0:  # piece not on board anymore
                        continue
                    cell = cell.item()
                    board_str = env.current_board_strs[0][::2]
                    if env.current_acting_player == 1:
                        # Flip absolute coordinates for player 1
                        board_str = board_str[::-1]
                    if board_str[cell].islower():  # piece is visible
                        self.assertTrue(
                            (torch.zeros(11, device="cuda") == we_threat_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[env.current_acting_player].info[piece_id]
                            == we_threat_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs, env.current_piece_ids, coords
                )
                env.apply_actions(action_tensor)

    def test_we_evaded(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("we_evaded_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("we_evaded_unknown") + 1
        self.assertEqual(s, 54)
        self.assertEqual(e, 65)
        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            trackers = [MyEvadeTracker(0), MyEvadeTracker(1)]
            last_abs_move = None
            for a in action_seq[:-1]:
                we_evade_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in trackers[env.current_acting_player].info:
                    cell = torch.argwhere(env.current_piece_ids[0].flatten() == piece_id)
                    if cell.numel() == 0:  # piece not on board anymore
                        continue
                    cell = cell.item()
                    board_str = env.current_board_strs[0][::2]
                    if env.current_acting_player == 1:
                        # Flip absolute coordinates for player 1
                        board_str = board_str[::-1]
                    if board_str[cell].islower():  # piece is visible
                        self.assertTrue(
                            (torch.zeros(11, device="cuda") == we_evade_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[env.current_acting_player].info[piece_id]
                            == we_evade_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs,
                    env.current_piece_ids,
                    coords,
                    last_abs_move,
                )
                last_abs_move = coords
                env.apply_actions(action_tensor)

    def test_we_actively_adjacent(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("we_actively_adj_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("we_actively_adj_unknown") + 1
        self.assertEqual(s, 65)
        self.assertEqual(e, 76)
        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            trackers = [MyActivelyAdjacentTracker(0), MyActivelyAdjacentTracker(1)]
            for a in action_seq[:-1]:
                we_actively_adj_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in trackers[env.current_acting_player].info:
                    cell = torch.argwhere(env.current_piece_ids[0].flatten() == piece_id)
                    if cell.numel() == 0:  # piece not on board anymore
                        continue
                    cell = cell.item()
                    board_str = env.current_board_strs[0][::2]
                    if env.current_acting_player == 1:
                        # Flip absolute coordinates for player 1
                        board_str = board_str[::-1]
                    if board_str[cell].islower():  # piece is visible
                        self.assertTrue(
                            (torch.zeros(11, device="cuda") == we_actively_adj_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[env.current_acting_player].info[piece_id]
                            == we_actively_adj_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs,
                    env.current_piece_ids,
                    coords,
                )
                env.apply_actions(action_tensor)

    def test_they_threatened(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("they_threatened_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("they_threatened_unknown") + 1
        self.assertEqual(s, 76)
        self.assertEqual(e, 87)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            trackers = [MyThreatTracker(0), MyThreatTracker(1)]
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            for a in action_seq[:-1]:
                they_threat_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in range(60, 100):
                    cell = torch.argwhere(env.current_piece_ids[0].flatten() == piece_id)
                    if cell.numel() == 0:  # piece not on board anymore
                        continue
                    cell = cell.item()
                    board_str = env.current_board_strs[0][::2]
                    if env.current_acting_player == 1:
                        # Flip absolute coordinates for player 1
                        board_str = board_str[::-1]
                    if board_str[cell].islower():  # piece is visible
                        self.assertTrue(
                            (torch.zeros(11, device="cuda") == they_threat_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[1 - env.current_acting_player].info[99 - piece_id]
                            == they_threat_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs, env.current_piece_ids, coords
                )
                env.apply_actions(action_tensor)

    def test_they_evaded(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("they_evaded_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("they_evaded_unknown") + 1
        self.assertEqual(s, 87)
        self.assertEqual(e, 98)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            last_abs_move = None
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            trackers = [MyEvadeTracker(0), MyEvadeTracker(1)]
            for a in action_seq[:-1]:
                they_evade_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in range(60, 100):
                    cell = torch.argwhere(env.current_piece_ids[0].flatten() == piece_id)
                    if cell.numel() == 0:  # piece not on board anymore
                        continue
                    cell = cell.item()
                    board_str = env.current_board_strs[0][::2]
                    if env.current_acting_player == 1:
                        # Flip absolute coordinates for player 1
                        board_str = board_str[::-1]
                    if board_str[cell].islower():  # piece is visible
                        self.assertTrue(
                            (torch.zeros(11, device="cuda") == they_evade_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[1 - env.current_acting_player].info[99 - piece_id]
                            == they_evade_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs, env.current_piece_ids, coords, last_abs_move
                )
                last_abs_move = coords
                env.apply_actions(action_tensor)

    def test_they_actively_adjacent(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("they_actively_adj_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("they_actively_adj_unknown") + 1
        self.assertEqual(s, 98)
        self.assertEqual(e, 109)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            trackers = [MyActivelyAdjacentTracker(0), MyActivelyAdjacentTracker(1)]
            for a in action_seq[:-1]:
                they_actively_adj_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in range(60, 100):
                    cell = torch.argwhere(env.current_piece_ids[0].flatten() == piece_id)
                    if cell.numel() == 0:  # piece not on board anymore
                        continue
                    cell = cell.item()
                    board_str = env.current_board_strs[0][::2]
                    if env.current_acting_player == 1:
                        # Flip absolute coordinates for player 1
                        board_str = board_str[::-1]
                    if board_str[cell].islower():  # piece is visible
                        self.assertTrue(
                            (
                                torch.zeros(11, device="cuda") == they_actively_adj_info[:, cell]
                            ).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[1 - env.current_acting_player].info[99 - piece_id]
                            == they_actively_adj_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs,
                    env.current_piece_ids,
                    coords,
                )
                env.apply_actions(action_tensor)


if __name__ == "__main__":
    unittest.main()
