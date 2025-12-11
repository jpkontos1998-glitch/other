import unittest
import os

import torch

from pyengine.core.env import Stratego
from pyengine import utils


pystratego = utils.get_pystratego()


cwd = os.path.dirname(__file__)
continuous_chase_games = sorted(os.listdir(os.path.join(cwd, "continuous_chase_games_new")))
violations = {"initial_boards": [], "action_sequences": [], "fn": []}

for game_file in continuous_chase_games[:1]:  # Limit to first game for testing
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


def l2i(c):
    c = c.lower()
    if c == "b" or c == "n":
        return letter_to_index[c] - 1
    return letter_to_index[c]


def is_adjacent(src, dst):
    is_to_right = (src == dst + 1) and (dst % 10 < 9)
    is_to_left = (src == dst - 1) and (dst % 10 > 0)
    is_above = (src == dst + 10) and (dst < 90)
    is_below = (src == dst - 10) and (dst > 9)
    return is_to_right or is_to_left or is_above or is_below


def is_two_squares_away(src, dst):
    is_to_right = (src == dst + 2) and (dst % 10 < 8)
    is_to_left = (src == dst - 2) and (dst % 10 > 1)
    is_above = (src == dst + 20) and (dst < 80)
    is_below = (src == dst - 20) and (dst > 10)
    is_to_right_above = (src == dst + 11) and (dst % 10 < 9) and (dst < 90)
    is_to_left_above = (src == dst + 9) and (dst % 10 > 0) and (dst < 90)
    is_to_right_below = (src == dst - 9) and (dst % 10 < 9) and (dst > 9)
    is_to_left_below = (src == dst - 11) and (dst % 10 > 0) and (dst > 9)
    return (
        is_to_right
        or is_to_left
        or is_above
        or is_below
        or is_to_right_above
        or is_to_left_above
        or is_to_right_below
        or is_to_left_below
    )


def is_defender_dies(attacker, defender):
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    if def_val == 10:  # flag
        return True
    elif (atk_val == 0) and (def_val == 9):  # spy takes marsh
        return True
    elif (atk_val == 2) and (def_val == 11):  # minter takes bomb
        return True
    return def_val <= atk_val


def is_defender_wins(attacker, defender):
    if defender == "a":
        return False
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    if def_val == 10:
        return False
    elif (atk_val == 0) and (def_val == 9):  # spy takes marsh
        return False
    elif (atk_val == 2) and (def_val == 11):  # minter takes bomb
        return False
    return atk_val < def_val


def is_attacker_dies(attacker, defender):
    if defender == "a":
        return False
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    if def_val == 10:
        return False
    if (atk_val == 0) and (def_val == 9):  # spy takes marsh
        return False
    if (atk_val == 2) and (def_val == 11):  # minter takes bomb
        return False
    return atk_val <= def_val


def is_on_board(pos):
    return pos >= 0 and pos < 100


def get_one_square_away_positions(pos):
    positions = []
    for move in [-10, -1, +1, +10]:
        if is_adjacent(pos, pos + move) and is_on_board(pos + move):
            positions.append(pos + move)
    return positions


def get_two_squares_away_positions(pos):
    positions = []
    for move1 in [-10, -1, +1, +10]:
        for move2 in [-10, -1, +1, +10]:
            if is_two_squares_away(pos, pos + move1 + move2) and is_on_board(pos + move1 + move2):
                positions.append(pos + move1 + move2)
    return positions


class MyProtectTracker:
    def __init__(self, player):
        self.protected_info = {i: torch.zeros(13, device="cuda") for i in range(40)}
        self.protected_against_info = {i: torch.zeros(13, device="cuda") for i in range(40)}
        self.was_protected_by_info = {i: torch.zeros(13, device="cuda") for i in range(40)}
        self.was_protected_against_info = {i: torch.zeros(13, device="cuda") for i in range(40)}
        self.player = player
        self.our_letters = letters_pl0 if player == 0 else letters_pl1
        self.opp_letters = letters_pl1 if player == 0 else letters_pl0

    def update(self, board_str, piece_ids, abs_move, last_abs_move):
        if last_abs_move is None:
            piece_ids = piece_ids[0].flatten()
            src, dst = abs_move[0]
            if is_adjacent(src, dst) or is_two_squares_away(src, dst):
                self.protected_info[piece_ids[src].item()][-2] = 1
                self.protected_against_info[piece_ids[src].item()][-1] = 1
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

        # Check if moving piece revealed opponent
        if board_str[dst].lower() in self.opp_letters and is_defender_wins(
            board_str[src], board_str[dst]
        ):
            for our_pos in get_two_squares_away_positions(dst):
                if board_str[our_pos].lower() not in self.our_letters:
                    continue
                for protectee_pos in get_one_square_away_positions(our_pos):
                    if board_str[protectee_pos] == "_":
                        continue
                    if board_str[protectee_pos].lower() in self.opp_letters:
                        continue
                    if not (
                        is_adjacent(protectee_pos, our_pos) and is_adjacent(protectee_pos, dst)
                    ):
                        continue
                    our_pos_id = piece_ids[our_pos].item()
                    self.protected_against_info[our_pos_id][l2i(board_str[dst].lower())] = 1
                    if board_str[protectee_pos] != "a":
                        protectee_id = piece_ids[protectee_pos].item()
                        self.was_protected_against_info[protectee_id][
                            l2i(board_str[dst].lower())
                        ] = 1
        # Check if moving piece cleared open cell for protection
        for opp_pos in get_one_square_away_positions(src):
            if opp_pos == dst and not is_defender_wins(board_str[src], board_str[dst]):
                continue
            if board_str[opp_pos] == "_":
                continue
            if board_str[opp_pos].lower() not in self.opp_letters:
                continue
            for protector_pos in get_one_square_away_positions(src):
                if not is_two_squares_away(protector_pos, opp_pos):
                    continue
                if board_str[protector_pos].lower() not in self.our_letters:
                    continue
                self.protected_info[piece_ids[protector_pos].item()][-2] = 1
                if board_str[opp_pos].isupper() and opp_pos != dst:
                    self.protected_against_info[piece_ids[protector_pos].item()][-1] = 1
                else:
                    self.protected_against_info[piece_ids[protector_pos].item()][
                        l2i(board_str[opp_pos])
                    ] = 1

        # Check if moving piece is actively protecting
        if not is_attacker_dies(board_str[src], board_str[dst]):
            for opp_pos in get_two_squares_away_positions(dst):
                # Only continue if there is an opponent piece
                if board_str[opp_pos].lower() not in self.opp_letters:
                    continue
                for move in [-10, -1, +1, +10]:
                    protectee_pos = opp_pos + move
                    if not (
                        is_adjacent(dst, protectee_pos) and is_adjacent(protectee_pos, opp_pos)
                    ):  # not connecting cell
                        continue
                    if board_str[protectee_pos] == "_":  # don't care about lakes
                        continue
                    if (
                        board_str[protectee_pos].lower() in self.opp_letters
                    ):  # can't protect opponent's piece
                        continue
                    if protectee_pos == src or board_str[protectee_pos] == "a":
                        protectee_idx = -2
                    elif board_str[protectee_pos].isupper():
                        protectee_idx = -1
                    else:
                        protectee_idx = l2i(board_str[protectee_pos])
                    self.protected_info[piece_id][protectee_idx] = 1
                    if board_str[opp_pos].isupper():
                        self.protected_against_info[piece_id][-1] = 1
                    else:
                        self.protected_against_info[piece_id][l2i(board_str[opp_pos])] = 1
                    if protectee_idx == -2:
                        continue
                    protectee_id = piece_ids[protectee_pos].item()
                    if board_str[src].isupper() and is_adjacent(src, dst):
                        self.was_protected_by_info[protectee_id][-1] = 1
                    else:
                        self.was_protected_by_info[protectee_id][l2i(board_str[src].lower())] = 1
                    if board_str[opp_pos].isupper():
                        self.was_protected_against_info[protectee_id][-1] = 1
                    else:
                        self.was_protected_against_info[protectee_id][l2i(board_str[opp_pos])] = 1

        # Check if moving piece is moving to protected position
        for opp_pos in get_one_square_away_positions(dst):
            if board_str[opp_pos].lower() not in self.opp_letters:
                continue
            for protector_pos in get_one_square_away_positions(dst):
                if not is_two_squares_away(protector_pos, opp_pos):
                    continue
                if board_str[protector_pos].lower() not in self.our_letters:
                    continue
                if is_attacker_dies(board_str[src], board_str[dst]):
                    continue
                if src == protector_pos:
                    continue
                protector_id = piece_ids[protector_pos].item()
                if board_str[dst].lower() in self.opp_letters or not is_adjacent(src, dst):
                    proctected_idx = l2i(board_str[src].lower())
                elif board_str[src].isupper():
                    proctected_idx = -1
                else:
                    proctected_idx = l2i(board_str[src])
                self.protected_info[protector_id][proctected_idx] = 1
                if board_str[protector_pos].isupper():
                    self.was_protected_by_info[piece_id][-1] = 1
                else:
                    self.was_protected_by_info[piece_id][l2i(board_str[protector_pos])] = 1
                if board_str[opp_pos].isupper():
                    self.protected_against_info[protector_id][-1] = 1
                    self.was_protected_against_info[piece_id][-1] = 1
                else:
                    self.protected_against_info[protector_id][l2i(board_str[opp_pos])] = 1
                    self.was_protected_against_info[piece_id][l2i(board_str[opp_pos])] = 1

        # No passive protection because theatener dies
        if board_str[last_dst].lower() not in self.opp_letters:
            return
        if dst == last_dst and is_defender_dies(board_str[src], board_str[last_dst]):
            return

        # Next, check if our pieces passively protected against opponent's last move
        for our_pos in get_two_squares_away_positions(last_dst):
            our_id = piece_ids[our_pos].item()
            # Only continue if there is one of our pieces
            if board_str[our_pos].lower() not in self.our_letters:
                continue
            if our_pos == src:  # protector moved away
                continue
            for move in [-10, -1, +1, +10]:
                protectee_pos = our_pos + move
                if not is_on_board(protectee_pos):
                    continue
                if protectee_pos == dst:
                    continue  # already checked
                protectee_id = piece_ids[protectee_pos].item()
                if not (
                    is_adjacent(protectee_pos, our_pos) and is_adjacent(protectee_pos, last_dst)
                ):  # not connecting cell
                    continue
                if board_str[protectee_pos] == "_":  # don't care about lakes
                    continue
                if (
                    board_str[protectee_pos].lower() in self.opp_letters
                ):  # can't protect opponent's piece
                    continue
                # If the to be protected piece is moving, then the to be protector is protecting an empty square.
                # The empty squares hasn't moved, so the protector must have moved.
                if protectee_pos == src:
                    continue
                if src == protectee_pos or board_str[protectee_pos] == "a":
                    protectee_idx = -2  # protecting empty cell
                elif board_str[protectee_pos].isupper():  # protectee is hidden
                    protectee_idx = -1
                else:
                    protectee_idx = l2i(board_str[protectee_pos])
                self.protected_info[our_id][protectee_idx] = 1
                if board_str[last_dst].isupper() and last_dst != dst:
                    self.protected_against_info[our_id][-1] = 1
                else:
                    self.protected_against_info[our_id][l2i(board_str[last_dst])] = 1
                if protectee_pos == src or board_str[protectee_pos] == "a":
                    continue
                if board_str[our_pos].isupper():
                    self.was_protected_by_info[protectee_id][-1] = 1
                else:
                    self.was_protected_by_info[protectee_id][l2i(board_str[our_pos])] = 1
                if board_str[last_dst].isupper() and last_dst != dst:
                    self.was_protected_against_info[protectee_id][-1] = 1
                else:
                    self.was_protected_against_info[protectee_id][l2i(board_str[last_dst])] = 1


class ProtectTest(unittest.TestCase):
    def test_protected_base(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_protected_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_protected_unknown") + 1
        self.assertEqual(s, 251)
        self.assertEqual(e, 264)
        s_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_protected_spy")
        e_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_protected_unknown") + 1
        self.assertEqual(s_their, 303)
        self.assertEqual(e_their, 316)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            trackers = [MyProtectTracker(0), MyProtectTracker(1)]
            last_abs_move = None
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            for a in action_seq[:-1]:
                we_protect_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in trackers[env.current_acting_player].protected_info:
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
                            (torch.zeros(13, device="cuda") == we_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[env.current_acting_player].protected_info[piece_id]
                            == we_protect_info[:, cell]
                        ).all()
                    )
                their_protect_info = env.current_infostate_tensor[0][s_their:e_their].flatten(
                    start_dim=1
                )
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
                            (torch.zeros(13, device="cuda") == their_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[1 - env.current_acting_player].protected_info[99 - piece_id]
                            == their_protect_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs, env.current_piece_ids, coords, last_abs_move
                )
                env.apply_actions(action_tensor)
                last_abs_move = coords

    def test_protect_against(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_protected_against_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_protected_against_unknown") + 1
        self.assertEqual(s, 264)
        self.assertEqual(e, 277)
        s_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_protected_against_spy")
        e_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_protected_against_unknown") + 1
        self.assertEqual(s_their, 316)
        self.assertEqual(e_their, 329)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            trackers = [MyProtectTracker(0), MyProtectTracker(1)]
            last_abs_move = None
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            for a in action_seq[:-1]:
                we_protect_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in trackers[env.current_acting_player].protected_info:
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
                            (torch.zeros(13, device="cuda") == we_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[env.current_acting_player].protected_against_info[piece_id]
                            == we_protect_info[:, cell]
                        ).all()
                    )
                their_protect_info = env.current_infostate_tensor[0][s_their:e_their].flatten(
                    start_dim=1
                )
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
                            (torch.zeros(13, device="cuda") == their_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[1 - env.current_acting_player].protected_against_info[
                                99 - piece_id
                            ]
                            == their_protect_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs, env.current_piece_ids, coords, last_abs_move
                )
                env.apply_actions(action_tensor)
                last_abs_move = coords

    def test_was_protected_by(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_was_protected_by_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_was_protected_by_unknown") + 1
        self.assertEqual(s, 277)
        self.assertEqual(e, 290)
        s_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_was_protected_by_spy")
        e_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_was_protected_by_unknown") + 1
        self.assertEqual(s_their, 329)
        self.assertEqual(e_their, 342)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            trackers = [MyProtectTracker(0), MyProtectTracker(1)]
            last_abs_move = None
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            for a in action_seq[:-1]:
                we_protect_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in trackers[env.current_acting_player].protected_info:
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
                            (torch.zeros(13, device="cuda") == we_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[env.current_acting_player].was_protected_by_info[piece_id]
                            == we_protect_info[:, cell]
                        ).all()
                    )
                their_protect_info = env.current_infostate_tensor[0][s_their:e_their].flatten(
                    start_dim=1
                )
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
                            (torch.zeros(13, device="cuda") == their_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[1 - env.current_acting_player].was_protected_by_info[
                                99 - piece_id
                            ]
                            == their_protect_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs, env.current_piece_ids, coords, last_abs_move
                )
                env.apply_actions(action_tensor)
                last_abs_move = coords

    def test_was_protected_against(self):
        utils.set_seed_everywhere(0)
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            full_info=False,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_was_protected_against_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_was_protected_against_unknown") + 1
        self.assertEqual(s, 290)
        self.assertEqual(e, 303)
        s_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_was_protected_against_spy")
        e_their = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_was_protected_against_unknown") + 1
        self.assertEqual(s_their, 342)
        self.assertEqual(e_their, 355)

        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for initial_board, action_seq in zip(
            violations["initial_boards"], violations["action_sequences"]
        ):
            trackers = [MyProtectTracker(0), MyProtectTracker(1)]
            last_abs_move = None
            env.change_reset_behavior_to_initial_board(initial_board)
            env.reset()
            for a in action_seq[:-1]:
                we_protect_info = env.current_infostate_tensor[0][s:e].flatten(start_dim=1)
                for piece_id in trackers[env.current_acting_player].protected_info:
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
                            (torch.zeros(13, device="cuda") == we_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[env.current_acting_player].was_protected_against_info[piece_id]
                            == we_protect_info[:, cell]
                        ).all()
                    )
                their_protect_info = env.current_infostate_tensor[0][s_their:e_their].flatten(
                    start_dim=1
                )
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
                            (torch.zeros(13, device="cuda") == their_protect_info[:, cell]).all()
                        )
                        continue
                    self.assertTrue(
                        (
                            trackers[1 - env.current_acting_player].was_protected_against_info[
                                99 - piece_id
                            ]
                            == their_protect_info[:, cell]
                        ).all()
                    )
                action_tensor[:] = a
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                trackers[env.current_acting_player].update(
                    env.current_board_strs, env.current_piece_ids, coords, last_abs_move
                )
                env.apply_actions(action_tensor)
                last_abs_move = coords


if __name__ == "__main__":
    unittest.main()
