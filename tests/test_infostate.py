import unittest

import torch

from pyengine.core.env import Stratego


p0_labels = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "B"]
p1_labels = ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "N"]
label_by_player = [p0_labels, p1_labels]

MY_IS_HIDDEN_CHANNEL = 36
OPPONENT_IS_HIDDEN_CHANNEL = 37
EMPTY_CHANNEL = 38
MY_HAS_MOVED = 39
OPPONENT_HAS_MOVED = 40


def get_empty_positions(env):
    positions = torch.zeros(env.num_envs, 10, 10, device="cuda", dtype=torch.bool)
    for b in range(env.num_envs):
        board_str = env.current_board_strs[b]
        for loc in range(100):
            abs_loc = 2 * (99 - loc) if env.current_player == 1 else 2 * loc
            piece_val = board_str[abs_loc]
            if piece_val == "a":
                positions[b, loc // 10, loc % 10] = True
    return positions


def get_all_piece_positions(env):
    positions = torch.zeros(env.num_envs, 10, 10, device="cuda", dtype=torch.bool)
    for b in range(env.num_envs):
        board_str = env.current_board_strs[b]
        for loc in range(100):
            piece_val = board_str[2 * loc].upper()
            if piece_val in label_by_player[0] + label_by_player[1]:
                positions[b, loc // 10, loc % 10] = True
    return positions


def get_abs_piece_positions(env, player, hidden=False):
    positions = torch.zeros(env.num_envs, 10, 10, device="cuda", dtype=torch.bool)
    for b in range(env.num_envs):
        board_str = env.current_board_strs[b]
        for loc in range(100):
            piece_val = board_str[2 * loc]
            if not hidden:
                piece_val = piece_val.upper()
            if piece_val in label_by_player[player]:
                positions[b, loc // 10, loc % 10] = True
    return positions


def get_piece_positions(env, player, hidden=False):
    positions = torch.zeros(env.num_envs, 10, 10, device="cuda", dtype=torch.bool)
    for b in range(env.num_envs):
        board_str = env.current_board_strs[b]
        for loc in range(100):
            abs_loc = 2 * (99 - loc) if env.current_player == 1 else 2 * loc
            piece_val = board_str[abs_loc]
            if not hidden:
                piece_val = piece_val.upper()
            if piece_val in label_by_player[player]:
                positions[b, loc // 10, loc % 10] = True
    return positions


def get_hidden_piece_counts(env, player):
    piece_counts = torch.zeros(env.num_envs, 12, device="cuda")
    labels = label_by_player[player]
    for b in range(env.num_envs):
        board_str = env.current_board_strs[b]
        for loc in range(0, 200, 2):
            piece_val = board_str[loc]
            if piece_val in labels:
                piece_counts[b, labels.index(piece_val)] += 1
    return piece_counts


def get_probabilities(env, player):
    probabilities = torch.zeros(env.num_envs, 12, 10, 10, device="cuda", dtype=torch.float)
    piece_counts = get_hidden_piece_counts(env, player)
    # First 10 pieces are movable
    normalized_movable_piece_counts = piece_counts[:, :10] / piece_counts[:, :10].sum(
        dim=-1, keepdim=True
    ).clamp(min=1)
    # Last two pieces are immovable
    n_immovable_pieces = piece_counts[:, 10:12].sum(dim=-1)
    if player != env.current_player:
        is_hidden = env.current_infostate_tensor[:, OPPONENT_IS_HIDDEN_CHANNEL]
    else:
        is_hidden = env.current_infostate_tensor[:, MY_IS_HIDDEN_CHANNEL]
    if player == env.current_player:
        has_moved = env.current_infostate_tensor[:, MY_HAS_MOVED] * is_hidden
    else:
        has_moved = env.current_infostate_tensor[:, OPPONENT_HAS_MOVED] * is_hidden
    n_not_has_moved = piece_counts.sum(dim=-1) - has_moved.sum(dim=-1).sum(dim=-1)
    piece_positions = get_piece_positions(env, player)
    for b in range(piece_positions.shape[0]):
        for loc in range(100):
            if piece_positions[b, loc // 10, loc % 10]:
                # See https://www.overleaf.com/9543495711gpwzyjkbgnjp#507d1c for posterior derivation
                if has_moved[b, loc // 10, loc % 10]:
                    probabilities[b, :10, loc // 10, loc % 10] = normalized_movable_piece_counts[b]
                else:
                    probabilities[b, :10, loc // 10, loc % 10] = normalized_movable_piece_counts[
                        b
                    ] * (1 - n_immovable_pieces[b] / n_not_has_moved[b])
                    probabilities[b, 10:12, loc // 10, loc % 10] = (
                        piece_counts[b, 10:12] / n_not_has_moved[b]
                    )
    for b in range(is_hidden.shape[0]):
        for loc in range(100):
            if (not is_hidden[b, loc // 10, loc % 10]) and piece_positions[b, loc // 10, loc % 10]:
                probabilities[b, :, loc // 10, loc % 10] = env.current_piece_type_onehot[
                    b, loc // 10, loc % 10, :12
                ].float()
    for b in range(piece_counts.shape[0]):
        for loc in range(100):
            if piece_positions[b, loc // 10, loc % 10]:
                assert torch.allclose(
                    probabilities[b, :, loc // 10, loc % 10].sum(),
                    torch.tensor(1, device="cuda", dtype=torch.float),
                )
    return probabilities


class InfostateTest(unittest.TestCase):
    def test_channels(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=1,
        )
        desc = env.INFOSTATE_CHANNEL_DESCRIPTION
        self.assertEqual(desc[MY_IS_HIDDEN_CHANNEL], "our_hidden_bool")
        self.assertEqual(desc[OPPONENT_IS_HIDDEN_CHANNEL], "their_hidden_bool")
        self.assertEqual(desc[EMPTY_CHANNEL], "empty_bool")
        self.assertEqual(desc[MY_HAS_MOVED], "our_moved_bool")
        self.assertEqual(desc[OPPONENT_HAS_MOVED], "their_moved_bool")

    def test_player_piece_types(self):
        for full_info in [True, False]:
            for barrage in [True, False]:
                env = Stratego(
                    num_envs=1,
                    traj_len_per_player=1,
                    full_info=full_info,
                    barrage=barrage,
                )
                action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

                for _ in range(50):
                    env.sample_random_legal_action(action_tensor)
                    env.apply_actions(action_tensor)
                    # Piece types are first 12 channels
                    infostate_piece_types = env.current_infostate_tensor[:, :12]
                    for b in range(infostate_piece_types.shape[0]):
                        for loc in range(
                            infostate_piece_types.shape[2] * infostate_piece_types.shape[3]
                        ):
                            # Board state uses player invariant indexing
                            board_str_loc = 2 * (99 - loc) if env.current_player == 1 else 2 * loc
                            if (
                                env.current_board_strs[b][board_str_loc].upper()
                                in label_by_player[env.current_player]
                            ):  # If board_str says we have piece in `loc`, assert value equality
                                board_str_value = label_by_player[env.current_player].index(
                                    env.current_board_strs[b][board_str_loc].upper()
                                )
                                # Check infostate says we have one piece in `loc`
                                self.assertTrue(
                                    torch.allclose(
                                        infostate_piece_types[b, :, loc // 10, loc % 10].sum(),
                                        torch.tensor(1, device="cuda", dtype=torch.float),
                                    )
                                )
                                # Check that piece value agrees with board state
                                self.assertTrue(
                                    torch.allclose(
                                        infostate_piece_types[b, :, loc // 10, loc % 10].argmax(
                                            dim=-1
                                        ),
                                        torch.tensor(board_str_value, device="cuda"),
                                    )
                                )
                            else:  # Otherwise, check infostate agrees there's no piece
                                self.assertTrue(
                                    torch.allclose(
                                        infostate_piece_types[b, :, loc // 10, loc % 10],
                                        torch.tensor(0, device="cuda", dtype=torch.float),
                                    )
                                )

    def test_piece_probabilities(self):
        for full_info in [True, False]:
            for barrage in [True, False]:
                env = Stratego(
                    num_envs=1,
                    barrage=barrage,
                    traj_len_per_player=1,
                    full_info=full_info,
                )
                action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

                for t in range(300):
                    env.sample_random_legal_action(action_tensor)
                    env.apply_actions(action_tensor)
                    # Opponent probabilities are channels 12-23 (inclusive)
                    infostate_opponent_probabilities = env.current_infostate_tensor[:, 12:24]
                    opponent_probabilities = get_probabilities(env, int(not env.current_player))
                    self.assertTrue(
                        torch.allclose(infostate_opponent_probabilities, opponent_probabilities)
                    )
                    # Our probabilities are channels 24-35 (inclusive)
                    infostate_our_probabilities = env.current_infostate_tensor[:, 24:36]
                    our_probabilities = get_probabilities(env, env.current_player)
                    self.assertTrue(torch.allclose(infostate_our_probabilities, our_probabilities))

    def test_hidden_pieces(self):
        for full_info in [True, False]:
            for barrage in [True, False]:
                env = Stratego(
                    num_envs=1,
                    barrage=barrage,
                    traj_len_per_player=1,
                    full_info=full_info,
                )
                action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

                for t in range(300):
                    env.sample_random_legal_action(action_tensor)
                    env.apply_actions(action_tensor)
                    # Our hidden pieces is channel 36
                    infostate_our_hidden = env.current_infostate_tensor[:, MY_IS_HIDDEN_CHANNEL]
                    our_hidden = get_piece_positions(env, env.current_player, True).float()
                    self.assertTrue(torch.allclose(infostate_our_hidden, our_hidden))
                    # Opponent hidden pieces is channel 37
                    infostate_opponent_hidden = env.current_infostate_tensor[
                        :, OPPONENT_IS_HIDDEN_CHANNEL
                    ]
                    opponent_hidden = get_piece_positions(
                        env, int(not env.current_player), True
                    ).float()
                    self.assertTrue(torch.allclose(infostate_opponent_hidden, opponent_hidden))

    def test_empty_cells(self):
        for full_info in [True, False]:
            for barrage in [True, False]:
                env = Stratego(
                    num_envs=1,
                    barrage=barrage,
                    traj_len_per_player=1,
                    full_info=full_info,
                )
                action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
                for t in range(300):
                    env.sample_random_legal_action(action_tensor)
                    env.apply_actions(action_tensor)
                    # Empty cells is channel 38
                    infostate_empty = env.current_infostate_tensor[:, EMPTY_CHANNEL]
                    empty = get_empty_positions(env).float()
                    self.assertTrue(torch.allclose(infostate_empty, empty))

    def test_pieces_moved(self):
        for full_info in [True, False]:
            for barrage in [True, False]:
                env = Stratego(
                    num_envs=1,
                    barrage=barrage,
                    traj_len_per_player=100,
                    full_info=full_info,
                )
                action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
                terminated = torch.zeros(env.num_envs, dtype=torch.bool, device="cuda")
                moved_pieces = torch.zeros(env.num_envs, 10, 10, dtype=torch.bool, device="cuda")
                old_boards = env.current_board_strs
                for t in range(300):
                    env.sample_random_legal_action(action_tensor)
                    env.apply_actions(action_tensor)
                    terminated = torch.logical_or(terminated, env.current_is_terminal)
                    boards = env.current_board_strs
                    positions = get_all_piece_positions(env)
                    diffs = torch.stack(
                        [
                            torch.tensor(
                                [
                                    p.upper() != p_.upper()
                                    for p, p_ in zip(old_boards[b][::2], boards[b][::2])
                                ],
                                dtype=torch.bool,
                                device="cuda",
                            ).view(10, 10)
                            for b in range(env.num_envs)
                        ]
                    )
                    moved_pieces = torch.logical_and(
                        positions, torch.logical_or(diffs, moved_pieces)
                    )
                    p0_positions = get_abs_piece_positions(env, 0)
                    p0_moved_pieces = torch.logical_and(moved_pieces, p0_positions)
                    p1_positions = get_abs_piece_positions(env, 1)
                    p1_moved_pieces = torch.logical_and(moved_pieces, p1_positions)
                    infostate_our_moved = env.current_infostate_tensor[:, MY_HAS_MOVED]
                    infostate_opp_moved = env.current_infostate_tensor[:, OPPONENT_HAS_MOVED]
                    if env.current_player == 0:
                        self.assertTrue(
                            torch.logical_or(
                                torch.isclose(infostate_our_moved, p0_moved_pieces.float())
                                .all(dim=-1)
                                .all(dim=-1),
                                terminated,
                            )
                        )
                        self.assertTrue(
                            torch.logical_or(
                                torch.isclose(infostate_opp_moved, p1_moved_pieces.float())
                                .all(dim=-1)
                                .all(dim=-1),
                                terminated,
                            )
                        )
                    if env.current_player == 1:
                        self.assertTrue(
                            torch.logical_or(
                                torch.isclose(
                                    infostate_our_moved,
                                    p1_moved_pieces.float().flip(-1).flip(-2),
                                )
                                .all(dim=-1)
                                .all(dim=-1),
                                terminated,
                            )
                        )
                        self.assertTrue(
                            torch.logical_or(
                                torch.isclose(
                                    infostate_opp_moved,
                                    p0_moved_pieces.float().flip(-1).flip(-2),
                                )
                                .all(dim=-1)
                                .all(dim=-1),
                                terminated,
                            )
                        )
                    old_boards = boards


if __name__ == "__main__":
    unittest.main()
