import unittest

import torch

from pyengine.core.env import Stratego

p0_labels = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "B"]
p1_labels = ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "N"]


def extract_unknown_piece_type_onehot(
    piece_type_onehot: torch.Tensor, is_unknown_piece: torch.Tensor, n_piece_per_player: int
) -> torch.Tensor:
    # Flatten board
    piece_type_onehot = piece_type_onehot.flatten(1, 2)
    is_unknown_piece = is_unknown_piece.flatten(1, 2)

    batch_size, _, n_pieces = piece_type_onehot.shape
    unknown_pieces = []

    # Iterate through each batch
    for b in range(batch_size):
        # Find the indices where is_unknown_piece is True for this batch
        unknown_indices = is_unknown_piece[b].nonzero(as_tuple=True)[0]

        # Select the corresponding columns from piece_type_onehot
        unknowns = piece_type_onehot[b, unknown_indices]

        # Pad if there are fewer than n_piece_per_player unknown pieces
        pad_size = n_piece_per_player - unknowns.shape[0]
        if pad_size > 0:
            padding = torch.zeros((pad_size, n_pieces), dtype=torch.bool, device="cuda")
            unknowns = torch.cat((unknowns, padding), dim=0)

        unknown_pieces.append(unknowns)

    # Stack to get the final tensor
    unknown_pieces = torch.stack(unknown_pieces)

    return unknown_pieces  # (batch_size, n_piece_per_player, n_pieces)


def extract_unknown_piece_position_onehot(
    is_unknown_piece: torch.Tensor, n_piece_per_player: int
) -> torch.Tensor:
    # Flatten board
    is_unknown_piece = is_unknown_piece.flatten(1, 2)

    batch_size, n_positions = is_unknown_piece.shape
    unknown_positions = []

    # Iterate through each batch
    for b in range(batch_size):
        # Find the indices where is_unknown_piece is True for this batch
        unknown_indices = is_unknown_piece[b].nonzero(as_tuple=True)[0]
        assert unknown_indices.shape[0] <= n_piece_per_player

        unknown_positions_ = torch.zeros(
            n_piece_per_player, n_positions, dtype=torch.bool, device="cuda"
        )
        for i, k in enumerate(unknown_indices):
            unknown_positions_[i, k] = True

        unknown_positions.append(unknown_positions_)

    # Stack to get the final tensor
    unknown_positions_tensor = torch.stack(unknown_positions)

    return unknown_positions_tensor  # (batch_size, n_piece_per_player, n_pieces)


def extract_unknown_has_moved(has_moved, current_is_unknown, n_piece_per_player) -> torch.Tensor:
    batch_size = has_moved.shape[0]
    unknown_has_moved = torch.zeros(batch_size, n_piece_per_player, dtype=torch.bool, device="cuda")
    for b in range(batch_size):
        i = 0
        for row in range(10):
            for col in range(10):
                if current_is_unknown[b, row, col]:
                    unknown_has_moved[b, i] = has_moved[b, row, col]
                    i += 1
    return unknown_has_moved


class GenMethodsTest(unittest.TestCase):
    def test_unknown_cells(self):
        # Test that perfect info games have no unknown cells
        env = Stratego(num_envs=1024, traj_len_per_player=100, full_info=True)
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        for _ in range(50):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        unknown_cells = env.current_is_unknown_piece
        assert unknown_cells.sum() == 0

        # Test that unknown in partial info games are equivalent to board_str info
        env = Stratego(num_envs=50, traj_len_per_player=100, full_info=False)
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        for _ in range(50):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        # Test for p0
        assert env.current_player == 0
        unknown_cells = env.current_is_unknown_piece
        for i in range(env.num_envs):
            for j in range(100):
                if unknown_cells[i].flatten()[j]:
                    self.assertTrue(env.current_board_strs[i][2 * j].isupper())
                if env.current_board_strs[i][2 * j] in p1_labels:
                    if env.current_board_strs[i][2 * j].isupper():
                        self.assertTrue(unknown_cells[i].flatten()[j])

        for _ in range(3):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        # Test for p1
        assert env.current_player == 1
        unknown_cells = env.current_is_unknown_piece
        for i in range(env.num_envs):
            for j in range(100):
                if unknown_cells[i].flatten()[j]:
                    self.assertTrue(env.current_board_strs[i][198 - (2 * j)].isupper())
                if env.current_board_strs[i][198 - (2 * j)] in p0_labels:
                    if env.current_board_strs[i][198 - (2 * j)].isupper():
                        self.assertTrue(unknown_cells[i].flatten()[j])

    def test_compute_piece_types(self):
        env = Stratego(num_envs=50, traj_len_per_player=100, full_info=False)
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        for _ in range(50):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        piece_types_one_hot = env.current_piece_type_onehot
        unknown_cells = env.current_is_unknown_piece

        # Test for p0
        assert env.current_player == 0
        for i in range(env.num_envs):
            for j in range(100):
                if env.current_board_strs[i][2 * j] in p1_labels or unknown_cells[i].flatten()[j]:
                    piece_id = piece_types_one_hot[i].view(-1, 14)[j].int().argmax()
                    self.assertEqual(env.current_board_strs[i][2 * j], p1_labels[piece_id])

        for _ in range(3):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        piece_types_one_hot = env.current_piece_type_onehot
        unknown_cells = env.current_is_unknown_piece

        # Test for p1
        assert env.current_player == 1
        for i in range(env.num_envs):
            for j in range(100):
                if (
                    env.current_board_strs[i][198 - (2 * j)] in p0_labels
                    or unknown_cells[i].flatten()[j]
                ):
                    piece_id = piece_types_one_hot[i].view(-1, 14)[j].int().argmax()
                    self.assertEqual(env.current_board_strs[i][198 - (2 * j)], p0_labels[piece_id])

    def test_unknown_piece_type_onehot(self):
        for barrage in [True, False]:
            env = Stratego(num_envs=50, traj_len_per_player=100, full_info=False, barrage=barrage)
            action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

            for _ in range(50):
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
                unknown_piece_type_onehot = extract_unknown_piece_type_onehot(
                    env.current_piece_type_onehot,
                    env.current_is_unknown_piece,
                    env.n_piece_per_player,
                )
                unknown_piece_type_onehot_batched = env.current_unknown_piece_type_onehot
                self.assertTrue(
                    torch.allclose(unknown_piece_type_onehot, unknown_piece_type_onehot_batched)
                )

    def test_unknown_piece_position_onehot(self):
        for barrage in [True, False]:
            env = Stratego(num_envs=50, traj_len_per_player=100, full_info=False, barrage=barrage)
            action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

            for _ in range(50):
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
                unknown_piece_position_onehot = extract_unknown_piece_position_onehot(
                    env.current_is_unknown_piece, env.n_piece_per_player
                )
                unknown_piece_position_onehot_batched = env.current_unknown_piece_position_onehot
                self.assertTrue(
                    torch.allclose(
                        unknown_piece_position_onehot, unknown_piece_position_onehot_batched
                    )
                )

    def test_unknown_piece_has_moved(self):
        for barrage in [True, False]:
            env = Stratego(num_envs=50, traj_len_per_player=100, full_info=False, barrage=barrage)
            action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
            for _ in range(50):
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
                has_moved = env.current_infostate_tensor[
                    :, env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_moved_bool")
                ].bool()
                unknown_has_moved = extract_unknown_has_moved(
                    has_moved, env.current_is_unknown_piece, env.n_piece_per_player
                )
                self.assertTrue(
                    torch.allclose(unknown_has_moved, env.current_unknown_piece_has_moved)
                )


if __name__ == "__main__":
    unittest.main()
