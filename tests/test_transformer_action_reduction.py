import unittest

import torch

from pyengine.utils.constants import N_ACTION
from pyengine.core.env import Stratego
from pyengine.networks.utils import create_srcdst_to_env_action_index


def source_dest_tensor_to_action_tensor(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape == (100, 100), f"Expected tensor shape (100, 100), got {tensor.shape}"

    output_tensor = torch.zeros(18, 100)

    for piece_pos in range(100):
        piece_row, piece_col = piece_pos // 10, piece_pos % 10

        # Rows 0-8: setting the row (excluding piece row)
        for new_row in range(10):
            if new_row != piece_row:
                row_idx = new_row if new_row < piece_row else new_row - 1
                new_pos = new_row * 10 + piece_col
                output_tensor[row_idx, piece_pos] = tensor[piece_pos, new_pos]

        # Rows 9-17: setting the column (excluding piece column)
        for new_col in range(10):
            if new_col != piece_col:
                row_idx = 9 + (new_col if new_col < piece_col else new_col - 1)
                new_pos = piece_row * 10 + new_col
                output_tensor[row_idx, piece_pos] = tensor[piece_pos, new_pos]

    return output_tensor


def make_source_dest_tensor_to_action_tensor_indices() -> torch.Tensor:
    idx = torch.zeros(18 * 100, dtype=torch.int64)
    for i in range(100):
        piece_row, piece_col = i // 10, i % 10

        # Rows 0-8: setting the row (excluding piece row)
        for new_row in range(10):
            if new_row != piece_row:
                row_idx = new_row if new_row < piece_row else new_row - 1
                new_pos = new_row * 10 + piece_col
                idx[row_idx * 100 + i] = 100 * i + new_pos

        # Rows 9-17: setting the column (excluding piece column)
        for new_col in range(10):
            if new_col != piece_col:
                row_idx = 9 + (new_col if new_col < piece_col else new_col - 1)
                new_pos = piece_row * 10 + new_col
                idx[row_idx * 100 + i] = 100 * i + new_pos

    return idx


def reconstruct_full_tensor(reduced_tensor: torch.Tensor, excluded_indices: list) -> torch.Tensor:
    # Step 1: Set up the full board
    board_size = 10
    num_positions = board_size * board_size

    # Step 2: Create a list of valid positions
    valid_positions = [i for i in range(num_positions) if i not in excluded_indices]

    # Step 3: Initialize the full tensor with zeros
    full_tensor = torch.zeros(num_positions, num_positions)

    # Step 4: Fill in the full tensor with values from the reduced tensor
    for i, from_pos in enumerate(valid_positions):
        for j, to_pos in enumerate(valid_positions):
            full_tensor[from_pos, to_pos] = reduced_tensor[i, j]

    return full_tensor


class ActionReductionTest(unittest.TestCase):
    def test_basic_action_reduction(self):
        batch_size = 256
        for _ in range(2):
            x = torch.rand(batch_size, 100, 100)
            y = torch.stack([source_dest_tensor_to_action_tensor(x_) for x_ in x])
            idx = make_source_dest_tensor_to_action_tensor_indices()
            y2 = x.flatten(start_dim=1)[:, idx].view(batch_size, 18, 100)
            self.assertTrue(torch.allclose(y, y2))

    def test_adv_action_reduction(self):
        batch_size = 256
        excluded = [42, 43, 46, 47, 52, 53, 56, 57]
        n_remaining = 100 - len(excluded)
        for _ in range(2):
            x = torch.rand(batch_size, n_remaining, n_remaining)
            full = torch.stack([reconstruct_full_tensor(x_, excluded) for x_ in x])
            y = torch.stack([source_dest_tensor_to_action_tensor(f) for f in full])
            idx = create_srcdst_to_env_action_index(torch.tensor(excluded))
            y2 = torch.zeros(batch_size, 18, 100)
            valid_idx = idx != -1
            y2.view(batch_size, -1)[:, valid_idx] = x.flatten(start_dim=1)[:, idx[valid_idx]]
            self.assertTrue(torch.allclose(y, y2))

    def test_legal_action_consistency(self):
        lakes = [42, 43, 46, 47, 52, 53, 56, 57]
        env = Stratego(num_envs=1024, traj_len_per_player=100)
        idx = create_srcdst_to_env_action_index(torch.tensor(lakes))
        invalid_idx = idx == -1
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for _ in range(1000):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            legal_actions = env.current_legal_action_mask
            self.assertFalse(torch.any(legal_actions[:, invalid_idx]))

    def test_basic_adv_consistency(self):
        idx = create_srcdst_to_env_action_index(torch.tensor([]))
        idx2 = make_source_dest_tensor_to_action_tensor_indices()
        self.assertTrue(torch.all(idx == idx2))

    def test_axes_aligned(self):
        vals = []
        for i in range(100):
            for j in range(100):
                vals.append((i, j))
        idx = make_source_dest_tensor_to_action_tensor_indices()
        reduced = []
        for k in idx:
            i, j = vals[k]
            reduced.append((i, j))
        self.assertEqual(len(reduced), N_ACTION)
        for ell in range(len(reduced)):
            cell = ell % 100
            self.assertEqual(reduced[ell][0], cell)


if __name__ == "__main__":
    unittest.main()
