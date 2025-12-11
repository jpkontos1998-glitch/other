import unittest

import torch

from pyengine.networks.utils import create_srcdst_to_env_action_index


def make_move_tensor_to_action_tensor_indices() -> torch.Tensor:
    output_tensor = torch.zeros(18, 100)

    for piece_pos in range(100):
        piece_row, piece_col = piece_pos // 10, piece_pos % 10

        for new_row in range(10):
            if new_row != piece_row:
                row_idx = new_row if new_row < piece_row else new_row - 1
                move_len = abs(new_row - piece_row)
                move_dist_idx = move_len - 1
                if new_row > piece_row:  # No offset
                    move = move_dist_idx
                else:  # Offset by 9
                    move = move_dist_idx + 9
                output_tensor[row_idx, piece_pos] = 100 * move + piece_pos

        for new_col in range(10):
            if new_col != piece_col:
                col_idx = 9 + (new_col if new_col < piece_col else new_col - 1)
                move_len = abs(new_col - piece_col)
                move_dist_idx = move_len - 1
                if new_col > piece_col:  # Offset by 18
                    move = move_dist_idx + 18
                else:  # Offset by 27
                    move = move_dist_idx + 27
                output_tensor[col_idx, piece_pos] = 100 * move + piece_pos

    return output_tensor.long().view(-1)


def make_abs_tensor_to_action_tensor_indices() -> torch.Tensor:
    output_tensor = torch.zeros(18, 100)

    for piece_pos in range(100):
        piece_row, piece_col = piece_pos // 10, piece_pos % 10

        for new_row in range(10):
            if new_row != piece_row:
                row_idx = new_row if new_row < piece_row else new_row - 1
                output_tensor[row_idx, piece_pos] = 100 * new_row + piece_pos

        for new_col in range(10):
            if new_col != piece_col:
                col_idx = 9 + (new_col if new_col < piece_col else new_col - 1)
                output_tensor[col_idx, piece_pos] = 1000 + 100 * new_col + piece_pos

    return output_tensor.long().view(-1)


def construct_18x100_tensor():
    # Initialize the resultant tensor
    result = torch.zeros(18, 100)

    # Loop through each entry in the tensor
    for y in range(100):
        # Decode the current position
        row = y // 10
        col = y % 10
        for row_move in range(9):
            new_row = row_move if row_move < row else row_move + 1
            new_y = new_row * 10 + col
            result[row_move, y] = new_y
        for col_move in range(9):
            new_col = col_move if col_move < col else col_move + 1
            new_y = row * 10 + new_col
            result[col_move + 9, y] = new_y
    return result


def construct_20x100_tensor():
    # Initialize the resultant tensor
    result = torch.zeros(20, 100)

    # Loop through each entry in the tensor
    for x in range(20):
        for y in range(100):
            # Decode the current position
            row = y // 10
            col = y % 10

            # Determine the movement
            if x < 10:
                # x represents a change in the row
                new_row = x
                new_col = col
            else:
                # x represents a change in the column
                new_row = row
                new_col = x - 10

            # Convert back to 1D index
            new_y = new_row * 10 + new_col

            if y == new_y:
                # Not moving is invalid so assign -1
                result[x, y] = -1
            else:
                # Assign the value to the result tensor
                result[x, y] = new_y

    return result


def construct_36x100_tensor():
    # Initialize the resultant tensor
    result = torch.zeros(36, 100)

    # Loop through each entry in the tensor
    for y in range(100):
        # Decode the current position
        row = y // 10
        col = y % 10

        for x in range(36):
            if x < 9:  # increase in row
                move_len = x + 1  # Movements lengths can be in [1, 9]
                new_row = row + move_len
                new_col = col
            elif x < 18:  # decrease in row
                move_len = x + 1 - 9  # Movements lengths can be in [1, 9]
                new_row = row - move_len
                new_col = col
            elif x < 27:  # increase in column
                move_len = x + 1 - 18  # Movements lengths can be in [1, 9]
                new_row = row
                new_col = col + move_len
            else:  # decrease in column
                move_len = x + 1 - 27  # Movements lengths can be in [1, 9]
                new_row = row
                new_col = col - move_len

            if new_row < 0 or new_row >= 10 or new_col < 0 or new_col >= 10:
                result[x, y] = -1
                continue

            # Convert back to 1D index
            new_y = new_row * 10 + new_col
            # Assign the value to the result tensor
            result[x, y] = new_y

    return result


def construct_100x100_tensor():
    # Initialize the resultant tensor
    result = torch.zeros(100, 100)

    # Loop through each entry in the tensor
    for y in range(100):
        for x in range(100):
            # Decode the current position
            row = y // 10
            col = y % 10
            new_row = x // 10
            new_col = x % 10

            if (row != new_row) ^ (col != new_col):
                result[x, y] = x
            else:
                result[x, y] = -1
    return result


class ActionTransformTest(unittest.TestCase):
    def test_meta_100x100(self):
        tensor = construct_100x100_tensor()
        target_tensor = construct_18x100_tensor()
        for y in range(100):
            unique18 = torch.unique(target_tensor[:, y])
            x = tensor[:, y].view(-1)[tensor[:, y].view(-1) != -1]
            self.assertTrue(torch.all(unique18 == x.unique()))

    def test_100x100_tensor(self):
        # Permute here because the function uses (source, dest) format
        tensor = construct_100x100_tensor().permute(1, 0)
        target_tensor = construct_18x100_tensor()
        mask = create_srcdst_to_env_action_index(torch.tensor([]))
        reduced_tensor = tensor.reshape(-1)[mask]
        self.assertTrue(torch.all(reduced_tensor == target_tensor.view(-1)))

    def test_meta_36x100(self):
        tensor = construct_36x100_tensor()
        target_tensor = construct_18x100_tensor()
        for i in range(100):
            x = tensor[:, i].view(-1)[tensor[:, i].view(-1) != -1]
            unique18 = torch.unique(target_tensor[:, i])
            self.assertTrue(torch.all(unique18 == x.unique()))

    def test_36x100_tensor(self):
        tensor = construct_36x100_tensor()
        target_tensor = construct_18x100_tensor()
        mask = make_move_tensor_to_action_tensor_indices()
        reduced_tensor = tensor.reshape(-1)[mask]
        self.assertTrue(torch.all(reduced_tensor == target_tensor.view(-1)))

    def test_meta_20x100(self):
        tensor = construct_20x100_tensor()
        target_tensor = construct_18x100_tensor()
        for i in range(100):
            x = tensor[:, i].view(-1)[tensor[:, i].view(-1) != -1]
            unique18 = torch.unique(target_tensor[:, i])
            self.assertTrue(torch.all(unique18 == x.unique()))

    def test_20x100_tensor(self):
        tensor = construct_20x100_tensor()
        target_tensor = construct_18x100_tensor()
        mask = make_abs_tensor_to_action_tensor_indices()
        reduced_tensor = tensor.reshape(-1)[mask]
        self.assertTrue(torch.all(reduced_tensor == target_tensor.view(-1)))


if __name__ == "__main__":
    unittest.main()
