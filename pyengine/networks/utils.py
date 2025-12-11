from typing import Optional

import torch
import torch.nn as nn

from pyengine.utils.constants import (
    BOARD_LEN,
    LAKE_INDICES,
    N_ACTION,
    N_BOARD_CELL,
    N_OCCUPIABLE_CELL,
    MAX_N_POSSIBLE_DST,
)
from pyengine.utils.load_pystratego import get_pystratego

pystratego = get_pystratego()


def create_srcdst_to_env_action_index(
    excluded: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create index tensor for converting reduced src-dst action param to environment action param.

    We only need to run this once, so it need not be efficient.

    Args:
        excluded: Tensor of cells excluded from src-dst parameterization.

    Returns:
        (MAX_N_POSSIBLE_DST * N_BOARD_CELL,): Value at index for action in environment param is index for same action in reduced src-dst action param.
            Value -1 means action is not represented in reduced src-dst action param.
    """
    if excluded is None:
        excluded = torch.tensor([], dtype=torch.long)
    n_valid_cells = N_BOARD_CELL - excluded.numel()

    # Mapping from all cells to cells represented by parameterization
    valid_cells = [i for i in range(N_BOARD_CELL) if i not in excluded]
    full_to_reduced = {pos: idx for idx, pos in enumerate(valid_cells)}

    # Tensor of indices to return.
    idx = torch.full((MAX_N_POSSIBLE_DST, N_BOARD_CELL), -1, dtype=torch.long)

    # Iterate over cells for action source.
    for src in range(N_BOARD_CELL):
        # If the action is not reprented in reduced src-dst action param, leave value as -1.
        if src in excluded:
            continue

        # Compute source row and column.
        src_row, src_col = src // BOARD_LEN, src % BOARD_LEN

        # Fill in indices for row moves.
        for new_row in range(BOARD_LEN):
            # If action is not represented in environment param, skip.
            if new_row == src_row:
                continue
            dst = new_row * BOARD_LEN + src_col
            # If action is not represented in reduced src-dst action param, leave value as -1.
            if dst in excluded:
                continue
            movement_idx = (
                new_row if new_row < src_row else new_row - 1
            )  # -1 is to exclude the current cell.
            idx[movement_idx, src] = full_to_reduced[src] * n_valid_cells + full_to_reduced[dst]

        # Fill in indices for column moves.
        for new_col in range(BOARD_LEN):
            # If action is not represented in environment param, skip.
            if new_col == src_col:
                continue
            dst = src_row * BOARD_LEN + new_col
            # If action is not represented in reduced src-dst action param, leave value as -1.
            if dst in excluded:
                continue
            movement_idx = (
                (BOARD_LEN - 1)  # access column movements
                + (new_col if new_col < src_col else new_col - 1)
            )  # -1 is to exclude the current cell.
            idx[movement_idx, src] = full_to_reduced[src] * n_valid_cells + full_to_reduced[dst]

    return idx.view(-1)


class LogitConverter(nn.Module):
    """Utility for converting network action param to environment action param.

    The network outputs action logits in a src-dst parameterization (excluding lake moves),
    whereas the backend uses a src-displacement parameterization.
    """

    def __init__(self):
        super().__init__()
        src_dst_to_env_action = create_srcdst_to_env_action_index(torch.tensor(LAKE_INDICES))
        # (N_ACTION): True for indices in backend action parameterization for which neither src nor dst is a lake.
        self.register_buffer("not_lake_move", src_dst_to_env_action != -1)
        # (N_ACTION - N_LAKE_ACTIONS): Value at index of environment action i is the environment action is
        # i-th action in a src-dst action parameterization (excluding lake moves).
        self.register_buffer("reparam_actions", src_dst_to_env_action[self.not_lake_move])

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[-1] == logits.shape[-2] == N_OCCUPIABLE_CELL
        out = torch.zeros(
            *logits.shape[:-2],
            N_ACTION,
            device=logits.device,
            dtype=logits.dtype,
        )
        out[..., self.not_lake_move] = logits.flatten(start_dim=-2)[..., self.reparam_actions]
        return out


def extract_mem(
    cell_embeddings: torch.Tensor,
    unknown_piece_position_onehot: torch.Tensor,
    cell_mask: torch.Tensor,
) -> torch.Tensor:
    piece_pos = unknown_piece_position_onehot[..., cell_mask]
    hidden_embeddings = torch.gather(
        cell_embeddings,
        1,
        piece_pos.int().argmax(dim=-1, keepdim=True).expand(-1, -1, cell_embeddings.size(-1)),
    )
    is_active = piece_pos.any(dim=-1, keepdim=True)
    hidden_embeddings.masked_fill_(~is_active, 0)
    return hidden_embeddings
