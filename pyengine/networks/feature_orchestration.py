from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyengine.utils.validation import expect_shape
from pyengine.utils.constants import (
    BOARD_LEN,
    LAKE_INDICES,
    N_BOARD_CELL,
    N_OCCUPIABLE_CELL,
    N_PIECE_ID,
)
from pyengine.utils import get_pystratego

pystratego = get_pystratego()


@dataclass
class FeatureOrchestratorConfig:
    use_piece_ids: bool = True
    use_threaten: bool = True
    use_evade: bool = True
    use_actadj: bool = True
    use_battle: bool = True
    use_cemetery: bool = True
    use_protect: bool = True
    plane_history_len: int = 32


class FeatureOrchestrator(nn.Module):
    def __init__(self, cfg: FeatureOrchestratorConfig):
        super().__init__()
        self.cfg = cfg

        plane_mask = []
        for desc in pystratego.BOARDSTATE_CHANNEL_DESCRIPTION:
            keep = not (
                ("threat" in desc and not cfg.use_threaten)
                or ("evade" in desc and not cfg.use_evade)
                or ("actively_adj" in desc and not cfg.use_actadj)
                or ("dead" in desc and not cfg.use_cemetery)
                or ("deathstatus" in desc and not cfg.use_battle)
                or ("protect" in desc and not cfg.use_protect)
            )
            plane_mask.append(keep)
        self.n_infostate_plane = sum(plane_mask) + cfg.plane_history_len
        self.n_piece_id_plane = N_PIECE_ID if cfg.use_piece_ids else 0
        self.in_channels = self.n_infostate_plane + self.n_piece_id_plane

        self.register_buffer("plane_mask", torch.tensor(plane_mask, dtype=torch.bool))
        self.register_buffer("cell_mask", torch.ones(N_BOARD_CELL, dtype=torch.bool))
        self.cell_mask[torch.tensor(LAKE_INDICES)] = False
        assert (
            self.cell_mask.sum() == N_OCCUPIABLE_CELL
        ), "Cell mask is not consistent with the number of occupiable cells"
        self.register_buffer("piece_id_onehot", torch.eye(N_PIECE_ID))

    def forward(
        self,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare infostate tensor and piece IDs for embedding.

        Args:
            infostate_tensor: (B, NUM_BOARD_STATE_CHANNELS + MOVE_HISTORY, BOARD_LEN, BOARD_LEN)
            piece_ids: (B, BOARD_LEN, BOARD_LEN)

        Returns:
            (B, in_channels, N_OCCUPIABLE_CELL)
        """
        expect_shape(
            infostate_tensor, ndim=4, dims={2: BOARD_LEN, 3: BOARD_LEN}, name="infostate_tensor"
        )
        expect_shape(piece_ids, ndim=3, dims={1: BOARD_LEN, 2: BOARD_LEN}, name="piece_ids")
        if not (infostate_tensor.shape[:-3] == piece_ids.shape[:-2]):
            raise ValueError("Leading dimensions must match")
        if not (infostate_tensor.shape[-3] >= self.plane_mask.size(0)):
            raise ValueError(
                "Infostate tensor must have at least as many channels as the plane mask"
            )
        # Filter undesired planes.
        boardstate_tensor = infostate_tensor[..., : pystratego.NUM_BOARD_STATE_CHANNELS, :, :][
            ..., self.plane_mask, :, :
        ]  # (..., sum(plane_mask), BOARD_LEN, BOARD_LEN)
        if (
            self.cfg.plane_history_len == 0
        ):  # NOTE: Need special case 0 since [-0:] is equivalent to [0:]
            x = boardstate_tensor  # (..., n_infostate_plane, BOARD_LEN, BOARD_LEN)
        else:
            move_history_tensor = infostate_tensor[
                ..., pystratego.NUM_BOARD_STATE_CHANNELS :, :, :
            ][
                ..., -self.cfg.plane_history_len :, :, :
            ]  # (..., plane_history_len, BOARD_LEN, BOARD_LEN)
            x = torch.cat(
                [boardstate_tensor, move_history_tensor], dim=-3
            )  # (..., n_infostate_plane, BOARD_LEN, BOARD_LEN)
        if self.cfg.use_piece_ids:
            # One-hot encode piece IDs
            piece_id_onehot = F.embedding(
                piece_ids.to(torch.int64), self.piece_id_onehot
            )  # (..., BOARD_LEN, BOARD_LEN, N_PIECE_ID)
            # Move hot encoding to channel dimension
            piece_id_onehot = piece_id_onehot.permute(
                *range(piece_id_onehot.ndim - 3), -1, -3, -2
            )  # (B, N_PIECE_ID, BOARD_LEN, BOARD_LEN)
            # Stack infostate and piece ID planes
            x = torch.cat([x, piece_id_onehot], dim=-3)  # (..., in_channels, BOARD_LEN, BOARD_LEN)
        # Collapse cells to single dimension
        x = x.flatten(start_dim=-2)
        # Move cells to channel dimension
        x = x.permute(*range(x.ndim - 2), -1, -2)  # (..., N_BOARD_CELL, in_channels)
        # Remove lake cells
        x = x[..., self.cell_mask, :]  # (..., N_OCCUPIABLE_CELL, in_channels)
        assert x.shape[-2:] == (N_OCCUPIABLE_CELL, self.in_channels)
        return x
