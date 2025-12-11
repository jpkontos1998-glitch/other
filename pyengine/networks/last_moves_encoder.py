import torch
from torch import nn

from pyengine.utils.validation import expect_shape
from pyengine.utils.constants import TRAILING_LAST_MOVES_DIM, LAST_MOVES_DICTIONARY_SIZE


class LastMovesEncoder(nn.Module):
    def __init__(self, embed_dim: int, pos_emb_std: float):
        super().__init__()
        self.encodings = nn.Parameter(
            torch.empty(TRAILING_LAST_MOVES_DIM, LAST_MOVES_DICTIONARY_SIZE, embed_dim)
        )
        nn.init.trunc_normal_(self.encodings, std=pos_emb_std)
        self.range_tensor = torch.arange(TRAILING_LAST_MOVES_DIM).unsqueeze(1)

    def forward(self, last_moves: torch.Tensor) -> torch.Tensor:
        expect_shape(last_moves, ndim=2, dims={1: TRAILING_LAST_MOVES_DIM}, name="last_moves")
        return self.encodings[self.range_tensor, last_moves.T].sum(dim=0)
