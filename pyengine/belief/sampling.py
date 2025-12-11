from typing import Callable

import torch
from torch.distributions import Categorical

from pyengine.belief.masking import create_mask
from pyengine.utils.validation import expect_shape, expect_same_batch
from pyengine.utils.constants import (
    PLAYER_PIECE_TYPE_SLICE,
    N_PIECE_TYPE,
    N_CLASSIC_PIECE,
    N_BARRAGE_PIECE,
    N_PLAYER_PIECE_TYPE,
    N_BOARD_CELL,
)


def sampling_loop(
    n_sample: int,
    unknown_piece_position_onehot: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
    unknown_piece_counts: torch.Tensor,
    partial: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Autoregressive sampling loop.

    Args:
        n_sample: Number of samples to generate.
        unknown_piece_position_onehot (n_piece, N_BOARD_CELL): Positions of unknown opponent pieces.
        unknown_piece_has_moved (n_piece): Whether unknown opponent pieces have moved.
        unknown_piece_counts (N_PLAYER_PIECE_TYPE): Counts of unknown opponent pieces.
        partial: Partial function that takes a tensor of samples and returns a tensor of logits.

    Returns:
        (n_sample, n_piece, N_PIECE_TYPE): Samples from the belief distribution.
        Like the input unknown_piece_position_onehot and unknown_piece_has_moved tensors,
        the returned tensor specifies samples for the first `n_unknown_piece` pieces.
        The remaining `n_piece - n_unknown_piece` rows are left as all-False.
    """
    check_sampling_args(
        unknown_piece_position_onehot, unknown_piece_has_moved, unknown_piece_counts
    )

    dev = unknown_piece_has_moved.device
    n_piece = unknown_piece_has_moved.shape[0]
    n_unknown_piece = unknown_piece_position_onehot.any(dim=-1).sum()

    # Initialize samples and helper for indexing.
    samples = torch.zeros(n_sample, n_piece, N_PIECE_TYPE, dtype=torch.bool, device=dev)
    samples_idx = torch.arange(n_sample, device=dev)

    # Expand along sample dimension for mask creation.
    unknown_piece_has_moved = unknown_piece_has_moved.expand(n_sample, -1)
    unknown_piece_counts = unknown_piece_counts.expand(n_sample, -1)

    # Sample loop.
    for i in range(n_unknown_piece):
        logits = partial(samples)
        mask = create_mask(unknown_piece_counts, samples, unknown_piece_has_moved)
        assert mask.any(dim=-1).all()
        logits.masked_fill_(~mask, torch.finfo(logits.dtype).min)
        conditional = Categorical(logits=logits[:, i])
        pieces = conditional.sample()
        samples[samples_idx, i, pieces] = True
    return samples


def marginalize(samples: torch.Tensor) -> torch.Tensor:
    """Marginalize the empirical distribution over samples.

    NOTE: The returned tensor only contains rows for unknown pieces, unlike the input tensor, which is padded.

    Args:
        samples (n_sample, n_piece, N_PIECE_TYPE): Samples from the belief distribution.

    Returns:
        (n_unknown_piece, N_PIECE_TYPE): Marginal distribution over unknown pieces.
    """
    expect_shape(
        samples,
        ndim=3,
        dims={1: (N_CLASSIC_PIECE, N_BARRAGE_PIECE), 2: N_PIECE_TYPE},
        name="samples",
    )
    marginal = samples.float().mean(dim=0)
    marginal = marginal[marginal.any(dim=-1)]
    marginal = marginal[:, PLAYER_PIECE_TYPE_SLICE]
    return marginal


def check_sampling_args(
    unknown_piece_position_onehot: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
    unknown_piece_counts: torch.Tensor,
) -> None:
    expect_shape(
        unknown_piece_position_onehot,
        ndim=2,
        dims={0: (N_CLASSIC_PIECE, N_BARRAGE_PIECE), 1: N_BOARD_CELL},
        name="unknown_piece_position_onehot",
    )
    expect_shape(unknown_piece_has_moved, ndim=1, name="unknown_piece_has_moved")
    expect_shape(
        unknown_piece_counts, ndim=1, dims={0: N_PLAYER_PIECE_TYPE}, name="unknown_piece_counts"
    )
    expect_same_batch(unknown_piece_position_onehot, unknown_piece_has_moved)
