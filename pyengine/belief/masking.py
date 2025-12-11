import torch

from pyengine.utils.validation import expect_shape, expect_same_batch, expect_same_device

from pyengine.utils.constants import (
    MOVABLE_PLAYER_PIECE_TYPE_SLICE,
    IMMOVABLE_PLAYER_PIECE_TYPE_SLICE,
    PLAYER_PIECE_TYPE_SLICE,
    N_MOVABLE_PLAYER_PIECE_TYPE,
    N_IMMOVABLE_PLAYER_PIECE_TYPE,
    N_PLAYER_PIECE_TYPE,
    N_PIECE_TYPE,
    N_BARRAGE_PIECE,
    N_CLASSIC_PIECE,
)


def create_mask(
    unknown_piece_counts: torch.Tensor,
    unknown_piece_type_onehot: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
) -> torch.Tensor:
    """Create a mask for feasible types of the unknown pieces.

    There are two constraints that determine feasibility:
    1. The number of pieces of each type must match the unknown piece counts.
    2. The moved pieces must be a movable type.

    Args:
        unknown_piece_counts (B, N_PLAYER_PIECE_TYPE): Number of unknown pieces of each type.
        unknown_piece_type_onehot (B, N_PLAYER_PIECE, N_PIECE_TYPE): Boolean one-hot encoding of unknown pieces.
        unknown_piece_has_moved (B, N_PLAYER_PIECE): Whether each unknown piece has moved.

    Returns:
        (B, N_PLAYER_PIECE, N_PIECE_TYPE): Boolean mask for feasible types of the unknown pieces.
        As with the arguments, the piece ordering associated to the returned mask is row-major from the perspective of the querying player.
    """
    check_arguments(unknown_piece_counts, unknown_piece_type_onehot, unknown_piece_has_moved)
    piece_count_mask = create_piece_count_mask(unknown_piece_counts, unknown_piece_type_onehot)
    movement_mask = create_movement_mask(
        unknown_piece_counts, unknown_piece_type_onehot, unknown_piece_has_moved
    )
    return piece_count_mask & movement_mask


def create_piece_count_mask(
    unknown_piece_counts: torch.Tensor, unknown_piece_type_onehot: torch.Tensor
) -> torch.Tensor:
    """Create mask for feasible types of unknown pieces based on the number of pieces of each type."""
    dev = unknown_piece_counts.device
    B, n_piece, _ = unknown_piece_type_onehot.shape

    # Pad piece counts to include zeros for non-player pieces
    piece_counts = torch.zeros(B, N_PIECE_TYPE, dtype=unknown_piece_counts.dtype, device=dev)
    piece_counts[:, PLAYER_PIECE_TYPE_SLICE] = unknown_piece_counts

    # Mask out piece types whose assignment counts match total counts
    unknown_piece_type_onehot_int = unknown_piece_type_onehot.int()
    cumulative_counts = unknown_piece_type_onehot_int.cumsum(dim=1) - unknown_piece_type_onehot_int
    mask = cumulative_counts < piece_counts.unsqueeze(1)

    # Clear mask for padding rows
    total_piece_counts = piece_counts.sum(dim=1, keepdim=True)
    range_tensor = torch.arange(n_piece, device=dev).unsqueeze(0).expand(B, -1)
    is_padding = range_tensor >= total_piece_counts
    mask |= is_padding.unsqueeze(-1)

    return mask


def create_movement_mask(
    unknown_piece_counts: torch.Tensor,
    unknown_piece_type_onehot: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
) -> torch.Tensor:
    """Create mask for feasible types of unknown pieces based on the movement of the pieces.

    The mask handles both preventing the assignment of immovable piece types to moved pieces
    and preventing the assignment of movable piece types to unmoved pieces where doing so would
    necessitate a downstream assignment of an immovable piece type to a moved piece.
    """
    dev = unknown_piece_has_moved.device
    B, n_piece = unknown_piece_has_moved.shape
    mask = torch.ones(B, n_piece, N_PIECE_TYPE, dtype=torch.bool, device=dev)

    # Only allow immovable piece types for unmoved pieces
    not_moved = ~unknown_piece_has_moved.unsqueeze(-1).expand(-1, -1, N_IMMOVABLE_PLAYER_PIECE_TYPE)
    mask[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE] &= not_moved

    # Compute the number of remaining immovable pieces
    total_immovable = unknown_piece_counts[:, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE].sum(dim=1)
    immovable = unknown_piece_type_onehot[:, :, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE]
    cumulative_immovable = immovable.cumsum(dim=1).sum(dim=-1)
    remaining_immovable = total_immovable.unsqueeze(1) - cumulative_immovable

    # Compute the number of remaining unmoved pieces
    range_tensor = torch.arange(n_piece, device=dev).unsqueeze(0).expand(B, -1)
    not_padding = range_tensor < unknown_piece_counts.sum(dim=1).unsqueeze(1)
    unmoved = (~unknown_piece_has_moved) & not_padding
    remaining_unmoved = unmoved.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

    # Only allow movable piece types for unmoved pieces when it would not cause a downstream movability constraint violation
    constrained = remaining_immovable == remaining_unmoved
    allowable = ~(constrained & unmoved).unsqueeze(-1).expand(-1, -1, N_MOVABLE_PLAYER_PIECE_TYPE)
    mask[:, :, MOVABLE_PLAYER_PIECE_TYPE_SLICE] &= allowable

    return mask


def check_arguments(
    unknown_piece_counts: torch.Tensor,
    unknown_piece_type_onehot: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
) -> None:
    expect_shape(
        unknown_piece_counts, ndim=2, dims={1: N_PLAYER_PIECE_TYPE}, name="unknown_piece_counts"
    )
    expect_shape(
        unknown_piece_type_onehot,
        ndim=3,
        dims={1: (N_CLASSIC_PIECE, N_BARRAGE_PIECE), 2: N_PIECE_TYPE},
        name="unknown_piece_type_onehot",
    )
    expect_shape(
        unknown_piece_has_moved,
        ndim=2,
        dims={1: (N_CLASSIC_PIECE, N_BARRAGE_PIECE)},
        name="unknown_piece_has_moved",
    )
    expect_same_batch(unknown_piece_counts, unknown_piece_type_onehot, unknown_piece_has_moved)
    expect_same_device(unknown_piece_counts, unknown_piece_type_onehot, unknown_piece_has_moved)
