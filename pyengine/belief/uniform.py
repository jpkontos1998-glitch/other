import torch

from pyengine.utils.validation import expect_shape, expect_same_batch, expect_type
from pyengine.belief.sampling import sampling_loop, check_sampling_args
from pyengine.utils.types import GenerateArgType
from pyengine.utils.constants import (
    INFOSTATE_OPP_PROB_SLICE,
    MOVABLE_PLAYER_PIECE_TYPE_SLICE,
    IMMOVABLE_PLAYER_PIECE_TYPE_SLICE,
    PLAYER_PIECE_TYPE_SLICE,
    N_PLAYER_PIECE_TYPE,
    N_CLASSIC_PIECE,
    N_BARRAGE_PIECE,
    N_BOARD_CELL,
    BOARD_LEN,
    N_PIECE_TYPE,
)


class _MarginalizedUniformBelief:
    generate_arg_type = GenerateArgType.MARGLIZED_UNIFORM

    def __call__(self, infostate_tensor: torch.Tensor, unknown_piece_position_onehot: torch.Tensor):
        """Compute posterior marginals for opponent piece types under uniform policy.

        The i-th row of the belief corresponds to the i-th opponent unknown piece for the row-major
        ordering of the unknown pieces from the perspective of the querying player.
        NOTE: If there are fewer opponent unknown pieces than total pieces, there is no guarantee
        on the return value of the excess rows.

        Args:
            infostate_tensor (B, N_BOARD_STATE_CHANNEL, BOARD_LEN, BOARD_LEN)
            unknown_piece_position_onehot (B, n_piece, N_BOARD_CELL): Positions of unknown opponent pieces.
        Returns:
            (batch_size, n_piece, N_PIECE_TYPE): Log-probability of the belief.
        """
        expect_shape(
            infostate_tensor, ndim=4, dims={2: BOARD_LEN, 3: BOARD_LEN}, name="infostate_tensor"
        )
        expect_shape(
            unknown_piece_position_onehot,
            ndim=3,
            dims={1: (N_CLASSIC_PIECE, N_BARRAGE_PIECE), 2: N_BOARD_CELL},
            name="unknown_piece_position_onehot",
        )
        expect_same_batch(infostate_tensor, unknown_piece_position_onehot)

        B, n_piece, _ = unknown_piece_position_onehot.shape
        belief = torch.zeros(B, n_piece, N_PIECE_TYPE, device=infostate_tensor.device)
        positions = unknown_piece_position_onehot.long().argmax(dim=-1)
        # The marginalized uniform posterior is computed by the infostate kernel,
        # so we can just slice the appropriate channels and gather the appropriate cells.
        b = infostate_tensor[:, INFOSTATE_OPP_PROB_SLICE].flatten(start_dim=2).permute(0, 2, 1)
        b = b.gather(1, positions.unsqueeze(-1).expand(-1, -1, b.size(-1)))
        belief[:, :, PLAYER_PIECE_TYPE_SLICE] = b
        return torch.log(belief + torch.finfo(belief.dtype).tiny)

    def generate(
        self,
        n_sample: int,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        infostate_tensor: torch.Tensor,
        **kwargs,
    ):
        """Generate samples from marginalized posterior under uniform policy.

        The samples are generated autoregressively from the associated marginals.
        The sampling loop will take care of masking to ensure that the samples are valid.
        NOTE: This is not the same as the posterior over unknown pieces under uniform policy.

        Args:
            n_sample: Number of samples to generate.
            unknown_piece_position_onehot (n_piece, N_BOARD_CELL): Positions of unknown opponent pieces.
            unknown_piece_has_moved (n_piece): Whether unknown opponent pieces have moved.
            unknown_piece_counts (N_PLAYER_PIECE_TYPE): Counts of unknown opponent pieces.
            infostate_tensor (N_BOARD_STATE_CHANNEL, BOARD_LEN, BOARD_LEN)
        Returns:
            (n_sample, n_piece, N_PIECE_TYPE): Samples from the marginalized uniform posterior.
        """
        expect_type(n_sample, int, name="n_sample")
        check_sampling_args(
            unknown_piece_position_onehot, unknown_piece_has_moved, unknown_piece_counts
        )
        if infostate_tensor.ndim != 3:
            raise ValueError(f"infostate_tensor must be 3D, got {infostate_tensor.ndim}D")
        if not (infostate_tensor.shape[1] == infostate_tensor.shape[2] == BOARD_LEN):
            raise ValueError(
                f"infostate_tensor must have shape (N_BOARD_STATE_CHANNEL, BOARD_LEN, BOARD_LEN), got {infostate_tensor.shape}"
            )
        beliefs = self(
            infostate_tensor.unsqueeze(0), unknown_piece_position_onehot.unsqueeze(0)
        ).repeat(n_sample, 1, 1)

        def partial(samples: torch.Tensor) -> torch.Tensor:
            return beliefs

        return sampling_loop(
            n_sample,
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
            partial,
        )


class _UniformBelief:
    generate_arg_type = GenerateArgType.UNIFORM

    def __call__(
        self,
        unknown_piece_type_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute posterior for opponent piece types under uniform policy.

        The i-th row of the belief corresponds to the i-th opponent unknown piece for the row-major
        ordering of the unknown pieces from the perspective of the querying player.
        NOTE: If there are fewer opponent unknown pieces than total pieces, there is no guarantee
        on the return value of the excess rows.

        Args:
            unknown_piece_type_onehot (B, n_piece, N_PIECE_TYPE): Types of unknown opponent pieces.
            unknown_piece_has_moved (B, n_piece): Whether unknown opponent pieces have moved.
            unknown_piece_counts (B, N_PLAYER_PIECE_TYPE): Counts of unknown opponent pieces.
        Returns:
            (B, n_piece, N_PIECE_TYPE): Log-probability of the belief.
        """
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
        expect_shape(
            unknown_piece_counts, ndim=2, dims={1: N_PLAYER_PIECE_TYPE}, name="unknown_piece_counts"
        )
        expect_same_batch(unknown_piece_type_onehot, unknown_piece_has_moved, unknown_piece_counts)

        batch_size, n_piece = unknown_piece_has_moved.shape
        remaining_unk, remaining_moved_unk = compute_remaining_counts(
            unknown_piece_type_onehot, unknown_piece_counts, unknown_piece_has_moved
        )
        belief = make_belief(
            remaining_unk.view(batch_size * n_piece, N_PLAYER_PIECE_TYPE),
            unknown_piece_has_moved.view(batch_size * n_piece),
            remaining_moved_unk.view(batch_size * n_piece),
        ).view(batch_size, n_piece, N_PIECE_TYPE)
        return torch.log(belief + torch.finfo(belief.dtype).tiny)

    def generate(
        self,
        n_sample: int,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
    ):
        """Generate samples from posterior over unknown pieces under uniform policy.

        Args:
            n_sample: Number of samples to generate.
            unknown_piece_position_onehot (n_piece, N_BOARD_CELL): Positions of unknown opponent pieces.
            unknown_piece_has_moved (n_piece): Whether unknown opponent pieces have moved.
            unknown_piece_counts (N_PLAYER_PIECE_TYPE): Counts of unknown opponent pieces.
        Returns:
            (n_sample, n_piece, N_PIECE_TYPE)
        """
        expect_type(n_sample, int, name="n_sample")
        check_sampling_args(
            unknown_piece_position_onehot, unknown_piece_has_moved, unknown_piece_counts
        )

        def partial(samples: torch.Tensor) -> torch.Tensor:
            return self(
                samples,
                unknown_piece_has_moved.unsqueeze(0).repeat(n_sample, 1),
                unknown_piece_counts.unsqueeze(0).repeat(n_sample, 1),
            )

        return sampling_loop(
            n_sample,
            unknown_piece_position_onehot,
            unknown_piece_has_moved,
            unknown_piece_counts,
            partial,
        )


def make_belief(
    unknown_piece_counts: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
    unknown_piece_has_moved_counts: torch.Tensor,
) -> torch.Tensor:
    """Make posterior over unknown pieces under uniform policy.

    We compute the belief in three parts:
    1. Pieces that have moved.
    2. Pieces that have not moved and are not movable.
    3. Pieces that have not moved and are movable.

    Args:
        unknown_piece_counts (B, N_PLAYER_PIECE_TYPE): Counts of unknown pieces.
        unknown_piece_has_moved (B,): Whether each piece has moved.
        unknown_piece_has_moved_counts (B,): Number of pieces that have moved.

    Returns:
        (B, N_PIECE_TYPE): Belief over unknown pieces.
    """
    expect_shape(
        unknown_piece_counts, ndim=2, dims={1: N_PLAYER_PIECE_TYPE}, name="unknown_piece_counts"
    )
    expect_shape(unknown_piece_has_moved, ndim=1, name="unknown_piece_has_moved")
    expect_shape(unknown_piece_has_moved_counts, ndim=1, name="unknown_piece_has_moved_counts")
    expect_same_batch(unknown_piece_counts, unknown_piece_has_moved, unknown_piece_has_moved_counts)

    # Initialize belief tensor.
    b = torch.zeros_like(unknown_piece_counts, dtype=torch.float32)

    # Tensors that are useful for multiple parts of belief computation.
    counts_mvbl = unknown_piece_counts[:, MOVABLE_PLAYER_PIECE_TYPE_SLICE]
    counts_imvbl = unknown_piece_counts[:, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE]
    n_mvble_piece = counts_mvbl.sum(dim=-1)
    has_not_moved_counts = unknown_piece_counts.sum(dim=-1) - unknown_piece_has_moved_counts
    unmoved_pieces_left = has_not_moved_counts > 0
    has_not_moved = ~unknown_piece_has_moved

    # 1. For pieces that have moved, the belief is:
    # P(piece type) = I[Piece type is movable] * |Count of that piece type| / |Count of movable pieces|.
    moved_belief = counts_mvbl / n_mvble_piece.clamp(min=1).unsqueeze(-1)
    b[unknown_piece_has_moved, MOVABLE_PLAYER_PIECE_TYPE_SLICE] = moved_belief[
        unknown_piece_has_moved
    ]

    # 2. For pieces that are immovable and unmoved, the belief is:
    # P(piece type) = |Count of that piece type| / |Count of unmoved pieces|.
    immovable_and_not_moved_belief = counts_imvbl / has_not_moved_counts.unsqueeze(-1).clamp(min=1)
    imm_unm = has_not_moved & unmoved_pieces_left
    b[imm_unm, IMMOVABLE_PLAYER_PIECE_TYPE_SLICE] = immovable_and_not_moved_belief[imm_unm]

    # 3. For pieces that are movable and unmoved, the belief is:
    # P(piece type) = (|Count of that piece type| / |Count of movable pieces|) (1 - |Count of unmovable pieces| / |Count of unmoved pieces|).
    reweighting = 1 - counts_imvbl.sum(dim=-1) / has_not_moved_counts.clamp(min=1)
    movable_and_not_moved_belief = moved_belief * reweighting.unsqueeze(-1)
    mvb_unm = has_not_moved & unmoved_pieces_left & (n_mvble_piece > 0)
    b[mvb_unm, MOVABLE_PLAYER_PIECE_TYPE_SLICE] = movable_and_not_moved_belief[mvb_unm]

    # Inject the `b` into a belief tensor containing zeros for non-player pieces.
    belief = torch.zeros(b.shape[0], N_PIECE_TYPE, device=b.device, dtype=torch.float32)
    belief[:, PLAYER_PIECE_TYPE_SLICE] = b

    return belief


def compute_remaining_counts(
    unknown_piece_type_onehot: torch.Tensor,
    unknown_piece_counts: torch.Tensor,
    unknown_piece_has_moved: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute remaining counts of unknown piece types and of unknown moved pieces.

    Args:
        unknown_piece_type_onehot (B, n_piece, N_PIECE_TYPE)
        unknown_piece_counts (B, N_PLAYER_PIECE_TYPE)
        unknown_piece_has_moved (B, n_piece)
    Returns:
        (B, n_piece, N_PLAYER_PIECE_TYPE): Remaining counts for unknown pieces.
        (B, n_piece): Remaining number of moved unknown pieces.
    """
    expect_shape(
        unknown_piece_type_onehot,
        ndim=3,
        dims={1: (N_CLASSIC_PIECE, N_BARRAGE_PIECE), 2: N_PIECE_TYPE},
        name="unknown_piece_type_onehot",
    )
    expect_shape(
        unknown_piece_counts, ndim=2, dims={1: N_PLAYER_PIECE_TYPE}, name="unknown_piece_counts"
    )
    expect_shape(
        unknown_piece_has_moved,
        ndim=2,
        dims={1: (N_CLASSIC_PIECE, N_BARRAGE_PIECE)},
        name="unknown_piece_has_moved",
    )
    expect_same_batch(unknown_piece_type_onehot, unknown_piece_counts, unknown_piece_has_moved)

    unknown_piece_type_onehot = unknown_piece_type_onehot.long()
    unknown_piece_has_moved = unknown_piece_has_moved.long()
    revealed_unk = unknown_piece_type_onehot.cumsum(dim=1) - unknown_piece_type_onehot
    remaining_unk = unknown_piece_counts.unsqueeze(1) - revealed_unk[:, :, PLAYER_PIECE_TYPE_SLICE]
    revealed_moved_unk = unknown_piece_has_moved.cumsum(dim=-1) - unknown_piece_has_moved
    remaining_moved_unk = unknown_piece_has_moved.sum(dim=1, keepdim=True) - revealed_moved_unk
    return remaining_unk, remaining_moved_unk


# We initialize the classes once here so we don't need to re-initialize them for each call.
marginalized_uniform_belief = _MarginalizedUniformBelief()
uniform_belief = _UniformBelief()
