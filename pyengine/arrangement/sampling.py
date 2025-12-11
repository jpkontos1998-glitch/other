import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from pyengine.utils.validation import expect_type
from pyengine.arrangement.utils import flip_arrangements, check_onehot_arrangements
from pyengine.networks.arrangement_transformer import ArrangementTransformer
from pyengine.utils import eval_mode
from pyengine.utils.constants import (
    ARRANGEMENT_ROW_INDICES,
    ARRANGEMENT_SIZE,
    CORRIDOR_COL_INDICES,
    N_ARRANGEMENT_ROW,
    N_ARRANGEMENT_COL,
    N_PIECE_TYPE,
    N_VF_CAT,
    PIECE_INDICES,
)


def generate_arrangements(
    n_sample: int,
    model: ArrangementTransformer,
    dtype: torch.dtype = torch.bfloat16,
    randomize: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    """Sample arrangements from model.

    Args:
        n_sample: Number of arrangements to generate.
        model: Model from which to generate arrangements.
        dtype: Data type with which to perform inference.
        randomize: Whether to randomize the left-right orientation of the generated arrangements.

    Returns:
        (n_sample, ARRANGEMENT_SIZE, N_PIECE_TYPE): Generated arrangements.
        (n_sample, ARRANGEMENT_SIZE, N_VF_CAT) OR (n_sample, ARRANGEMENT_SIZE): Predicted values for generated arrangement prefixes.
        (n_sample, ARRANGEMENT_SIZE): Predicted normalized future accumulated entropy for generated arrangement prefixes.
        (n_sample, ARRANGEMENT_SIZE, N_PIECE_TYPE): Log probabilities of next piece placements for generated arrangement prefixes.
        (n_sample): Whether the orientation of generated arrangements has been flipped.
        dict: Logging statistics.
    """
    expect_type(n_sample, int, name="n_sample")
    expect_type(dtype, torch.dtype, name="dtype")
    expect_type(randomize, bool, name="randomize")
    if not (n_sample > 0):
        raise ValueError(f"n_sample must be positive, got {n_sample}")

    device = model.device

    # Allocate tensors
    base_shape = (n_sample, ARRANGEMENT_SIZE)
    samples = torch.zeros(*base_shape, N_PIECE_TYPE, device=device)
    if model.cfg.use_cat_vf:
        values = torch.zeros(*base_shape, N_VF_CAT, device=device, dtype=torch.float32)
    else:
        values = torch.zeros(*base_shape, device=device, dtype=torch.float32)
    ents = torch.zeros(*base_shape, device=device, dtype=torch.float32)
    log_probs = torch.zeros(*base_shape, N_PIECE_TYPE, device=device, dtype=torch.float32)
    batch_idx = torch.arange(n_sample, device=device)

    # Generate arrangements
    for t in range(ARRANGEMENT_SIZE):
        # Computation
        with autocast(device.type, dtype=dtype), torch.no_grad(), eval_mode(model):
            tensor_dict = model(samples)
            logit_pi, v, rv = tensor_dict["logits"], tensor_dict["value"], tensor_dict["ent_pred"]
            log_pi_t = nn.functional.log_softmax(logit_pi[:, t], dim=-1)
            piece_t = Categorical(logits=log_pi_t).sample()
        # Assignment
        samples[batch_idx, t, piece_t] = 1
        log_probs[batch_idx, t] = log_pi_t
        if model.cfg.use_cat_vf:
            values[batch_idx, t] = v[:, t].to(torch.float32)
        else:
            values[batch_idx, t] = v[:, t].squeeze(-1).to(torch.float32)
        ents[batch_idx, t] = rv[:, t].squeeze(-1).to(torch.float32)

    # Compute logging statistics
    piece_info: dict[str, float] = {
        **{
            f"arr_gen/{name}_leftside": left_side_piece_count(samples, idx).mean().item()
            for name, idx in PIECE_INDICES.items()
        },
        **{
            f"arr_gen/{name}_row{row}": row_piece_count(samples, idx, row).mean().item()
            for name, idx in PIECE_INDICES.items()
            for row in ARRANGEMENT_ROW_INDICES
        },
        "arr_gen/open_flag": is_open_flag(samples, dtype).float().mean().item(),
    }

    # Randomly flip arrangements if specified
    flipped_mask = torch.zeros(n_sample, device=device, dtype=torch.bool)
    if randomize:
        flipped_mask = torch.rand(n_sample, device=device) > 0.5
        samples[flipped_mask] = flip_arrangements(samples[flipped_mask])

    return samples, values, ents, log_probs, flipped_mask, piece_info


def left_side_piece_count(arrangements: torch.Tensor, piece_idx: int) -> torch.Tensor:
    """Count number of pieces of type `piece_idx` on left side of each arrangement."""
    check_onehot_arrangements(arrangements)
    if piece_idx not in PIECE_INDICES.values():
        raise ValueError(f"piece_idx must be in {PIECE_INDICES.values()}")
    by_side = arrangements.reshape(-1, N_ARRANGEMENT_ROW, 2, N_ARRANGEMENT_COL // 2, N_PIECE_TYPE)
    left_side = by_side[:, :, 0]
    return left_side[:, :, :, piece_idx].flatten(start_dim=1).sum(dim=-1)


def row_piece_count(arrangements: torch.Tensor, piece_idx: int, row_idx: int) -> torch.Tensor:
    """Count number of pieces of type `piece_idx` in the `row_idx` row of each arrangement."""
    check_onehot_arrangements(arrangements)
    if piece_idx not in PIECE_INDICES.values():
        raise ValueError(f"piece_idx must be in {PIECE_INDICES.values()}")
    if row_idx not in ARRANGEMENT_ROW_INDICES:
        raise ValueError(f"row_idx must be in {ARRANGEMENT_ROW_INDICES}")
    by_row = arrangements.reshape(-1, N_ARRANGEMENT_ROW, N_ARRANGEMENT_COL, N_PIECE_TYPE)
    row = by_row[:, row_idx]
    return row[:, :, piece_idx].flatten(start_dim=1).sum(dim=-1)


def is_open_flag(arrangements: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Check whether flag is open for each arrangement.

    A flag is open iff opponent can reach it without going through bomb.
    """
    check_onehot_arrangements(arrangements)
    expect_type(dtype, torch.dtype, name="dtype")
    device = arrangements.device
    B = arrangements.size(0)

    # Extract masks for flag and non-bomb cells.
    arrangements_BRC = arrangements.argmax(dim=-1).view(B, N_ARRANGEMENT_ROW, N_ARRANGEMENT_COL)
    flag = arrangements_BRC == PIECE_INDICES["flag"]
    nonbomb = ~(arrangements_BRC == PIECE_INDICES["bomb"])

    # Initialize reachable set as front row corridors cells that are not bombs.
    reachable = torch.zeros_like(flag)
    reachable[:, -1, CORRIDOR_COL_INDICES] = nonbomb[:, -1, CORRIDOR_COL_INDICES]

    # Adjacency kernel
    k = torch.tensor(
        [
            [0, 1, 0],  # above
            [1, 0, 1],  # left, right
            [0, 1, 0],  # below
        ],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)

    # Flood-fill until reachable set stops growing.
    while True:
        nbr = F.conv2d(reachable.unsqueeze(1).to(dtype), k, padding=1).squeeze(1).bool()
        new_reachable = reachable | (nbr & nonbomb)
        if torch.equal(new_reachable, reachable):
            break
        reachable = new_reachable

    # Flag is open iff it is in reachable set.
    return (reachable & flag).flatten(start_dim=1).any(dim=-1)
