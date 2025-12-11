import torch

from pyengine.utils.validation import expect_shape, expect_type
from pyengine.utils import get_pystratego
from pyengine.utils.constants import (
    N_ARRANGEMENT_ROW,
    N_ARRANGEMENT_COL,
    N_PIECE_TYPE,
    ARRANGEMENT_SIZE,
)

pystratego = get_pystratego()


def flip_arrangements(arrangements: torch.Tensor) -> torch.Tensor:
    """Flip arrangements across middle column.

    Args:
        arrangements (B, ARRANGEMENT_SIZE, N_PIECE_TYPE): Arrangements to flip.

    Returns:
        Flipped arrangements.
    """
    check_onehot_arrangements(arrangements)
    return (
        arrangements.reshape(-1, N_ARRANGEMENT_ROW, N_ARRANGEMENT_COL, N_PIECE_TYPE)
        .flip(-2)
        .reshape(-1, ARRANGEMENT_SIZE, N_PIECE_TYPE)
    )


def to_string(arrangements: torch.Tensor) -> list[str]:
    """Convert arrangements in tensor representation to string representation.

    Args:
        arrangements (B, ARRANGEMENT_SIZE, N_PIECE_TYPE): Arrangements to convert.

    Returns:
        string_arrs: String representations of arrangements.
    """
    check_onehot_arrangements(arrangements)
    return pystratego.util.arrangement_strings_from_tensor(
        arrangements.argmax(dim=-1).type(torch.uint8)
    )


def filter_terminal(arrangements: list[str]) -> list[str]:
    """Filter terminal arrangements.

    Args:
        arrangements: Arrangements to filter.

    Returns:
        Nonterminal arrangements.
    """
    expect_type(arrangements, list, name="arrangements")
    for a in arrangements:
        expect_type(a, str, name="arrangements[i]")
    is_terminal = pystratego.util.is_terminal_arrangement(arrangements)
    return [arrangement for arrangement, is_t in zip(arrangements, is_terminal) if not is_t]


def check_onehot_arrangements(arrangements: torch.Tensor) -> None:
    """Validate `arrangements` as one-hot encoded arrangements."""
    expect_shape(
        arrangements, ndim=3, dims={1: ARRANGEMENT_SIZE, 2: N_PIECE_TYPE}, name="arrangements"
    )
    if not arrangements.sum(dim=-1).all():
        raise ValueError("arrangements must be one-hot encoded")
