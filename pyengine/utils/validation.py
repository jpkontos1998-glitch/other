from typing import Any, Iterable

import torch


def expect_dtype(x: torch.Tensor, dtype: torch.dtype, name: str = "tensor") -> None:
    """Validate `x` has the expected dtype.

    Args:
        x: The tensor to validate.
        dtype: The expected dtype.
    """
    if x.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {x.dtype}")


def expect_device(x: torch.Tensor, device: torch.device, name: str = "tensor") -> None:
    """Validate `x` has the expected device.

    Args:
        x: The tensor to validate.
        device: The expected device.
    """
    if x.device != device:
        raise ValueError(f"{name} must be on device {device}, got {x.device}")


def expect_shape(
    x: torch.Tensor,
    ndim: int | None = None,
    dims: dict[int, int | Iterable[int]] | None = None,
    name: str = "tensor",
) -> None:
    """Validate `x` has the expected dimensionality and per-axis sizes.

    Args:
        x: The tensor to validate.
        ndim: If provided, require x.ndim == ndim.
        dims: Mapping axis_index -> allowed size(s). Each value can be:
              - a single int (exact match)
              - an iterable of ints (any of these sizes is allowed)
        name: Used in error messages.

    Raises:
        ValueError if any check fails.
    """
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {x.ndim}D (shape={tuple(x.shape)})")

    if dims:
        for axis, allowed in dims.items():
            if axis < 0:
                axis += x.ndim
            if axis < 0 or axis >= x.ndim:
                raise ValueError(f"{name}: axis {axis} is out of bounds for ndim={x.ndim}")

            # Normalize allowed to a tuple of ints
            if isinstance(allowed, Iterable) and not isinstance(allowed, (int, torch.SymInt)):
                allowed_sizes = tuple(int(a) for a in allowed)
            else:
                allowed_sizes = (int(allowed),)

            if x.shape[axis] not in allowed_sizes:
                raise ValueError(
                    f"{name} dim {axis} must be one of {allowed_sizes}, got {x.shape[axis]} "
                    f"(shape={tuple(x.shape)})"
                )


def expect_same_batch(*tensors: torch.Tensor) -> None:
    """Validate tensors have same batch size.

    Args:
        tensors: The tensors to validate.

    Raises:
        ValueError if any check fails.
    """
    if not all(t.shape[0] == tensors[0].shape[0] for t in tensors):
        raise ValueError(
            f"Tensors must have same batch size, got {tuple(t.shape[0] for t in tensors)}"
        )


def expect_same_device(*tensors: torch.Tensor) -> None:
    """Validate tensors have same device.

    Args:
        tensors: The tensors to validate.
    """
    if not all(t.device == tensors[0].device for t in tensors):
        raise ValueError(f"Tensors must have same device, got {tuple(t.device for t in tensors)}")


def expect_type(x: Any, expected_type: type, name: str = "value") -> None:
    """Validate `x` is of type `expected_type`.

    Args:
        x: The value to validate.
        expected_type: The expected type.
    """
    if not isinstance(x, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(x)}")
