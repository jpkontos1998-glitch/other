from dataclasses import dataclass
from typing import Callable, Generator

import torch
import torch.nn.functional as F

from pyengine.utils.validation import expect_dtype, expect_shape, expect_type
from pyengine.arrangement.utils import flip_arrangements, check_onehot_arrangements
from pyengine.utils import get_pystratego
from pyengine.utils.constants import (
    CATEGORICAL_AGGREGATION,
    ARRANGEMENT_SIZE,
    UPPER_QUANTILES,
    N_PIECE_TYPE,
    N_VF_CAT,
)

pystratego = get_pystratego()


@dataclass
class Batch:
    arrangements: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    reg_returns: torch.Tensor
    advantages: torch.Tensor


class ArrangementBuffer:
    """Buffer for arrangements.

    Provides facilities for:
    1. Storing arrangements and associated data.
    2. Computing quantities for training.
    3. Sampling minibatches for training.

    Intended usage:
    1. Whenever the reset behavior of the training environment is updated to a new set of arrangements,
    call `add_arrangements` to add the new arrangements and associated data to the buffer.
    2. After each step of the training environment, call `add_rewards` to add the rewards for the
    arrangements associated to the games that terminated on that step.
    3. Before training, call `process_data` to compute the quantities necessary for training.
    4. To train, call `sample` in a loop to sample minibatches for training.
    5. After training, call `filter` to remove expired data from the buffer.

    NOTE: The buffer is not fixed size. The user must periodically call `filter` to prevent the buffer from growing indefinitely.
    NOTE: `add_arrangements` must be called after each `filter` call before other methods can be used.
    """

    def __init__(
        self,
        storage_duration: int,
        barrage: bool,
        device: torch.device,
        use_cat_vf: bool,
    ):
        """
        Args:
            storage_duration: Number of steps for which buffer is guaranteed to retain arrangements.
                NOTE: Should be at least as large as the sum of:
                    1. The maximum number of steps in a game.
                    2. The number of steps in between environment arrangement updates.
                Otherwise, `add_rewards` may attempt to lookup arrangements that have already been filtered.
            barrage: Whether arrangements are for barrage or classic.
            device: Device on which to store tensors.
            use_cat_vf: Whether the value function is categorical.
        """
        if not isinstance(storage_duration, int):
            raise ValueError(f"storage_duration must be an integer, got {type(storage_duration)}")
        if not isinstance(barrage, bool):
            raise ValueError(f"barrage must be a boolean, got {type(barrage)}")
        if not isinstance(device, torch.device):
            raise ValueError(f"device must be a torch.device, got {type(device)}")
        if not isinstance(use_cat_vf, bool):
            raise ValueError(f"use_cat_vf must be a bool, got {type(use_cat_vf)}")

        self.storage_duration = storage_duration
        self.use_cat_vf = use_cat_vf
        self.categorical_aggregation = CATEGORICAL_AGGREGATION.to(device)
        self.device = device

        # Whether the buffer needs new arrangements
        self.need_arrangements = True

        # Shapes
        self.value_shape: tuple[int, ...]
        self.reward_shape: tuple[int, ...]
        self.count_shape: tuple[int, ...]
        if use_cat_vf:
            self.value_shape = (ARRANGEMENT_SIZE, N_VF_CAT)
            self.reward_shape = (N_VF_CAT,)
            self.count_shape = (1,)
        else:
            self.value_shape = (ARRANGEMENT_SIZE,)
            self.reward_shape = ()
            self.count_shape = ()

        # Tensors added with new arrangements
        self.arrangements = torch.zeros((0, ARRANGEMENT_SIZE, N_PIECE_TYPE), device=device)
        self.values = torch.zeros((0, *self.value_shape), device=device, dtype=torch.float32)
        self.ents = torch.zeros((0, ARRANGEMENT_SIZE), device=device, dtype=torch.float32)
        self.log_probs = torch.zeros(
            (0, ARRANGEMENT_SIZE, N_PIECE_TYPE), device=device, dtype=torch.float32
        )
        self.needs_flip = torch.zeros(0, dtype=torch.bool, device=device)
        self.step_added = torch.zeros(0, dtype=torch.int, device=device)

        # Initialize appropriate generator to assign arrangement identifiers
        if barrage:
            self.gen = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.BARRAGE)
        else:
            self.gen = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.CLASSIC)

        # For logging
        self.quantiles_to_compute = UPPER_QUANTILES.clone().to(device)

    @torch.no_grad()
    def add_arrangements(
        self,
        arrangements: torch.Tensor,
        values: torch.Tensor,
        ents: torch.Tensor,
        log_probs: torch.Tensor,
        needs_flip: torch.Tensor,
        step: int,
    ) -> None:
        """Update buffer with new arrangements and associated creation-time data.

        Args:
            arrangements (B, ARRANGEMENT_SIZE, N_PIECE_TYPE): Arrangements (in one-hot representation) to add to buffer.
            values (B, *self.value_shape): Value predictions for each arrangement prefix.
            ents (B, ARRANGEMENT_SIZE): Future accumulated entropy predictions for each arrangement prefix.
            log_probs (B, ARRANGEMENT_SIZE, N_PIECE_TYPE): Log probabilities of possible piece placements for each arrangement prefix.
            needs_flip (B,): Whether each arrangement needs to be flipped to match the network orientation.
                The network (left/right) orientation of the arrangements is not guaranteed to match the envionmenment orientation of the arrangements.
                The buffer stores arrangements in the environment orientation.
            step (int): Environment step at which reset behavior was updated.
        """
        check_onehot_arrangements(arrangements)
        N = arrangements.shape[0]
        if not (values.shape == (N, *self.value_shape)):
            raise ValueError(f"values must have shape (N, *self.value_shape), got {values.shape}")
        if not (values.dtype == torch.float32):
            raise ValueError(f"values must be of type torch.float32, got {values.dtype}")
        if not (ents.shape == (N, ARRANGEMENT_SIZE)):
            raise ValueError(f"ents must have shape (N, ARRANGEMENT_SIZE), got {ents.shape}")
        if not (ents.dtype == torch.float32):
            raise ValueError(f"ents must be of type torch.float32, got {ents.dtype}")
        if not (log_probs.shape == (N, ARRANGEMENT_SIZE, N_PIECE_TYPE)):
            raise ValueError(
                f"log_probs must have shape (N, ARRANGEMENT_SIZE, N_PIECE_TYPE), got {log_probs.shape}"
            )
        if not (log_probs.dtype == torch.float32):
            raise ValueError(f"log_probs must be of type torch.float32, got {log_probs.dtype}")
        if not (needs_flip.shape == (N,)):
            raise ValueError(f"needs_flip must have shape (N,), got {needs_flip.shape}")
        if not (needs_flip.dtype == torch.bool):
            raise ValueError(f"needs_flip must be a boolean tensor, got {needs_flip.dtype}")
        if not isinstance(step, int):
            raise ValueError(f"step must be an integer, got {type(step)}")

        # Add new data to the buffer.
        self.arrangements = torch.cat((arrangements, self.arrangements), dim=0)
        if self.use_cat_vf:
            values = torch.softmax(values, dim=-1)
        self.values = torch.cat((values, self.values), dim=0)
        self.ents = torch.cat((ents, self.ents), dim=0)
        self.log_probs = torch.cat((log_probs, self.log_probs), dim=0)
        self.needs_flip = torch.cat((needs_flip, self.needs_flip), dim=0)
        self.step_added = torch.cat(
            (torch.full((arrangements.shape[0],), step, device=self.device), self.step_added), dim=0
        )

        # Filter duplicates by age (ie keep the most recent instance)
        old_arrangement_ids = self.gen.arrangement_ids(
            self.arrangements.argmax(dim=-1).type(torch.uint8)
        )
        mask = torch.tensor(
            mark_most_recent_appearance(old_arrangement_ids, self.step_added), device=self.device
        )
        self.arrangements = self.arrangements[mask]
        self.values = self.values[mask]
        self.ents = self.ents[mask]
        self.log_probs = self.log_probs[mask]
        self.needs_flip = self.needs_flip[mask]
        self.step_added = self.step_added[mask]

        # Set tensor sizes for data collection
        self.counts = torch.zeros(
            (self.arrangements.size(0), *self.count_shape),
            device=self.device,
        )
        self.rewards = torch.zeros(
            self.arrangements.size(0), *self.reward_shape, device=self.device, dtype=torch.float32
        )
        self.ready_flags = torch.zeros(
            self.arrangements.size(0), dtype=torch.bool, device=self.device
        )

        # Set tensor sizes for post-processing
        self.adv_est = torch.zeros(
            (self.arrangements.size(0), ARRANGEMENT_SIZE), device=self.device, dtype=torch.float32
        )
        self.val_est = torch.zeros(
            (self.arrangements.size(0), *self.value_shape), device=self.device, dtype=torch.float32
        )
        self.reg_val_est = torch.zeros(
            (self.arrangements.size(0), ARRANGEMENT_SIZE), device=self.device, dtype=torch.float32
        )

        # Update lookup table
        self.lookup_indices = lookup_indices_factory(
            self.gen.arrangement_ids(self.arrangements.argmax(dim=-1).type(torch.uint8))
        )

        # Update buffer mode
        self.need_arrangements = False

    @torch.no_grad()
    def add_rewards(
        self,
        arrangements: torch.Tensor,
        is_newly_terminal: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        """Update buffer with game outcomes.

        Args:
            arrangements (B, ARRANGEMENT_SIZE): Arrangements (in integer representation) for which games are running.
            is_newly_terminal (B,): Whether each game terminated on the immediately previous step.
            rewards (B,): Rewards for each game.
        """
        N = arrangements.shape[0]
        if not (arrangements.shape == (N, ARRANGEMENT_SIZE)):
            raise ValueError(
                f"arrangements must have shape (N, ARRANGEMENT_SIZE), got {arrangements.shape}"
            )
        if not (is_newly_terminal.shape == (N,)):
            raise ValueError(
                f"is_newly_terminal must have shape (N,), got {is_newly_terminal.shape}"
            )
        if not (is_newly_terminal.dtype == torch.bool):
            raise ValueError(
                f"is_newly_terminal must be a boolean tensor, got {is_newly_terminal.dtype}"
            )
        if not (rewards.shape == (N,)):
            raise ValueError(f"rewards must have shape (N,), got {rewards.shape}")
        if self.need_arrangements:
            raise RuntimeError("Buffer needs new arrangements")

        # If no games have finished, do nothing. Otherwise, select the arrangements and rewards for which games have finished.
        if not is_newly_terminal.any():
            return
        arrangements = arrangements[is_newly_terminal].type(torch.uint8)
        rewards = rewards[is_newly_terminal]

        # Update reward shape for value function flavor if necessary.
        if self.use_cat_vf:
            # +1 below maps scalar rewards (-1, 0, 1) to categorical indices (0, 1, 2)
            rewards = F.one_hot((rewards + 1).long(), num_classes=N_VF_CAT).to(torch.float32)

        # Update information for selected arrangements
        idx = torch.tensor(
            self.lookup_indices(self.gen.arrangement_ids(arrangements)), device=self.device
        )
        self.rewards[idx] = (self.counts[idx] * self.rewards[idx] + rewards) / (
            self.counts[idx] + 1
        )
        self.counts[idx] += 1
        self.ready_flags[idx] = True

    @torch.no_grad()
    def process_data(
        self, td_lambda: float, gae_lambda: float, reg_temp: float, reg_norm: float
    ) -> dict[str, float]:
        """Compute value and advantage estimates for training.

        Args:
            td_lambda: Parameter for TD(lambda).
            gae_lambda: Parameter for GAE(lambda).
            reg_temp: Regularization temperature.
                Objective is return + reg_temp * entropy.
            reg_norm: Normalization for network entropy prediction.
                Network predicts entropy / reg_norm.

        Returns:
            Statistics for logging.
        """
        expect_type(td_lambda, float, name="td_lambda")
        expect_type(gae_lambda, float, name="gae_lambda")
        expect_type(reg_temp, float, name="reg_temp")
        expect_type(reg_norm, float, name="reg_norm")

        if self.need_arrangements:
            raise RuntimeError("Buffer needs new arrangements")
        if not self.ready_flags.any():
            return {}

        # Select rows that are ready for training.
        rewards = self.rewards[self.ready_flags]
        values = self.values[self.ready_flags]
        ents = (
            reg_norm * self.ents[self.ready_flags]
        )  # Multiplying by reg_norm gives network entropy prediction.
        N = self.ready_flags.nonzero().size(0)
        adv_est = torch.zeros(N, *self.value_shape, device=self.device, dtype=torch.float32)
        val_est = torch.zeros(N, *self.value_shape, device=self.device, dtype=torch.float32)
        reg_val_est = torch.zeros(N, ARRANGEMENT_SIZE, device=self.device, dtype=torch.float32)

        # Estimate grounded components of values and advantages.
        for step in range(ARRANGEMENT_SIZE - 1, -1, -1):
            if step == ARRANGEMENT_SIZE - 1:
                delta = rewards - values[:, step]
                td_trace = delta
                gae_trace = delta
            else:
                delta = values[:, step + 1] - values[:, step]
                td_trace = delta + td_lambda * td_trace
                gae_trace = delta + gae_lambda * gae_trace
            val_est[:, step] = td_trace + values[:, step]
            adv_est[:, step] = gae_trace
        # NOTE: Beyond this point, adv_est is guaranteed to be shape (N, ARRANGEMENT_SIZE).
        if self.use_cat_vf:
            adv_est = adv_est @ self.categorical_aggregation

        # Compute the negative log likelihood of the arrangements.
        nll = -self.log_probs[self.ready_flags]
        # Log probs stores the log likelihood of each possible piece placement.
        # We want the only log probs of the piece placement that was actually made, so we need to do a gather.
        arr = self.arrangements[self.ready_flags]
        # The arrangements are stored in environment orientation, but the log probs are stored in network orientation,
        # so we need to flip before gathering.
        needs_flip = self.needs_flip[self.ready_flags]
        arr[needs_flip] = flip_arrangements(arr[needs_flip])
        nll = nll.gather(-1, arr.argmax(dim=-1).long().unsqueeze(-1)).squeeze(-1)

        # Estimate values and advantages for regularization.
        reg_td_trace = torch.zeros(1, device=self.device)
        reg_gae_trace = torch.zeros(1, device=self.device)
        for step in range(ARRANGEMENT_SIZE - 1, -1, -1):
            if step == ARRANGEMENT_SIZE - 1:
                delta = nll[:, step] - ents[:, step]
                reg_td_trace = delta
                reg_gae_trace = delta
            else:
                delta = nll[:, step] + ents[:, step + 1] - ents[:, step]
                reg_td_trace = delta + td_lambda * reg_td_trace
                reg_gae_trace = delta + gae_lambda * reg_gae_trace
            reg_val_est[:, step] = reg_td_trace + ents[:, step]
            # Need to mutate instead of assigning to retain grounded component of advantage estimate.
            adv_est[:, step] += reg_temp * reg_gae_trace

        # Update estimate tensors.
        self.adv_est[self.ready_flags] = adv_est
        self.val_est[self.ready_flags] = val_est
        self.reg_val_est[self.ready_flags] = (
            reg_val_est / reg_norm
        )  # Renormalize regularization estimate for network prediction.

        # Compute statistics for logging.
        abs_adv = adv_est.abs()
        if self.use_cat_vf:
            values = values @ self.categorical_aggregation
        abs_val = values.abs()
        nll = nll.sum(dim=-1)
        return {
            f"init_{name}/{round(q.item(), 3)}th_quantile": v.item()
            for name, x in zip(["abs_adv", "abs_val", "nll"], [abs_adv, abs_val, nll])
            for q, v in zip(
                self.quantiles_to_compute, x.float().quantile(self.quantiles_to_compute)
            )
        }

    def sample(self, batch_size: int) -> Generator[Batch, None, None]:
        """Yield shuffled minibatches of rows that are ready for training."""
        if self.need_arrangements:
            raise RuntimeError("Buffer is not ready to sample")
        expect_type(batch_size, int, name="batch_size")
        if not (batch_size > 0):
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Select rows that are ready for training
        arrangements = self.arrangements[self.ready_flags]  # NOTE:
        log_probs = self.log_probs[self.ready_flags]
        needs_flip = self.needs_flip[self.ready_flags]
        val_est = self.val_est[self.ready_flags]
        reg_val_est = self.reg_val_est[self.ready_flags]
        adv_est = self.adv_est[self.ready_flags]

        # Flip arrangements whose environment and network orientations differ.
        arrangements[needs_flip] = flip_arrangements(arrangements[needs_flip])

        # Shuffle indices to randomize minibatches
        n_ready_examples = torch.nonzero(self.ready_flags).size(0)
        permuted_idx = torch.randperm(n_ready_examples, device=self.device)

        # Steam batches
        for i in range(0, n_ready_examples, batch_size):
            batch_idx = permuted_idx[i : i + batch_size]
            yield Batch(
                arrangements[batch_idx],
                log_probs[batch_idx],
                val_est[batch_idx],
                reg_val_est[batch_idx],
                adv_est[batch_idx],
            )

    def filter(self, current_step: int) -> None:
        """Filter rows whose age exceeds `storage_duration`."""
        expiration_step = self.step_added + self.storage_duration
        should_keep = expiration_step >= current_step
        for attr in (
            "arrangements",
            "values",
            "ents",
            "log_probs",
            "needs_flip",
            "step_added",
        ):
            setattr(self, attr, getattr(self, attr)[should_keep])
        self.need_arrangements = True


def lookup_indices_factory(reference: list[int]) -> Callable[[list[int]], list[int]]:
    """Create function that takes list of values and returns indices at which those values appear in `reference`."""
    value_to_index = {val: idx for idx, val in enumerate(reference)}

    def lookup_indices(values: list[int]) -> list[int]:
        """Map each value to an index at which it appears in `reference`.

        There is no guarantee about which index is returned for values appearing more than once.
        """
        indices = []
        for v in values:
            try:
                indices.append(value_to_index[v])
            except KeyError:
                raise ValueError(f"Value {v} not found in `reference`")
        return indices

    return lookup_indices


def mark_most_recent_appearance(values: list[int], timestamps: torch.Tensor) -> list[bool]:
    """Mark one index as True for each unique value in `values` among those with maximal timestamp.

    Args:
        values: Values to mark.
        timestamps: Timestamps of values.

    Returns:
        Mask that is True for exactly one index for each unique value in `values`.
            When there are duplicate values, the index whose timestamp is maximal is marked True.
            When there are duplicate values jointly with maximal timestamp, there is no guarantee
            about which index is marked True (except that it will be only one of them.)
    """
    expect_shape(timestamps, ndim=1, dims={0: len(values)}, name="timestamps")
    expect_dtype(timestamps, torch.long, name="timestamps")

    # Record the maximum timestamp for each value.
    max_timestamps: dict[int, int] = {}
    for value, timestamp in zip(values, timestamps):
        if value not in max_timestamps or timestamp > max_timestamps[value]:
            max_timestamps[value] = timestamp

    # Mark the index corresponding to the latest appearance of each value among those with maximal timestamp.
    seen: set[int] = set()
    mask: list[bool] = []
    for timestamp, value in zip(timestamps, values):
        if timestamp == max_timestamps[value] and value not in seen:
            seen.add(value)
            mask.append(True)
        else:
            mask.append(False)

    return mask
