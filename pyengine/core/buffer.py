from dataclasses import dataclass
from typing import Generator, Optional
import random

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego
from pyengine.utils.constants import (
    N_PLAYER,
    CATEGORICAL_AGGREGATION,
    N_VF_CAT,
    UPPER_QUANTILES,
    UPPER_BOUND_N_LEGAL_ACTION,
)

pystratego = get_pystratego()


@dataclass
class Batch:
    infostates: torch.Tensor
    piece_ids: torch.Tensor
    legal_actions: torch.Tensor
    num_moves: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    adv_mask: Optional[torch.Tensor]

    def cat(self, other: "Batch") -> "Batch":
        return Batch(
            infostates=torch.cat([self.infostates, other.infostates], dim=0),
            piece_ids=torch.cat([self.piece_ids, other.piece_ids], dim=0),
            legal_actions=torch.cat([self.legal_actions, other.legal_actions], dim=0),
            num_moves=torch.cat([self.num_moves, other.num_moves], dim=0),
            actions=torch.cat([self.actions, other.actions], dim=0),
            log_probs=torch.cat([self.log_probs, other.log_probs], dim=0),
            returns=torch.cat([self.returns, other.returns], dim=0),
            advantages=torch.cat([self.advantages, other.advantages], dim=0),
            values=torch.cat([self.values, other.values], dim=0),
            adv_mask=torch.cat([self.adv_mask, other.adv_mask], dim=0),
        )

    def apply_mask(self) -> "Batch":
        return Batch(
            infostates=self.infostates[self.adv_mask],
            piece_ids=self.piece_ids[self.adv_mask],
            legal_actions=self.legal_actions[self.adv_mask],
            num_moves=self.num_moves[self.adv_mask],
            actions=self.actions[self.adv_mask],
            log_probs=self.log_probs[self.adv_mask],
            returns=self.returns[self.adv_mask],
            advantages=self.advantages[self.adv_mask],
            values=self.values[self.adv_mask],
            adv_mask=None,
        )


class CircularBuffer:
    def __init__(
        self,
        num_envs: int,
        traj_len: int,
        train_every_per_player: int,
        use_cat_vf: bool,
        device: torch.device,
        adv_filt_rate: float = 1.0,
        adv_filt_thresh: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        assert traj_len >= train_every_per_player
        self.num_row = N_PLAYER * traj_len
        self.traj_len = traj_len
        # NOTE: We train every N_PLAYER * (train_every_per_player - 1) steps, not every N_PLAYER * train_every_per_player steps, to support testing against legacy
        self.train_every_per_player = train_every_per_player
        self.use_cat_vf = use_cat_vf
        self.adv_filt_rate = adv_filt_rate
        self.adv_filt_thresh = adv_filt_thresh
        self.device = device
        self.dtype = dtype
        self.categorical_aggregation = CATEGORICAL_AGGREGATION.to(device, dtype=dtype)

        # Indexing
        self.curr_idx = 0
        self.last_idx = N_PLAYER  # Offset is to support testing against legacy

        # Counts for consistency checks
        self.n_pre_act_calls = 0
        self.n_post_act_calls = 0

        # Data
        shape = (self.num_row, num_envs)
        value_shape = shape if not use_cat_vf else (*shape, N_VF_CAT)
        self.steps = torch.full((self.num_row,), -1, dtype=torch.int32, device=device)
        self.num_moves = torch.full(shape, -1, dtype=torch.int32, device=device)
        self.is_terminated_position = torch.zeros(shape, dtype=torch.bool, device=device)
        self.actions = torch.zeros(shape, dtype=torch.int32, device=device)
        self.terminal_rewards = torch.zeros(shape, dtype=dtype, device=device)
        self.is_terminating_action = torch.zeros(shape, dtype=torch.bool, device=device)
        self.values = torch.zeros(value_shape, dtype=dtype, device=device)
        self.log_probs = torch.zeros(
            self.num_row, num_envs, UPPER_BOUND_N_LEGAL_ACTION, dtype=dtype, device=device
        )
        self.target_values = torch.zeros(value_shape, dtype=dtype, device=device)
        self.ready_rows = torch.zeros(self.num_row, dtype=torch.bool, device=device)

        # Additional quantities to be computed from the data
        self.returns = torch.zeros(value_shape, dtype=dtype, device=device)
        self.advantages = torch.zeros(shape, dtype=dtype, device=device)

        # For logging
        self.quantiles_to_log = UPPER_QUANTILES.clone().to(device)
        self.stats = {}

    def add_pre_act(
        self,
        step,
        num_moves,
        legal_action_mask,
        is_terminal,
    ):
        assert self.n_pre_act_calls == self.n_post_act_calls
        self.n_pre_act_calls += 1
        curr_idx = self.curr_idx % self.num_row
        self.steps[curr_idx] = step
        self.num_moves[curr_idx] = num_moves
        self.is_terminated_position[curr_idx] = is_terminal
        self.last_legal_action_mask = legal_action_mask

    def add_post_act(self, action, value, log_prob, reward, is_terminal):
        assert self.n_post_act_calls == self.n_pre_act_calls - 1
        self.n_post_act_calls += 1
        curr_idx = self.curr_idx % self.num_row

        self.actions[curr_idx] = action
        self.values[curr_idx] = value
        self.log_probs[curr_idx] = compress_log_probs(log_prob, self.last_legal_action_mask)
        self.terminal_rewards[curr_idx] = is_terminal * reward
        self.is_terminating_action[curr_idx] = is_terminal
        boundary_transition = is_terminal & ~self.is_terminated_position[curr_idx]

        if self.curr_idx > 0:
            prev_idx = (curr_idx - 1) % self.num_row
            # Basic consistency checks
            assert (
                self.is_terminating_action[prev_idx] == self.is_terminated_position[curr_idx]
            ).all(), "If the current position is terminated, the previous action should be recorded as terminating"
            dummy_transition = self.is_terminated_position[curr_idx] & is_terminal
            assert self.terminal_rewards[prev_idx, dummy_transition].allclose(
                -self.terminal_rewards[curr_idx, dummy_transition]
            ), "The terminal reward for successive terminated positions is inconsistent."
            # Because we train the players solipsistically, each should view its own action as having terminated the game and triggered the terminal reward.
            # Thus, when the acting player's action (objectively) terminates the game, we overwrite the data for the previous player as if its action had terminated the game.
            self.terminal_rewards[prev_idx, boundary_transition] -= self.terminal_rewards[
                curr_idx, boundary_transition
            ]  # Flip sign b/c previous idx belong to opponent
            self.is_terminating_action[prev_idx, boundary_transition] |= is_terminal[
                boundary_transition
            ]

        # Construct target values for the acting player's previous position
        if self.curr_idx > 1:
            prev_prev_idx = (curr_idx - N_PLAYER) % self.num_row
            if self.use_cat_vf:
                self.target_values[prev_prev_idx] = torch.where(
                    self.is_terminating_action[prev_prev_idx].unsqueeze(1),
                    torch.nn.functional.one_hot(
                        (self.terminal_rewards[prev_prev_idx] + 1).long(), num_classes=N_VF_CAT
                    ),
                    torch.softmax(self.values[curr_idx], dim=-1),
                )
            else:
                self.target_values[prev_prev_idx] = torch.where(
                    self.is_terminating_action[prev_prev_idx],
                    self.terminal_rewards[prev_prev_idx],
                    self.values[curr_idx],
                )
            # Update training status of previous previous position
            self.ready_rows[prev_prev_idx] = True

        self.curr_idx += 1

    def ready_to_train(self):
        return (
            self.curr_idx
            == self.last_idx
            + N_PLAYER
            * (self.train_every_per_player - 1)  # Offset by 1 is to support testing against legacy
        ) and self.n_pre_act_calls == self.n_post_act_calls == self.curr_idx

    def process_data(self, td_lambda: float, gae_lambda: float) -> None:
        """Process the collected data.

        Compute returns, advantages, advantage filtering mask, and data statistics.
        """
        # Variables whose shapes depend on value function flavor.
        if self.use_cat_vf:
            values = torch.softmax(self.values.view(self.traj_len, -1, N_VF_CAT), dim=-1)
            scalar_values = values @ self.categorical_aggregation
            target_values = self.target_values.view(self.traj_len, -1, N_VF_CAT)
            scalar_target_values = target_values @ self.categorical_aggregation
            returns = self.returns.view(self.traj_len, -1, N_VF_CAT)
            expanded_terminals = self.is_terminating_action.view(self.traj_len, -1, 1)
        else:
            values = self.values.view(self.traj_len, -1)
            scalar_values = values
            target_values = self.target_values.view(self.traj_len, -1)
            scalar_target_values = target_values
            returns = self.returns.view(self.traj_len, -1)
            expanded_terminals = self.is_terminating_action.view(self.traj_len, -1)

        # Variables whose shapes are independent of value function flavor.
        advantages = self.advantages.view(self.traj_len, -1)
        terminals = self.is_terminating_action.view(self.traj_len, -1)
        curr_step = (self.curr_idx % self.num_row) // N_PLAYER

        # Compute returns and advantages.
        indices = torch.tensor(
            modular_span(curr_step - self.traj_len, curr_step - 1, self.traj_len),
            device=self.device,
        )
        delta = target_values[indices] - values[indices]
        scalar_delta = scalar_target_values[indices] - scalar_values[indices]
        td_lambdas = td_lambda * (~expanded_terminals[indices])
        gae_lambdas = gae_lambda * (~terminals[indices])
        returns[indices] = segmented_discounted_cumsum(delta, td_lambdas) + values[indices]
        advantages[indices] = segmented_discounted_cumsum(scalar_delta, gae_lambdas)

        # Compute absolute advantage statistics.
        mask = ~self.is_terminated_position
        abs_adv = self.advantages[mask].abs()
        mask_ = mask.view(self.traj_len, -1)
        if self.use_cat_vf:
            abs_val = scalar_values[mask_].abs()
        else:
            abs_val = values[mask_].abs()
        threshold = max(abs_adv.float().quantile(self.adv_filt_rate), self.adv_filt_thresh)
        self.adv_mask = mask & (self.advantages.abs() >= threshold)

        return {
            f"buffer_abs_{name}/{round(q.item(), 3)}th_quantile": v.item()
            for name, x in zip(["adv", "val"], [abs_adv, abs_val])
            for q, v in zip(self.quantiles_to_log, x.float().quantile(self.quantiles_to_log))
        }

    def reset(self):
        assert self.ready_to_train()
        self.last_idx = self.curr_idx
        self.ready_rows.fill_(False)

    def sample(self, env: Stratego) -> Generator[Optional[Batch], None, None]:
        assert self.ready_to_train()

        ready_row_indices = [r for r in range(self.num_row) if self.ready_rows[r]]
        random.shuffle(ready_row_indices)

        for row in ready_row_indices:
            step = self.steps[row]
            mask = ~self.is_terminated_position[row]

            if not self.adv_mask[row][mask].any():
                continue

            yield (
                Batch(
                    infostates=env.infostate_tensor(step)[mask],
                    piece_ids=env.piece_ids(step)[mask],
                    num_moves=self.num_moves[row][mask],
                    legal_actions=env.legal_action_mask(step)[mask],
                    actions=self.actions[row][mask],
                    log_probs=expand_log_probs(self.log_probs[row], env.legal_action_mask(step))[
                        mask
                    ],
                    returns=self.returns[row][mask],
                    advantages=self.advantages[row][mask],
                    values=self.values[row][mask],
                    adv_mask=self.adv_mask[row][mask],
                )
            )

    def n_batch(self) -> int:
        return self.adv_mask.any(dim=-1)[self.ready_rows].sum()


def modular_span(lower, upper, mod):
    assert lower <= upper
    return [x % mod for x in range(lower, upper)]


def compress_log_probs(
    log_probs: torch.Tensor,
    legal_action_mask: torch.BoolTensor,
) -> torch.Tensor:
    """Store log probabilities of legal actions in block

    log_probs: (B, C)
    legal_action_mask: (B, C)
        legal_action_mask.sum(-1) <= D
    returns: (B, D) tensor containing compressed log probabilities
    """
    assert log_probs.ndim == 2
    assert legal_action_mask.ndim == 2
    assert UPPER_BOUND_N_LEGAL_ACTION < log_probs.shape[1] == legal_action_mask.shape[1]
    assert (legal_action_mask.sum(-1) <= UPPER_BOUND_N_LEGAL_ACTION).all()

    B = log_probs.shape[0]
    compressed = torch.zeros(
        (B, UPPER_BOUND_N_LEGAL_ACTION), dtype=log_probs.dtype, device=log_probs.device
    )
    rows, cols = torch.where(legal_action_mask)
    if rows.numel():
        d_idx = (legal_action_mask.cumsum(dim=1) - 1)[rows, cols]
        compressed[rows, d_idx] = log_probs[rows, cols]
    expanded = expand_log_probs(compressed, legal_action_mask)
    assert expanded.shape == log_probs.shape
    assert expanded.isclose(log_probs)[
        legal_action_mask
    ].all(), "Compressed log probs are not consistent with the original log probs"
    return compressed


def expand_log_probs(
    block: torch.Tensor, legal_action_mask: torch.BoolTensor, fill_value: float = -1e10
) -> torch.Tensor:
    """Expand block into full log_probs tensor

    block: (B, D)
    legal_action_mask: (B, C)
        legal_action_mask.sum(-1) <= D

    returns: (B, C)
    """
    assert block.ndim == 2
    assert legal_action_mask.ndim == 2
    assert block.shape[0] == legal_action_mask.shape[0]
    assert block.shape[1] < legal_action_mask.shape[1]
    assert (legal_action_mask.sum(-1) <= block.shape[1]).all()

    B, C = legal_action_mask.shape
    log_probs = torch.full((B, C), fill_value, dtype=block.dtype, device=block.device)
    rows, cols = torch.where(legal_action_mask)
    if rows.numel():
        d_idx = (legal_action_mask.cumsum(dim=1) - 1)[rows, cols]
        log_probs[rows, cols] = block[rows, d_idx]
    return log_probs


@torch.jit.script
def segmented_discounted_cumsum(x: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    y[-1] = x[-1]
    for t in range(x.size(0) - 2, -1, -1):
        y[t] = x[t] + discount[t] * y[t + 1]
    return y
