from dataclasses import dataclass
from typing import Generator

import torch
import numpy as np

from pyengine.core.env import Stratego
from pyengine.utils.constants import N_PLAYER


@dataclass
class Batch:
    infostates: torch.Tensor
    piece_ids: torch.Tensor
    legal_actions: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    weight: torch.Tensor

    def filter_by_advantage(self, keep_rate: float):
        if keep_rate == 1:
            return
        assert keep_rate < 1 and keep_rate > 0
        bsize = self.advantages.size(0)
        _, topk_adv_idxs = torch.topk(self.advantages.abs(), int(keep_rate * bsize), dim=0)
        self.infostates = self.infostates[topk_adv_idxs]
        self.piece_ids = self.piece_ids[topk_adv_idxs]
        self.legal_actions = self.legal_actions[topk_adv_idxs]
        self.actions = self.actions[topk_adv_idxs]
        self.log_probs = self.log_probs[topk_adv_idxs]
        self.returns = self.returns[topk_adv_idxs]
        self.advantages = self.advantages[topk_adv_idxs]
        self.weight = self.weight[topk_adv_idxs]


class OldCircularBuffer:
    """Circular buffer to store trajectories for ALL players.

    The column is for the parallel environments
    The row is each player's record (s, a, r, term), alternating between players

    """

    def __init__(
        self,
        num_envs,
        traj_len,
        move_memory,
        use_cat_vf: bool,
        device,
        keep_rate: float = 1.0,
        dtype=torch.float32,
        debug_mode=0,
    ):
        self.num_col = num_envs
        self.move_memory = move_memory
        self.traj_len = traj_len
        self.num_row = traj_len * N_PLAYER
        self.use_cat_vf = use_cat_vf
        self.keep_rate = keep_rate
        self.device = device
        self.CATEGORICAL_AGGREGATION = torch.tensor(
            [-1, 0, 1], dtype=torch.float, device=self.device
        )

        self.curr_idx = 0
        self.stage = 0  # 0 means pre_act has not been called yet, 1 means pre_act has been called

        # data book
        self.obs_steps = [-1 for _ in range(self.num_row)]  # +1 to track last state
        shape = (self.num_row, self.num_col)
        value_shape = shape if not use_cat_vf else (self.num_row, self.num_col, 3)

        self.actions = torch.zeros(shape, dtype=torch.int32, device=device)

        # NOTE: we do not change the shape of the rewards that are coming from the CUDA
        # backend even when we are interested in categorical values.
        self.rewards = torch.zeros(shape, dtype=dtype, device=device)
        self.terminals = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(value_shape, dtype=dtype, device=device)
        self.log_probs = torch.zeros(shape, dtype=dtype, device=device)

        ## valid_inputs mask out data that should not be used for training
        ## initilize to 1 because the first step is guaranteed to be useful
        ## even after reset
        self.valid_inputs = torch.ones(shape, dtype=dtype, device=device)

        # additional value to be computed from the raw data
        self.returns = torch.zeros(value_shape, dtype=dtype, device=device)
        self.advantages = torch.zeros(shape, dtype=dtype, device=device)

        # for debug
        self.debug_mode = debug_mode
        self.ref_infostates: list[torch.Tensor] = [torch.zeros(1) for _ in range(self.num_row)]
        self.ref_legal_actions: list[torch.Tensor] = [torch.zeros(1) for _ in range(self.num_row)]
        self.ref_piece_ids: list[torch.Tensor] = [torch.zeros(1) for _ in range(self.num_row)]

        self.quantiles_to_compute = torch.tensor(
            [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0], device=device
        )

    def add_pre_act(self, obs_step, obs, piece_ids, legal_action):
        """
        Adds the current step to the buffer.
        """

        assert self.stage == 0
        self.stage = 1
        self.obs_steps[self.curr_idx] = obs_step
        if self.debug_mode:
            self.ref_infostates[self.curr_idx] = obs.clone()
            self.ref_legal_actions[self.curr_idx] = legal_action.clone()
            self.ref_piece_ids[self.curr_idx] = piece_ids.clone()

    def add_post_act(self, action, value, log_prob, reward, terminal):
        assert self.stage == 1
        self.stage = 0

        self.actions[self.curr_idx] = action
        self.values[self.curr_idx] = value
        self.log_probs[self.curr_idx] = log_prob
        self.rewards[self.curr_idx] = reward
        self.terminals[self.curr_idx] = terminal.int()

        # set valid_input & fix reward for previous step
        if self.curr_idx > 0:
            prev_idx = self.curr_idx - 1
            # valid input should be set before overwriting terminal & reward
            # this input is useless for training if:
            # 1. previous step terminates
            # 2. the current step is not a new game (i.e. also terminates)
            self.valid_inputs[self.curr_idx] = 1 - (
                self.terminals[prev_idx] * self.terminals[self.curr_idx]
            )
            # fix the reward & terminal for previous step, i.e. the opponent
            prev_not_term = 1 - self.terminals[prev_idx]
            # If the current step terminated the game
            # (prev_not_term=1,terminals[curr]=1,terminals[prev]=0->terminals[prev]=1)
            # we overwrite the previous terminal flag, since, from the perspective of
            # the player acting at that time step, their action terminated the game.
            # Otherwise, terminals[prev] is left unchanged.
            self.terminals[prev_idx] += prev_not_term * self.terminals[self.curr_idx]
            # The logic for rewards is the same as for terminals except we have to
            # flip the sign since the other player experiences the opposite outcome
            # as the terminating player.
            self.rewards[prev_idx] -= prev_not_term * self.rewards[self.curr_idx]

        self.curr_idx += 1

    def ready_to_train(self):
        # ready to train when:
        # 1. finish writing all num_row (traj_len) data
        # 2. finish writing the state of an extra step so that we can bootstrap from it
        return self.curr_idx == self.num_row and self.stage == 0

    def reset(self):
        assert self.ready_to_train()
        # copy the last step to the beginning of the buffer because
        # we have not yet trained on the last step because it has
        # no bootstrapping target
        for i in range(N_PLAYER):
            self.obs_steps[i] = self.obs_steps[-N_PLAYER + i]
            self.actions[i] = self.actions[-N_PLAYER + i]
            self.rewards[i] = self.rewards[-N_PLAYER + i]
            self.terminals[i] = self.terminals[-N_PLAYER + i]
            self.values[i] = self.values[-N_PLAYER + i]
            self.log_probs[i] = self.log_probs[-N_PLAYER + i]
            self.valid_inputs[i] = self.valid_inputs[-N_PLAYER + i]

            if self.debug_mode:
                self.ref_infostates[i] = self.ref_infostates[-N_PLAYER + i]
                self.ref_legal_actions[i] = self.ref_legal_actions[-N_PLAYER + i]
                self.ref_piece_ids[i] = self.ref_piece_ids[-N_PLAYER + i]

        self.stage = 0
        self.curr_idx = N_PLAYER

    def process_data(self, gamma: float, return_gae_lambda: float, adv_gae_lambda: float) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate
        - TD(0) is one-step estimate with bootstrapping
        """
        if not self.use_cat_vf:
            rewards = self.rewards.view(self.traj_len, -1)
            terminals = self.terminals.view(self.traj_len, -1)
            values = self.values.view(self.traj_len, -1)
            advantages = self.advantages.view(self.traj_len, -1)
            returns = self.returns.view(self.traj_len, -1)

            return_last_gae_lam = 0
            adv_last_gae_lam = 0
            for step in range(self.traj_len - 2, -1, -1):
                delta = (
                    terminals[step] * rewards[step]
                    + gamma * (1 - terminals[step]) * values[step + 1]
                    - values[step]
                )
                return_last_gae_lam = (
                    delta + gamma * return_gae_lambda * (1 - terminals[step]) * return_last_gae_lam
                )
                adv_last_gae_lam = (
                    delta + gamma * adv_gae_lambda * (1 - terminals[step]) * adv_last_gae_lam
                )
                returns[step] = return_last_gae_lam + values[step]
                advantages[step] = adv_last_gae_lam
        else:
            rewards = self.rewards.view(self.traj_len, -1)
            scalar_rewards = rewards
            # Check all rewards are in {-1, 0, 1}
            assert torch.allclose(
                rewards**3 - rewards,
                torch.tensor(0, device=self.device, dtype=rewards.dtype),
            ), "All rewards should be in {-1, 0, +1}"

            R2 = rewards**2
            rewards = torch.stack(
                [
                    (R2 - rewards) * 0.5,  # This is 1 where rewards == -1
                    (-R2 + 1),  # This is 1 where rewards == 0
                    (R2 + rewards) * 0.5,  # This is 1 where rewards == +1
                ],
                dim=-1,
            )

            terminals = self.terminals.view(self.traj_len, -1)
            values = torch.softmax(
                self.values.view(self.traj_len, -1, 3), dim=-1
            )  # The softmax is because the value head returns logits
            scalar_values = values @ self.CATEGORICAL_AGGREGATION
            advantages = self.advantages.view(self.traj_len, -1)
            returns = self.returns.view(self.traj_len, -1, 3)

            return_last_gae_lam = torch.zeros(
                1, returns.shape[-1], device=self.device, dtype=torch.float
            )
            for step in range(self.traj_len - 2, -1, -1):
                delta = (
                    terminals[step].unsqueeze(1) * rewards[step]
                    + (1 - terminals[step].unsqueeze(1)) * values[step + 1]
                    - values[step]
                )
                return_last_gae_lam = (
                    delta
                    + return_gae_lambda * (1 - terminals[step].unsqueeze(1)) * return_last_gae_lam
                )
                returns[step] = return_last_gae_lam + values[step]

            adv_last_gae_lam = 0
            for step in range(self.traj_len - 2, -1, -1):
                delta = (
                    terminals[step] * scalar_rewards[step]
                    + gamma * (1 - terminals[step]) * scalar_values[step + 1]
                    - scalar_values[step]
                )
                adv_last_gae_lam = delta + adv_gae_lambda * (1 - terminals[step]) * adv_last_gae_lam
                advantages[step] = adv_last_gae_lam

        valid_input = self.valid_inputs.bool()
        abs_adv = self.advantages[valid_input].abs()
        valid_input = valid_input.view(self.traj_len, -1)
        if self.use_cat_vf:
            abs_val = scalar_values[valid_input].abs()
        else:
            abs_val = values[valid_input].abs()

        return {
            f"abs_{name}/{round(q.item(), 3)}th_quantile": v.item()
            for name, x in zip(["adv", "val"], [abs_adv, abs_val])
            for q, v in zip(self.quantiles_to_compute, x.quantile(self.quantiles_to_compute))
        }

    def sample(
        self, batch_size: int, env: Stratego, train_pl0: int, train_pl1: int
    ) -> Generator[Batch, None, None]:
        """
        When this function is called, we have collected enough data for training.
        Note that we do not sample the last step in the buffer for training becasue
        the return and advantage of it is not computed
        """
        assert self.ready_to_train()
        assert (
            batch_size % self.num_col == 0
        ), f"batch_size has to be multiples of num_env {batch_size} % {self.num_col} "

        num_row_per_batch = batch_size // self.num_col
        if train_pl0 and train_pl1:
            rows = range(self.num_row - N_PLAYER)
        elif train_pl0:
            rows = range(0, self.num_row - N_PLAYER, 2)
        elif train_pl1:
            rows = range(1, self.num_row - N_PLAYER, 2)
        else:
            return []
        permuted_rows = np.random.permutation(rows)
        for i in range(0, len(permuted_rows), num_row_per_batch):
            row_batch = permuted_rows[i : i + num_row_per_batch]
            valid_inputs = []
            infostates = []
            piece_ids = []
            legal_actions = []
            actions = []
            log_probs = []
            returns = []
            advantages = []
            for row in row_batch:
                obs_step = self.obs_steps[row]

                # TODO: redundant clone & stack
                # it may be more efficient to first create the tensor and copy
                infostates.append(env.infostate_tensor(obs_step))
                legal_actions.append(env.legal_action_mask(obs_step))
                piece_ids.append(env.piece_ids(obs_step))
                actions.append(self.actions[row])
                log_probs.append(self.log_probs[row])
                returns.append(self.returns[row])
                advantages.append(self.advantages[row])
                valid_inputs.append(self.valid_inputs[row])

            valid_inputs = torch.cat(valid_inputs, dim=0).bool()
            batch = Batch(
                infostates=torch.cat(infostates, dim=0)[valid_inputs],
                piece_ids=torch.cat(piece_ids, dim=0)[valid_inputs],
                legal_actions=torch.cat(legal_actions, dim=0)[valid_inputs],
                actions=torch.cat(actions, dim=0)[valid_inputs],
                log_probs=torch.cat(log_probs, dim=0)[valid_inputs],
                returns=torch.cat(returns, dim=0)[valid_inputs],
                advantages=torch.cat(advantages, dim=0)[valid_inputs],
                weight=torch.ones_like(valid_inputs)[valid_inputs],
            )

            batch.filter_by_advantage(self.keep_rate)

            yield batch

    def verify(self, env: Stratego):
        assert self.debug_mode
        for i, step in enumerate(self.obs_steps):
            infostate = env.infostate_tensor(step)
            legal_action = env.legal_action_mask(step)
            piece_ids = env.piece_ids(step)

            ref_infostate = self.ref_infostates[i]
            ref_legal_action = self.ref_legal_actions[i]
            ref_piece_ids = self.ref_piece_ids[i]
            assert torch.allclose(infostate, ref_infostate)
            assert torch.allclose(legal_action, ref_legal_action)
            assert torch.allclose(piece_ids, ref_piece_ids)
