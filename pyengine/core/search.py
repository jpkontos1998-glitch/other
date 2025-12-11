from typing import Optional, Tuple
from collections import defaultdict
import contextlib
import math

import torch
from torch.amp import autocast
from torch import nn
from torch.distributions import Categorical

from pyengine.utils.types import GenerateArgType
from pyengine.belief.sampling import marginalize
from pyengine.core.env import Stratego
from pyengine.networks.legacy_rl import TransformerRL
from pyengine.utils.constants import CATEGORICAL_AGGREGATION, N_PLAYER, N_VF_CAT
from pyengine import utils

pystratego = utils.get_pystratego()


def sample_deterministic(policy, total_samples):
    expected_counts = (policy * total_samples).round().int()

    # Adjust the counts to ensure the total is exactly num_samples
    while expected_counts.sum() < total_samples:
        # Find the category with the largest discrepancy from its expected value
        discrepancies = policy * total_samples - expected_counts.float()
        idx = discrepancies.argmax()
        expected_counts[idx] += 1

    while expected_counts.sum() > total_samples:
        # Find the category with the smallest discrepancy from its expected value
        discrepancies = expected_counts.float() - policy * total_samples
        idx = discrepancies.argmax()
        expected_counts[idx] -= 1

    # Construct the sample tensor
    samples = torch.cat([torch.full((count,), i) for i, count in enumerate(expected_counts)])

    return samples.int()


def get_stats(action_mask, q_estimates, bp_policy, search_policy, counts):
    str2val = {}
    win_prob = 0
    for i in range(len(action_mask)):
        if action_mask[i]:
            cell_idx = i % 100
            row = cell_idx // 10
            col = cell_idx % 10
            horizontal = i >= 900
            new_coord = (i // 100) % 9
            if not horizontal:
                new_col = col
                new_row = new_coord + int(new_coord >= row)
            else:
                new_row = row
                new_col = new_coord + int(new_coord >= col)
            str2val[
                f"Action {i}: {(row, col)} -> {(new_row, new_col)}; Q {round(q_estimates[i].item(), 2)}; bp {round(bp_policy[i].item(), 2)}, pi {round(search_policy[i].item(), 2)}; samples {counts[i].item()}"
            ] = search_policy[i]
            win_prob += search_policy[i] * (q_estimates[i] + 1) / 2
    return win_prob


class SearchBot:
    def __init__(
        self,
        rl_model: TransformerRL,
        env: Stratego,
        depth: int,
        stepsize: float,
        temperature: float,
        td_lambda: float,
        max_num_samples: int,
        uniform_magnet: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        belief_model: Optional[nn.Module] = None,
    ) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        if depth % 2 == 1:
            raise ValueError("depth must be even")
        if env.num_envs < 100:
            raise ValueError("env.num_envs must be at least 100")
        if belief_model is None:
            print("WARNING: No belief model provided, will use ground truth state for search.")
        if td_lambda < 0 or td_lambda > 1:
            raise ValueError("td_lambda must be between 0 and 1")

        self.rl_model = rl_model
        self.env = env
        self.auxiliary_env = Stratego(
            num_envs=env.conf.max_num_moves // N_PLAYER,
            traj_len_per_player=N_PLAYER,
            quiet=2,
            max_num_moves_between_attacks=env.conf.max_num_moves_between_attacks,
            max_num_moves=env.conf.max_num_moves,
            nonsteppable=True,
            cuda_device=env.conf.cuda_device,
        )
        self.depth = depth
        self.stepsize = stepsize
        self.temperature = temperature
        self.td_lambda = td_lambda
        self.max_num_samples = max_num_samples
        self.uniform_magnet = uniform_magnet
        self.dtype = dtype
        self.belief_model = belief_model
        self.categorical_aggregation = CATEGORICAL_AGGREGATION.to("cuda").to(torch.float32)
        self.stats: defaultdict[str, list] = defaultdict(list)
        self.stopwatch = utils.Stopwatch()
        self._reset_last_info()

    def __call__(
        self,
        env_states: pystratego.EnvState,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        legal_action_mask: torch.Tensor,
        num_moves: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
    ) -> torch.Tensor:
        self._reset_last_info()

        # 1. Prepare rollout states
        n_sample = min(self.env.num_envs // legal_action_mask.sum().item(), self.max_num_samples)
        with self.stopwatch.time("generate_env_state"):
            posterior_env_states = self._generate_env_state(
                env_states,
                infostate_tensor,
                piece_ids,
                num_moves,
                unknown_piece_position_onehot,
                unknown_piece_counts,
                unknown_piece_has_moved,
                n_sample,
            )

        # 2. Estimate Q-values
        with self.stopwatch.time("estimate_q_values"):
            q_estimates, search_counts = self.estimate_q_values(
                posterior_env_states,
                legal_action_mask.squeeze(0),
                self.rl_model,
                self.rl_model,
                self.depth,
                self.td_lambda,
            )

        # 3. Compute search policy
        with torch.no_grad(), utils.eval_mode(self.rl_model):
            tensor_dict = self.rl_model(infostate_tensor, piece_ids, legal_action_mask, num_moves)
            bp_logits_root = tensor_dict["action_log_probs"].squeeze(0)
            v_logits_root = tensor_dict["value"].squeeze(0)
        magnet_policy = utils.helper.get_weighted_uniform_policy(legal_action_mask)[0]
        search_policy = compute_search_policy(
            q_estimates,
            bp_logits_root,
            legal_action_mask.squeeze(0),
            self.temperature,
            self.stepsize,
            self.uniform_magnet,
        )

        # 4. Diagnostics
        self._record_stats(
            bp_logits_root.exp(),
            magnet_policy,
            search_policy,
            q_estimates,
            legal_action_mask.squeeze(0),
            v_logits_root,
            search_counts,
        )
        return Categorical(probs=search_policy).sample().int().view(1)

    def estimate_q_values(
        self,
        env_state: pystratego.EnvState,
        action_mask: torch.Tensor,
        player_model: nn.Module,
        opponent_model: nn.Module,
        depth: int,
        td_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = action_mask.shape[0]
        uniform_policy = action_mask.int() / action_mask.sum()

        self.env.change_reset_behavior_to_env_state(env_state)
        self.env.reset()
        player = env_state.to_play
        if self.rl_model.cfg.use_cat_vf:
            leaf_vals = torch.zeros(self.env.num_envs, N_VF_CAT, device="cuda", dtype=torch.float32)
            done = self.env.current_is_terminal.view(-1, 1)
            q = torch.zeros(A, N_VF_CAT, device="cuda", dtype=torch.float32)
            cum = torch.zeros_like(q)
            counts = torch.zeros(A, device="cuda", dtype=torch.float32)
        else:
            leaf_vals = torch.zeros(self.env.num_envs, device="cuda", dtype=torch.float32)
            done = self.env.current_is_terminal
            q = torch.zeros(A, device="cuda", dtype=torch.float32)
            cum = torch.zeros_like(q)
            counts = torch.zeros(A, device="cuda", dtype=torch.float32)

        actions = sample_deterministic(uniform_policy, self.env.num_envs).cuda()
        tracked = actions.long()
        for d in range(depth // 2):
            for parity in range(N_PLAYER):
                self.env.apply_actions(actions)
                with autocast(
                    device_type="cuda", dtype=self.dtype
                ), torch.no_grad(), utils.eval_mode(player_model):
                    model = player_model if self.env.current_player == player else opponent_model
                    tensor_dict = model(
                        self.env.current_infostate_tensor,
                        self.env.current_piece_ids,
                        self.env.current_legal_action_mask,
                        self.env.current_num_moves,
                    )
                    actions = tensor_dict["action"]
                    values = tensor_dict["value"]

                if parity == 0:
                    reward = self.env.current_is_newly_terminal * self.env.current_reward_pl0
                    continue

                reward += self.env.current_is_newly_terminal * self.env.current_reward_pl0
                if player == 1:
                    reward *= -1
                term = self.env.current_is_terminal
                if self.rl_model.cfg.use_cat_vf:
                    reward = torch.nn.functional.one_hot((reward + 1).long(), num_classes=N_VF_CAT)
                    values = torch.softmax(values.to(torch.float32), dim=-1)
                    term = term.view(-1, 1)
                assert (
                    player == self.env.current_player
                ), "We should only be accumulating value network outputs on the search player's turn."

                if td_lambda < 1:
                    if d < depth // 2 - 1:
                        normalizer = (1 - td_lambda) * (td_lambda**d)
                    else:
                        normalizer = td_lambda**d
                    target = torch.where(term, (td_lambda**d) * reward, normalizer * values)
                else:
                    target = torch.where(term, reward, (d == depth // 2 - 1) * values)
                leaf_vals += (~done) * target

                done |= term
                if done.all():
                    break

        if self.rl_model.cfg.use_cat_vf:
            cum.scatter_add_(
                0, tracked.unsqueeze(-1).expand(-1, N_VF_CAT), leaf_vals
            )  # NOTE: Need to be careful with broadcasting here.
            counts += torch.bincount(tracked.squeeze(), minlength=A)
            nz = counts > 0
            q[nz] = cum[nz] / counts[nz].unsqueeze(1)
            if not torch.allclose(
                q[nz].sum(dim=-1), torch.tensor([1], device=q.device, dtype=q.dtype)
            ):
                print("WARNING: Categorical Q-values do not sum to 1")
            self.last_search_info["cat_q"] = q
            scalar_q = q @ self.categorical_aggregation
            self.last_search_info["rollout_values"] = (
                leaf_vals @ self.categorical_aggregation
            ).cpu()
            return scalar_q, counts
        else:
            cum.scatter_add_(0, tracked, leaf_vals)
            counts += torch.bincount(tracked.squeeze(), minlength=A)
            nz = counts > 0
            q[nz] = cum[nz] / counts[nz]
            self.last_search_info["rollout_values"] = leaf_vals.cpu()
            return q, counts

    def _generate_env_state(
        self,
        env_state: pystratego.EnvState,
        infostate_tensor: torch.Tensor,
        piece_ids: torch.Tensor,
        num_moves: torch.Tensor,
        unknown_piece_position_onehot: torch.Tensor,
        unknown_piece_counts: torch.Tensor,
        unknown_piece_has_moved: torch.Tensor,
        samples_per_action: int,
    ) -> Optional[pystratego.EnvState]:
        if (unknown_piece_position_onehot.sum() > 0) and (self.belief_model is not None):
            with torch.no_grad(), autocast(device_type="cuda", dtype=self.dtype):
                if isinstance(self.belief_model, nn.Module):
                    eval_ctx = utils.eval_mode(self.belief_model)
                else:
                    eval_ctx = contextlib.nullcontext()
                with self.stopwatch.time("generate_env_state/sample_belief"):
                    with eval_ctx:
                        if self.belief_model.generate_arg_type == GenerateArgType.MARGLIZED_UNIFORM:
                            samples = self.belief_model.generate(
                                n_sample=samples_per_action,
                                unknown_piece_position_onehot=unknown_piece_position_onehot.squeeze(
                                    0
                                ),
                                unknown_piece_has_moved=unknown_piece_has_moved.squeeze(0),
                                unknown_piece_counts=unknown_piece_counts.squeeze(0),
                                infostate_tensor=infostate_tensor.squeeze(0),
                            )
                        elif self.belief_model.generate_arg_type == GenerateArgType.UNIFORM:
                            samples = self.belief_model.generate(
                                n_sample=samples_per_action,
                                unknown_piece_position_onehot=unknown_piece_position_onehot.squeeze(
                                    0
                                ),
                                unknown_piece_has_moved=unknown_piece_has_moved.squeeze(0),
                                unknown_piece_counts=unknown_piece_counts.squeeze(0),
                            )
                        elif (
                            self.belief_model.generate_arg_type
                            == GenerateArgType.PLANAR_TRANSFORMER
                        ):
                            samples = self.belief_model.generate(
                                n_sample=samples_per_action,
                                unknown_piece_position_onehot=unknown_piece_position_onehot.squeeze(
                                    0
                                ),
                                unknown_piece_has_moved=unknown_piece_has_moved.squeeze(0),
                                unknown_piece_counts=unknown_piece_counts.squeeze(0),
                                infostate_tensor=infostate_tensor.squeeze(0),
                                piece_ids=piece_ids.squeeze(0),
                            )
                        else:
                            assert (
                                self.belief_model.generate_arg_type
                                == GenerateArgType.TEMPORAL_TRANSFORMER
                            )
                            s = env_state.clone()
                            state_num_envs = s.num_envs
                            s.tile(math.ceil(self.auxiliary_env.num_envs / state_num_envs))
                            self.auxiliary_env.change_reset_behavior_to_env_state(
                                s.slice(0, self.auxiliary_env.num_envs)
                            )
                            samples = self.belief_model.generate(
                                n_sample=samples_per_action,
                                unknown_piece_position_onehot=unknown_piece_position_onehot.squeeze(
                                    0
                                ),
                                unknown_piece_has_moved=unknown_piece_has_moved.squeeze(0),
                                unknown_piece_counts=unknown_piece_counts.squeeze(0),
                                infostate_tensor=self.auxiliary_env.current_infostate_tensor[
                                    :state_num_envs
                                ],
                                piece_ids=self.auxiliary_env.current_piece_ids[:state_num_envs],
                                num_moves=self.auxiliary_env.current_num_moves[:state_num_envs],
                            )
            self.last_search_info["belief"] = marginalize(samples).cpu()
            samples = samples.repeat(self.env.num_envs // samples_per_action + 1, 1, 1)[
                : self.env.num_envs
            ]
            with self.stopwatch.time("generate_env_state/assign_opponent_hidden_pieces"):
                if env_state.num_envs > 1:
                    env_state = env_state.slice(env_state.num_envs - 1, env_state.num_envs)
                env_state = pystratego.util.assign_opponent_hidden_pieces(
                    env_state, samples.to(torch.uint8)
                )
            return env_state

        env_state.tile(self.env.num_envs)
        return env_state

    def _record_stats(
        self,
        bp: torch.Tensor,
        magnet: torch.Tensor,
        search: torch.Tensor,
        q: torch.Tensor,
        mask: torch.Tensor,
        v_bp: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        cat = torch.distributions.Categorical
        kl = torch.distributions.kl.kl_divergence
        self.stats["actions_searched"].append(mask.sum().item())
        self.stats["bp_ent"].append(cat(probs=bp).entropy().mean().item())
        self.stats["magnet_ent"].append(cat(probs=magnet).entropy().mean().item())
        self.stats["search_ent"].append(cat(probs=search).entropy().mean().item())
        self.stats["search_bp_kl"].append(kl(cat(probs=search), cat(probs=bp)).mean().item())
        self.stats["search_magnet_kl"].append(
            kl(cat(probs=search), cat(probs=magnet)).mean().item()
        )
        self.stats["bp_magnet_kl"].append(kl(cat(probs=bp), cat(probs=magnet)).mean().item())
        self.stats["ent_diff"].append(self.stats["search_ent"][-1] - self.stats["bp_ent"][-1])
        self.stats["magnet_kl_diff"].append(
            self.stats["search_magnet_kl"][-1] - self.stats["bp_magnet_kl"][-1]
        )
        self.stats["l1_diff"].append((search - bp).abs().sum().item())
        self.stats["linf_diff"].append((search - bp).abs().max().item())
        self.stats["ev_diff"].append((q[mask] * (search[mask] - bp[mask])).sum().item())
        self.stats["policy_net_regret"].append(
            q[mask].max().item() - (q[mask] * bp[mask]).sum().item()
        )
        self.stats["search_regret"].append(
            q[mask].max().item() - (q[mask] * search[mask]).sum().item()
        )

        # Store full tensors for external inspection
        self.last_search_info.update(
            {
                "v_bp": v_bp.exp().cpu(),
                "pi_bp": bp.cpu(),
                "pi_search": search.cpu(),
                "q_search": q.cpu(),
                "counts": counts.int().cpu(),
            }
        )

    def _reset_last_info(self) -> None:
        self.last_search_info = {
            "v_bp": None,
            "pi_bp": None,
            "pi_search": None,
            "q_search": None,
            "counts": None,
            "belief": None,
            "cat_q": None,
            "rollout_values": None,
        }


def compute_search_policy(
    q: torch.Tensor,
    bp_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    temperature: float,
    stepsize: float,
    uniform_magnet: bool = False,
) -> torch.Tensor:
    # Logits and values over the L legal actions
    logits_L = bp_logits[legal_action_mask]
    q_L = q[legal_action_mask]
    if uniform_magnet:
        search_logits_L = (logits_L + stepsize * q_L) / (1.0 + temperature * stepsize)
    else:
        # Compute log magnet policy over the L legal actions
        log_magnet_L = utils.helper.get_weighted_uniform_policy(legal_action_mask.unsqueeze(0))[0]
        log_magnet_L = log_magnet_L[legal_action_mask].log()
        # Compute search logits over the L legal actions
        temp_ss = temperature * stepsize
        search_logits_L = (logits_L + stepsize * q_L + temp_ss * log_magnet_L) / (1.0 + temp_ss)
    # Embed back into the full action space
    search_policy = torch.zeros_like(bp_logits)
    search_policy[legal_action_mask] = torch.softmax(search_logits_L, dim=-1)
    return search_policy
