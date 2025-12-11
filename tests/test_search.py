import unittest

import torch

from pyengine.belief.uniform import uniform_belief
from pyengine.core.buffer import CircularBuffer
from pyengine.core.env import Stratego
from pyengine.core.search import SearchBot
from pyengine.networks.legacy_rl import TransformerRL, TransformerRLConfig
from pyengine.utils.constants import CATEGORICAL_AGGREGATION
import pyengine.utils as utils

class SearchTest(unittest.TestCase):
    def test_q_value_computation(self):
        env = Stratego(num_envs=1, traj_len_per_player=100)
        search_env = Stratego(num_envs=100, traj_len_per_player=100, quiet=1)
        for use_cat_vf in [True, False]:
            policy = TransformerRL(
                torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
                TransformerRLConfig(depth=1, embed_dim_per_head_over8=1, use_cat_vf=use_cat_vf),
            ).to("cuda")
            for depth in [2, 8]:
                for td_lambda in [0.0, 0.5, 1.0]:
                    search_bot = SearchBot(
                        policy,
                        search_env,
                        depth=depth,
                        stepsize=10,
                        temperature=1e-3,
                        td_lambda=td_lambda,
                        max_num_samples=100,
                        dtype=torch.float32,
                        belief_model=uniform_belief,
                    )
                    action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
                    for i in range(1000):
                        env.sample_random_legal_action(action_tensor)
                        if torch.rand(1) > 0.01 and not env.current_is_terminal:
                            continue
                        state = env.current_state
                        search_bot(
                            state,
                            env.current_infostate_tensor,
                            env.current_piece_ids,
                            env.current_legal_action_mask,
                            env.current_num_moves,
                            env.current_unknown_piece_position_onehot,
                            env.current_unknown_piece_counts,
                            env.current_unknown_piece_has_moved,
                        )
                        buffer = CircularBuffer(
                            num_envs=search_env.num_envs,
                            traj_len=search_bot.depth // 2 + 1,
                            train_every_per_player=search_bot.depth // 2 + 1,
                            use_cat_vf=use_cat_vf,
                            device="cuda",
                        )
                        for step in range(
                            search_env.current_step - search_bot.depth, search_env.current_step
                        ):
                            if step == search_env.current_step - search_bot.depth:
                                assert (
                                    search_env.infostate_tensor(step)
                                    == env.current_infostate_tensor
                                ).all()
                            buffer.add_pre_act(
                                step=step,
                                num_moves=search_env.num_moves(step),
                                legal_action_mask=search_env.legal_action_mask(step),
                                is_terminal=search_env.is_terminal(step),
                            )
                            with torch.no_grad(), utils.eval_mode(policy):
                                tensor_dict = policy(
                                    search_env.infostate_tensor(step),
                                    search_env.piece_ids(step),
                                    search_env.legal_action_mask(step),
                                    search_env.num_moves(step),
                                )
                                values = tensor_dict["value"]
                                log_probs = tensor_dict["action_log_probs"]
                            rewards = search_env.reward_pl0(step + 1)
                            if search_env.acting_player(step + 1) == 0:
                                rewards *= -1
                            buffer.add_post_act(
                                action=search_env.played_actions(step),
                                value=values,
                                log_prob=log_probs,
                                reward=rewards,
                                is_terminal=search_env.is_terminal(step + 1),
                            )
                        # Special case last step
                        buffer.add_pre_act(
                            step=search_env.current_step,
                            num_moves=search_env.current_num_moves,
                            legal_action_mask=search_env.current_legal_action_mask,
                            is_terminal=search_env.current_is_terminal,
                        )
                        with torch.no_grad(), utils.eval_mode(policy):
                            tensor_dict = policy(
                                search_env.current_infostate_tensor,
                                search_env.current_piece_ids,
                                search_env.current_legal_action_mask,
                                search_env.current_num_moves,
                            )
                            values = tensor_dict["value"]
                            log_probs = tensor_dict["action_log_probs"]

                        buffer.add_post_act(
                            action=search_env.played_actions(step),
                            value=values,
                            log_prob=log_probs,
                            reward=-rewards,  # Dummy value
                            is_terminal=search_env.current_is_terminal,  # Dummy value
                        )
                        # last step for opponent (all dummy values)
                        buffer.add_pre_act(
                            step=search_env.current_step,
                            num_moves=search_env.current_num_moves,
                            legal_action_mask=search_env.current_legal_action_mask,
                            is_terminal=search_env.current_is_terminal,
                        )
                        buffer.add_post_act(
                            action=search_env.played_actions(step),
                            value=values,
                            log_prob=log_probs,
                            reward=rewards,  # Dummy value
                            is_terminal=search_env.current_is_terminal,  # Dummy value
                        )
                        assert buffer.ready_to_train()
                        buffer.process_data(
                            td_lambda=search_bot.td_lambda,
                            gae_lambda=1.0,
                        )
                        if use_cat_vf:
                            returns = (buffer.returns @ CATEGORICAL_AGGREGATION.cuda()).cpu()[0]
                        else:
                            returns = buffer.returns.cpu()[0]
                        self.assertTrue(
                            torch.allclose(
                                returns, search_bot.last_search_info["rollout_values"], atol=1e-6
                            )
                        )

    def test_q_cat_norm(self):
        env = Stratego(num_envs=1, traj_len_per_player=100)
        search_env = Stratego(num_envs=100, traj_len_per_player=100, quiet=1)
        policy = TransformerRL(
            torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
            TransformerRLConfig(depth=1, embed_dim_per_head_over8=1),
        ).to("cuda")
        for depth in [2, 8]:
            for td_lambda in [0.0, 0.5, 1.0]:
                search_bot = SearchBot(
                    policy,
                    search_env,
                    depth=depth,
                    stepsize=10,
                    temperature=1e-3,
                    max_num_samples=100,
                    td_lambda=td_lambda,
                    dtype=torch.bfloat16,
                    belief_model=uniform_belief,
                )
                action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
                for i in range(1000):
                    env.sample_random_legal_action(action_tensor)
                    if torch.rand(1) > 0.01 and not env.current_is_terminal:
                        continue
                    state = env.current_state
                    search_bot(
                        state,
                        env.current_infostate_tensor,
                        env.current_piece_ids,
                        env.current_legal_action_mask,
                        env.current_num_moves,
                        env.current_unknown_piece_position_onehot,
                        env.current_unknown_piece_counts,
                        env.current_unknown_piece_has_moved,
                    )
                    last_cat_q = search_bot.last_search_info["cat_q"]
                    counts = search_bot.last_search_info["counts"]
                    self.assertTrue(
                        torch.allclose(
                            torch.sum(last_cat_q[counts > 0], dim=-1),
                            torch.tensor([1.0], device="cuda", dtype=last_cat_q.dtype),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
