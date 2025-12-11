import unittest

import torch

from pyengine.core.buffer import CircularBuffer, modular_span
from pyengine.core.env import Stratego
from pyengine.networks.legacy_rl import TransformerRL, TransformerRLConfig
from pyengine.utils.constants import N_PLAYER, CATEGORICAL_AGGREGATION
import pyengine.utils as utils

from old_buffer import OldCircularBuffer

pystratego = utils.get_pystratego()

categorical_aggregation = CATEGORICAL_AGGREGATION.to("cuda")


def compute_1step_linear(rewards, terminals, values, advantages, returns, traj_len, gamma):
    returns_shape = returns.shape
    advantages_shape = advantages.shape

    rewards = rewards.view(traj_len, -1)
    terminals = terminals.view(traj_len, -1).bool()
    values = values.view(traj_len, -1)
    advantages = advantages.view(traj_len, -1)
    returns = returns.view(traj_len, -1)

    for step in range(traj_len - N_PLAYER, -1, -1):
        returns[step] = rewards[step] + gamma * (~terminals[step]) * values[step + 1]
        advantages[step] = returns[step] - values[step]

    return returns.view(*returns_shape), advantages.view(*advantages_shape)


def compute_mc_linear(rewards, terminals, values, advantages, returns, traj_len, gamma):
    returns_shape = returns.shape
    advantages_shape = advantages.shape

    rewards = rewards.view(traj_len, -1)
    terminals = terminals.view(traj_len, -1).bool()
    values = values.view(traj_len, -1)
    advantages = advantages.view(traj_len, -1)
    returns = returns.view(traj_len, -1)

    for step in range(traj_len - N_PLAYER, -1, -1):
        # Last time step of trajectory, there's no further information
        # so we're forced to bootstrap
        if step == traj_len - N_PLAYER:
            returns[step] = rewards[step] + gamma * (~terminals[step]) * values[step + 1]
        # If we're breaking transition boundary reset MC return; otherwise use
        # discounted MC return for next step.
        else:
            returns[step] = (
                terminals[step] * rewards[step] + (~terminals[step]) * gamma * returns[step + 1]
            )
        advantages[step] = returns[step] - values[step]

    return returns.view(*returns_shape), advantages.view(*advantages_shape)


def compute_1step_categorical(rewards, terminals, values, advantages, returns, traj_len):
    returns_shape = returns.shape
    advantages_shape = advantages.shape

    rewards = rewards.view(traj_len, -1)
    # Check all rewards are in {-1, 0, 1}
    assert torch.allclose(
        rewards**3 - rewards, torch.tensor(0, device="cuda", dtype=rewards.dtype)
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

    terminals = terminals.view(traj_len, -1).bool()
    values = torch.softmax(
        values.view(traj_len, -1, 3), dim=-1
    )  # The softmax is because the value head returns logits
    advantages = advantages.view(traj_len, -1)
    returns = returns.view(traj_len, -1, 3)

    for step in range(traj_len - N_PLAYER, -1, -1):
        # Intermediate rewards are written as draws, so we need to ignore them
        returns[step] = values[step + 1] + (rewards[step] - values[step + 1]) * terminals[
            step
        ].unsqueeze(1)
        advantages[step] = (returns[step] - values[step]) @ categorical_aggregation

    return returns.view(*returns_shape), advantages.view(*advantages_shape)


def compute_mc_categorical(rewards, terminals, values, advantages, returns, traj_len):
    returns_shape = returns.shape
    advantages_shape = advantages.shape

    rewards = rewards.view(traj_len, -1)
    # Check all rewards are in {-1, 0, 1}
    assert torch.allclose(
        rewards**3 - rewards, torch.tensor(0, device="cuda", dtype=rewards.dtype)
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

    terminals = terminals.view(traj_len, -1).bool()
    values = torch.softmax(
        values.view(traj_len, -1, 3), dim=-1
    )  # The softmax is because the value head returns logits
    advantages = advantages.view(traj_len, -1)
    returns = returns.view(traj_len, -1, 3)

    for step in range(traj_len - N_PLAYER, -1, -1):
        # Last time step of trajectory, there's no further information
        # so we're forced to bootstrap
        if step == traj_len - N_PLAYER:
            returns[step] = torch.where(
                terminals[step].unsqueeze(1), rewards[step], values[step + 1]
            )
        # Otherwise, we can re-use target from next time step
        else:
            returns[step] = torch.where(
                terminals[step].unsqueeze(1), rewards[step], returns[step + 1]
            )
        advantages[step] = (returns[step] - values[step]) @ categorical_aggregation

    return returns.view(*returns_shape), advantages.view(*advantages_shape)


class ReturnsAndAdvantagesTest(unittest.TestCase):
    def test_1step_gae_consistent_linear(self):
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = False
        gamma = 1.0
        move_memory = 86
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=traj_len_per_player,
            move_memory=move_memory,
            full_info=True,
        )
        buffer = CircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            train_every_per_player=traj_len_per_player,
            use_cat_vf=use_cat_vf,
            device="cuda",
        )
        old_buffer = OldCircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            move_memory=move_memory,
            use_cat_vf=use_cat_vf,
            device="cuda",
            dtype=torch.float,
        )
        policy = TransformerRL(
            torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
            TransformerRLConfig(depth=1, embed_dim_per_head_over8=1, use_cat_vf=use_cat_vf),
        ).to("cuda")

        for i in range(2):
            while not buffer.ready_to_train():
                buffer.add_pre_act(
                    step=env.current_step,
                    num_moves=env.current_num_moves,
                    legal_action_mask=env.current_legal_action_mask,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_pre_act(
                    obs_step=env.current_step,
                    obs=env.current_infostate_tensor,
                    piece_ids=env.current_piece_ids,
                    legal_action=env.current_legal_action_mask,
                )
                with torch.no_grad():
                    tensor_dict = policy(
                        env.current_infostate_tensor,
                        env.current_piece_ids,
                        env.current_legal_action_mask,
                        env.current_num_moves,
                    )
                    actions = tensor_dict["action"]
                    values = tensor_dict["value"]
                    log_probs = tensor_dict["action_log_probs"]
                env.apply_actions(actions)

                # We store the reward for the player that acted just before us.
                rewards = env.current_reward_pl0
                if env.current_player == 0:
                    rewards *= -1

                buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs,
                    reward=rewards,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(
                        -1
                    ),
                    reward=rewards,
                    terminal=env.current_is_terminal,
                )

            # Test 1-step is consistent with GAE(0)
            returns1, advantages1 = compute_1step_linear(
                rewards=old_buffer.rewards.clone(),
                terminals=old_buffer.terminals.clone(),
                values=old_buffer.values.clone(),
                advantages=old_buffer.advantages.clone(),
                returns=old_buffer.returns.clone(),
                traj_len=old_buffer.traj_len,
                gamma=gamma,
            )
            buffer.process_data(td_lambda=0, gae_lambda=0)
            curr_idx = (buffer.curr_idx) % buffer.num_row
            reindex = torch.tensor(
                modular_span(
                    curr_idx - N_PLAYER * buffer.traj_len,
                    curr_idx - N_PLAYER,
                    buffer.num_row,
                )
            )
            returns_gae = buffer.returns.clone()[reindex]
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(returns1[:-2], returns_gae))
            self.assertTrue(torch.allclose(advantages1[:-2], advantages_gae))
            # test to make sure returns and advantages work separately as intended
            buffer.process_data(td_lambda=0, gae_lambda=0)
            returns_gae = buffer.returns.clone()[reindex]
            self.assertTrue(torch.allclose(returns1[:-2], returns_gae))
            buffer.process_data(td_lambda=1, gae_lambda=0)
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(advantages1[:-2], advantages_gae))
            buffer.reset()
            old_buffer.reset()

    def test_mc_gae_consistent_linear(self):
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = False
        gamma = 1.0
        utils.set_seed_everywhere(1)
        move_memory = 86
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=traj_len_per_player,
            move_memory=move_memory,
            full_info=True,
        )
        buffer = CircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            train_every_per_player=traj_len_per_player,
            use_cat_vf=use_cat_vf,
            device="cuda",
            dtype=torch.float,
        )
        old_buffer = OldCircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            move_memory=move_memory,
            use_cat_vf=use_cat_vf,
            device="cuda",
            dtype=torch.float,
        )
        policy = TransformerRL(
            torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
            TransformerRLConfig(depth=1, embed_dim_per_head_over8=1, use_cat_vf=use_cat_vf),
        ).to("cuda")
        for i in range(3):
            while not buffer.ready_to_train():
                buffer.add_pre_act(
                    step=env.current_step,
                    num_moves=env.current_num_moves,
                    legal_action_mask=env.current_legal_action_mask,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_pre_act(
                    obs_step=env.current_step,
                    obs=env.current_infostate_tensor,
                    piece_ids=env.current_piece_ids,
                    legal_action=env.current_legal_action_mask,
                )

                with torch.no_grad():
                    tensor_dict = policy(
                        env.current_infostate_tensor,
                        env.current_piece_ids,
                        env.current_legal_action_mask,
                        env.current_num_moves,
                    )
                    actions = tensor_dict["action"]
                    values = tensor_dict["value"]
                    log_probs = tensor_dict["action_log_probs"]
                env.apply_actions(actions)

                # We store the reward for the player that acted just before us.
                rewards = env.current_reward_pl0
                if env.current_player == 0:
                    rewards *= -1

                buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs,
                    reward=rewards,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(
                        -1
                    ),
                    reward=rewards,
                    terminal=env.current_is_terminal,
                )

            # Test MC is consistent with GAE(1)
            returns_mc, advantages_mc = compute_mc_linear(
                rewards=old_buffer.rewards.clone(),
                terminals=old_buffer.terminals.clone(),
                values=old_buffer.values.clone(),
                advantages=old_buffer.advantages.clone(),
                returns=old_buffer.returns.clone(),
                traj_len=old_buffer.traj_len,
                gamma=gamma,
            )
            buffer.process_data(td_lambda=1, gae_lambda=1)
            curr_idx = (buffer.curr_idx) % buffer.num_row
            reindex = torch.tensor(
                modular_span(
                    curr_idx - N_PLAYER * buffer.traj_len,
                    curr_idx - N_PLAYER,
                    buffer.num_row,
                )
            )
            returns_gae = buffer.returns.clone()[reindex]
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(returns_mc[:-2], returns_gae, atol=1e-6))
            self.assertTrue(torch.allclose(advantages_mc[:-2], advantages_gae, atol=1e-6))
            # test to make sure returns and advantages work separately as intended
            buffer.process_data(td_lambda=1, gae_lambda=0)
            returns_gae = buffer.returns.clone()[reindex]
            self.assertTrue(torch.allclose(returns_mc[:-2], returns_gae, atol=1e-6))
            buffer.process_data(td_lambda=0, gae_lambda=1)
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(advantages_mc[:-2], advantages_gae, atol=1e-6))
            buffer.reset()
            old_buffer.reset()

    def test_1step_gae_consistent_categorical(self):
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = True
        move_memory = 86
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=traj_len_per_player,
            move_memory=move_memory,
            full_info=True,
        )
        buffer = CircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            train_every_per_player=traj_len_per_player,
            use_cat_vf=use_cat_vf,
            device="cuda",
        )
        old_buffer = OldCircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            move_memory=move_memory,
            use_cat_vf=use_cat_vf,
            device="cuda",
        )
        policy = TransformerRL(
            torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
            TransformerRLConfig(depth=1, embed_dim_per_head_over8=1),
        ).to("cuda")
        for i in range(3):
            while not buffer.ready_to_train():
                buffer.add_pre_act(
                    step=env.current_step,
                    num_moves=env.current_num_moves,
                    legal_action_mask=env.current_legal_action_mask,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_pre_act(
                    obs_step=env.current_step,
                    obs=env.current_infostate_tensor,
                    piece_ids=env.current_piece_ids,
                    legal_action=env.current_legal_action_mask,
                )
                with torch.no_grad():
                    tensor_dict = policy(
                        env.current_infostate_tensor,
                        env.current_piece_ids,
                        env.current_legal_action_mask,
                        env.current_num_moves,
                    )
                    actions = tensor_dict["action"]
                    values = tensor_dict["value"]
                    log_probs = tensor_dict["action_log_probs"]
                env.apply_actions(actions)

                # We store the reward for the player that acted just before us.
                rewards = env.current_reward_pl0
                if env.current_player == 0:
                    rewards *= -1

                buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs,
                    reward=rewards,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(
                        -1
                    ),
                    reward=rewards,
                    terminal=env.current_is_terminal,
                )
            # Test 1-step is consistent with GAE(0)
            returns1, advantages1 = compute_1step_categorical(
                rewards=old_buffer.rewards.clone(),
                terminals=old_buffer.terminals.clone(),
                values=old_buffer.values.clone(),
                advantages=old_buffer.advantages.clone(),
                returns=old_buffer.returns.clone(),
                traj_len=old_buffer.traj_len,
            )
            buffer.process_data(td_lambda=0, gae_lambda=0)
            curr_idx = (buffer.curr_idx) % buffer.num_row
            reindex = torch.tensor(
                modular_span(
                    curr_idx - N_PLAYER * buffer.traj_len,
                    curr_idx - N_PLAYER,
                    buffer.num_row,
                )
            )
            returns_gae = buffer.returns.clone()[reindex]
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(returns1[:-2], returns_gae))
            self.assertTrue(torch.allclose(advantages1[:-2], advantages_gae, atol=1e-6))
            # test to make sure returns and advantages work separately as intended
            buffer.process_data(td_lambda=0, gae_lambda=1)
            returns_gae = buffer.returns.clone()[reindex]
            self.assertTrue(torch.allclose(returns1[:-2], returns_gae))
            buffer.process_data(td_lambda=1, gae_lambda=0)
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(advantages1[:-2], advantages_gae, atol=1e-6))
            buffer.reset()
            old_buffer.reset()

    def test_mc_gae_consistent_categorical(self):
        num_envs = 1
        traj_len_per_player = 200
        use_cat_vf = True
        utils.set_seed_everywhere(1)
        move_memory = 86
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=traj_len_per_player,
            move_memory=move_memory,
            full_info=True,
        )
        buffer = CircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            train_every_per_player=traj_len_per_player,
            use_cat_vf=use_cat_vf,
            device="cuda",
            dtype=torch.float,
        )
        old_buffer = OldCircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            move_memory=move_memory,
            use_cat_vf=use_cat_vf,
            device="cuda",
            dtype=torch.float,
        )
        policy = TransformerRL(
            torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
            TransformerRLConfig(depth=1, embed_dim_per_head_over8=1, use_cat_vf=use_cat_vf),
        ).to("cuda")
        for i in range(3):
            while not buffer.ready_to_train():
                buffer.add_pre_act(
                    step=env.current_step,
                    num_moves=env.current_num_moves,
                    legal_action_mask=env.current_legal_action_mask,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_pre_act(
                    obs_step=env.current_step,
                    obs=env.current_infostate_tensor,
                    piece_ids=env.current_piece_ids,
                    legal_action=env.current_legal_action_mask,
                )

                with torch.no_grad():
                    tensor_dict = policy(
                        env.current_infostate_tensor,
                        env.current_piece_ids,
                        env.current_legal_action_mask,
                        env.current_num_moves,
                    )
                    actions = tensor_dict["action"]
                    values = tensor_dict["value"]
                    log_probs = tensor_dict["action_log_probs"]
                env.apply_actions(actions)

                # We store the reward for the player that acted just before us.
                rewards = env.current_reward_pl0
                if env.current_player == 0:
                    rewards *= -1

                buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs,
                    reward=rewards,
                    is_terminal=env.current_is_terminal,
                )
                old_buffer.add_post_act(
                    action=actions,
                    value=values,
                    log_prob=log_probs.gather(dim=-1, index=actions.long().unsqueeze(-1)).squeeze(
                        -1
                    ),
                    reward=rewards,
                    terminal=env.current_is_terminal,
                )

            # Test MC is consistent with GAE(1)
            returns_mc, advantages_mc = compute_mc_categorical(
                rewards=old_buffer.rewards.clone(),
                terminals=old_buffer.terminals.clone(),
                values=old_buffer.values.clone(),
                advantages=old_buffer.advantages.clone(),
                returns=old_buffer.returns.clone(),
                traj_len=old_buffer.traj_len,
            )
            buffer.process_data(td_lambda=1, gae_lambda=1)
            curr_idx = (buffer.curr_idx) % buffer.num_row
            reindex = torch.tensor(
                modular_span(
                    curr_idx - N_PLAYER * buffer.traj_len,
                    curr_idx - N_PLAYER,
                    buffer.num_row,
                )
            )
            returns_gae = buffer.returns.clone()[reindex]
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(returns_mc[:-2], returns_gae, atol=1e-6))
            self.assertTrue(torch.allclose(advantages_mc[:-2], advantages_gae, atol=1e-6))
            # test to make sure returns and advantages work separately as intended
            buffer.process_data(td_lambda=1, gae_lambda=0)
            returns_gae = buffer.returns.clone()[reindex]
            self.assertTrue(torch.allclose(returns_mc[:-2], returns_gae, atol=1e-6))
            buffer.process_data(td_lambda=0, gae_lambda=1)
            advantages_gae = buffer.advantages.clone()[reindex]
            self.assertTrue(torch.allclose(advantages_mc[:-2], advantages_gae, atol=1e-6))
            buffer.reset()
            old_buffer.reset()

    def test_categorical_average(self):
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = True
        utils.set_seed_everywhere(1)
        move_memory = 86
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=traj_len_per_player,
            move_memory=move_memory,
            full_info=True,
        )
        buffer = CircularBuffer(
            num_envs=num_envs,
            traj_len=traj_len_per_player,
            train_every_per_player=traj_len_per_player,
            use_cat_vf=use_cat_vf,
            device="cuda",
            dtype=torch.float,
        )
        policy = TransformerRL(
            torch.tensor(utils.init_helpers.COUNTERS["classic"] + [0, 0], device="cuda"),
            TransformerRLConfig(depth=1, embed_dim_per_head_over8=1, use_cat_vf=use_cat_vf),
        ).to("cuda")
        while not buffer.ready_to_train():
            buffer.add_pre_act(
                step=env.current_step,
                num_moves=env.current_num_moves,
                legal_action_mask=env.current_legal_action_mask,
                is_terminal=env.current_is_terminal,
            )

            with torch.no_grad():
                tensor_dict = policy(
                    env.current_infostate_tensor,
                    env.current_piece_ids,
                    env.current_legal_action_mask,
                    env.current_num_moves,
                )
                actions = tensor_dict["action"]
                values = tensor_dict["value"]
                log_probs = tensor_dict["action_log_probs"]
            env.apply_actions(actions)

            # We store the reward for the player that acted just before us.
            rewards = env.current_reward_pl0
            if env.current_player == 0:
                rewards *= -1

            buffer.add_post_act(
                action=actions,
                value=values,
                log_prob=log_probs,
                reward=rewards,
                is_terminal=env.current_is_terminal,
            )

        buffer.process_data(td_lambda=0.95, gae_lambda=0.95)
        returns_gae = buffer.returns.clone().sum(dim=-1)[:-N_PLAYER]
        self.assertTrue(
            torch.allclose(returns_gae, torch.tensor(1, dtype=torch.float, device="cuda"))
        )
        # test to make sure returns and advantages work separately as intended
        buffer.process_data(td_lambda=0.95, gae_lambda=0)
        returns_gae = buffer.returns.clone().sum(dim=-1)[:-N_PLAYER]
        self.assertTrue(
            torch.allclose(returns_gae, torch.tensor(1, dtype=torch.float, device="cuda"))
        )


if __name__ == "__main__":
    unittest.main()
