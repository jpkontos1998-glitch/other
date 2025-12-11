import unittest

import torch

from pyengine.arrangement.buffer import ArrangementBuffer
from pyengine.networks.arrangement_transformer import (
    ArrangementTransformer,
    ArrangementTransformerConfig,
)
from pyengine.core.env import Stratego
from pyengine.arrangement.sampling import generate_arrangements
from pyengine.utils.init_helpers import COUNTERS
from pyengine.utils.constants import CATEGORICAL_AGGREGATION, N_PLAYER
from pyengine import utils

categorical_aggregation = CATEGORICAL_AGGREGATION.to("cuda")

pystratego = utils.get_pystratego()


def compute_1step_linear(rewards, values, advantages, returns, traj_len, gamma):
    for step in reversed(range(traj_len)):
        if step == traj_len - 1:
            returns[:, step] = rewards
        else:
            returns[:, step] = gamma * values[:, step + 1]
        advantages[:, step] = returns[:, step] - values[:, step]
    return returns, advantages


def compute_mc_linear(rewards, values, advantages, returns, traj_len, gamma):
    for step in reversed(range(traj_len)):
        if step == traj_len - 1:
            returns[:, step] = rewards
        else:
            returns[:, step] = gamma * returns[:, step + 1]
    advantages = returns - values
    return returns, advantages


def compute_1step_categorical(rewards, values, advantages, returns, traj_len):
    for step in reversed(range(traj_len)):
        if step == traj_len - 1:
            returns[:, step] = rewards
        else:
            returns[:, step] = values[:, step + 1]
        advantages[:, step] = (returns[:, step] - values[:, step]) @ categorical_aggregation
    return returns, advantages


def compute_mc_categorical(rewards, values, advantages, returns, traj_len):
    for step in reversed(range(traj_len)):
        returns[:, step] = rewards
        advantages[:, step] = (rewards - values[:, step]) @ categorical_aggregation
    return returns, advantages


class ProcessArrDataTest(unittest.TestCase):
    def test_1step_gae_consistent_linear(self):
        num_envs = 1000
        traj_len_per_player = 100
        use_cat_vf = False
        gamma = 1.0
        env = Stratego(num_envs=num_envs, traj_len_per_player=traj_len_per_player, full_info=True)
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device="cuda")
        buffer = ArrangementBuffer(
            storage_duration=N_PLAYER * traj_len_per_player + env.conf.max_num_moves,
            barrage=True,
            use_cat_vf=use_cat_vf,
            device=torch.device("cuda"),
        )
        init_policy = ArrangementTransformer(
            piece_counts, ArrangementTransformerConfig(use_cat_vf=use_cat_vf)
        ).to("cuda")
        init_arrs, values, reg_values, log_probs, flipped_mask, _ = generate_arrangements(
            100, init_policy
        )
        buffer.add_arrangements(
            init_arrs,
            values,
            reg_values,
            log_probs,
            flipped_mask,
            0,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                pystratego.util.arrangement_strings_from_tensor(arr)
                for arr in 2 * [init_arrs.argmax(dim=-1).type(torch.uint8)]
            ]
        )
        env.reset()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for t in range(100):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            red_arr, blue_arr = env.current_zero_arrangements
            buffer.add_rewards(
                arrangements=red_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=env.current_reward_pl0,
            )
            buffer.add_rewards(
                arrangements=blue_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=-env.current_reward_pl0,
            )

        # Test 1-step is consistent with GAE(0)
        returns1, advantages1 = compute_1step_linear(
            rewards=buffer.rewards.clone()[buffer.ready_flags],
            values=buffer.values.clone()[buffer.ready_flags],
            advantages=buffer.adv_est.clone()[buffer.ready_flags],
            returns=buffer.val_est.clone()[buffer.ready_flags],
            traj_len=40,
            gamma=gamma,
        )
        buffer.process_data(td_lambda=0.0, gae_lambda=0.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns1, returns_gae, atol=1e-6))
        self.assertTrue(torch.allclose(advantages1, advantages_gae, atol=1e-6))
        # test that returns and advantages work separately as intended
        buffer.process_data(td_lambda=0.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns1, returns_gae, atol=1e-6))
        buffer.process_data(td_lambda=1.0, gae_lambda=0.0, reg_temp=0.0, reg_norm=1.0)
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(advantages1, advantages_gae, atol=1e-6))

    def test_mc_gae_consistent_linear(self):
        utils.set_seed_everywhere(1)
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = False
        gamma = 1.0
        env = Stratego(num_envs=num_envs, traj_len_per_player=traj_len_per_player, full_info=True)
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device="cuda")
        buffer = ArrangementBuffer(
            storage_duration=N_PLAYER * traj_len_per_player + env.conf.max_num_moves,
            barrage=True,
            use_cat_vf=use_cat_vf,
            device=torch.device("cuda"),
        )
        init_policy = ArrangementTransformer(
            piece_counts, ArrangementTransformerConfig(use_cat_vf=use_cat_vf)
        ).to("cuda")
        init_arrs, values, reg_values, log_probs, flipped_mask, _ = generate_arrangements(
            100, init_policy
        )
        buffer.add_arrangements(
            init_arrs,
            values,
            reg_values,
            log_probs,
            flipped_mask,
            0,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                pystratego.util.arrangement_strings_from_tensor(arr)
                for arr in 2 * [init_arrs.argmax(dim=-1).type(torch.uint8)]
            ]
        )
        env.reset()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for t in range(100):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            red_arr, blue_arr = env.current_zero_arrangements
            buffer.add_rewards(
                arrangements=red_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=env.current_reward_pl0,
            )
            buffer.add_rewards(
                arrangements=blue_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=-env.current_reward_pl0,
            )

        # Test mc is consistent with GAE(1)
        returns_mc, advantages_mc = compute_mc_linear(
            rewards=buffer.rewards.clone()[buffer.ready_flags],
            values=buffer.values.clone()[buffer.ready_flags],
            advantages=buffer.adv_est.clone()[buffer.ready_flags],
            returns=buffer.val_est.clone()[buffer.ready_flags],
            traj_len=40,
            gamma=gamma,
        )
        buffer.process_data(td_lambda=1.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns_mc, returns_gae, atol=1e-6))
        self.assertTrue(torch.allclose(advantages_mc, advantages_gae, atol=1e-6))
        # test that returns and advantages work separately as intended
        buffer.process_data(td_lambda=1.0, gae_lambda=0.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns_mc, returns_gae, atol=1e-6))
        buffer.process_data(td_lambda=0.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=1.0)
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(advantages_mc, advantages_gae, atol=1e-6))

    def test_1step_gae_consistent_categorical(self):
        utils.set_seed_everywhere(1)
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = True
        env = Stratego(num_envs=num_envs, traj_len_per_player=traj_len_per_player, full_info=True)
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device="cuda")
        buffer = ArrangementBuffer(
            storage_duration=N_PLAYER * traj_len_per_player + env.conf.max_num_moves,
            barrage=True,
            use_cat_vf=use_cat_vf,
            device=torch.device("cuda"),
        )
        init_policy = ArrangementTransformer(piece_counts, ArrangementTransformerConfig()).to(
            "cuda"
        )
        init_arrs, values, reg_values, log_probs, flipped_mask, _ = generate_arrangements(
            100, init_policy
        )
        buffer.add_arrangements(
            init_arrs,
            values,
            reg_values,
            log_probs,
            flipped_mask,
            0,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                pystratego.util.arrangement_strings_from_tensor(arr)
                for arr in 2 * [init_arrs.argmax(dim=-1).type(torch.uint8)]
            ]
        )
        env.reset()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for t in range(100):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            red_arr, blue_arr = env.current_zero_arrangements
            buffer.add_rewards(
                arrangements=red_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=env.current_reward_pl0,
            )
            buffer.add_rewards(
                arrangements=blue_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=-env.current_reward_pl0,
            )

        # Test 1-step is consistent with GAE(0)
        returns1, advantages1 = compute_1step_categorical(
            rewards=buffer.rewards.clone()[buffer.ready_flags],
            values=buffer.values.clone()[buffer.ready_flags],
            advantages=buffer.adv_est.clone()[buffer.ready_flags],
            returns=buffer.val_est.clone()[buffer.ready_flags],
            traj_len=40,
        )
        buffer.process_data(td_lambda=0.0, gae_lambda=0.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns1, returns_gae, atol=1e-6))
        self.assertTrue(torch.allclose(advantages1, advantages_gae, atol=1e-6))
        # test that returns and advantages work separately as intended
        buffer.process_data(td_lambda=0.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns1, returns_gae, atol=1e-6))
        buffer.process_data(td_lambda=1.0, gae_lambda=0.0, reg_temp=0.0, reg_norm=1.0)
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(advantages1, advantages_gae, atol=1e-6))

    def test_mc_gae_consistent_categorical(self):
        utils.set_seed_everywhere(1)
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = True
        env = Stratego(num_envs=num_envs, traj_len_per_player=traj_len_per_player, full_info=True)
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device="cuda")
        buffer = ArrangementBuffer(
            storage_duration=N_PLAYER * traj_len_per_player + env.conf.max_num_moves,
            barrage=True,
            use_cat_vf=use_cat_vf,
            device=torch.device("cuda"),
        )
        init_policy = ArrangementTransformer(piece_counts, ArrangementTransformerConfig()).to(
            "cuda"
        )
        init_arrs, values, reg_values, log_probs, flipped_mask, _ = generate_arrangements(
            100, init_policy
        )
        buffer.add_arrangements(
            init_arrs,
            values,
            reg_values,
            log_probs,
            flipped_mask,
            0,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                pystratego.util.arrangement_strings_from_tensor(arr)
                for arr in 2 * [init_arrs.argmax(dim=-1).type(torch.uint8)]
            ]
        )
        env.reset()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for t in range(100):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            red_arr, blue_arr = env.current_zero_arrangements
            buffer.add_rewards(
                arrangements=red_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=env.current_reward_pl0,
            )
            buffer.add_rewards(
                arrangements=blue_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=-env.current_reward_pl0,
            )

        # Test mc is consistent with GAE(1)
        returns_mc, advantages_mc = compute_mc_categorical(
            rewards=buffer.rewards.clone()[buffer.ready_flags],
            values=buffer.values.clone()[buffer.ready_flags],
            advantages=buffer.adv_est.clone()[buffer.ready_flags],
            returns=buffer.val_est.clone()[buffer.ready_flags],
            traj_len=40,
        )
        buffer.process_data(td_lambda=1.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns_mc, returns_gae, atol=1e-6))
        self.assertTrue(torch.allclose(advantages_mc, advantages_gae, atol=1e-6))
        # test that returns and advantages work separately as intended
        buffer.process_data(td_lambda=1.0, gae_lambda=0.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(returns_mc, returns_gae, atol=1e-6))
        buffer.process_data(td_lambda=0.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=1.0)
        advantages_gae = buffer.adv_est.clone()[buffer.ready_flags]
        self.assertTrue(torch.allclose(advantages_mc, advantages_gae, atol=1e-6))

    def test_categorical_average(self):
        utils.set_seed_everywhere(1)
        num_envs = 50
        traj_len_per_player = 100
        use_cat_vf = True
        env = Stratego(num_envs=num_envs, traj_len_per_player=traj_len_per_player, full_info=True)
        piece_counts = torch.tensor(COUNTERS["barrage"] + [0, 32], device="cuda")
        buffer = ArrangementBuffer(
            storage_duration=N_PLAYER * traj_len_per_player + env.conf.max_num_moves,
            barrage=True,
            use_cat_vf=use_cat_vf,
            device=torch.device("cuda"),
        )
        init_policy = ArrangementTransformer(piece_counts, ArrangementTransformerConfig()).to(
            "cuda"
        )
        init_arrs, values, reg_values, log_probs, flipped_mask, _ = generate_arrangements(
            100, init_policy
        )
        buffer.add_arrangements(
            init_arrs,
            values,
            reg_values,
            log_probs,
            flipped_mask,
            0,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                pystratego.util.arrangement_strings_from_tensor(arr)
                for arr in 2 * [init_arrs.argmax(dim=-1).type(torch.uint8)]
            ]
        )
        env.reset()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for t in range(100):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            red_arr, blue_arr = env.current_zero_arrangements
            buffer.add_rewards(
                arrangements=red_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=env.current_reward_pl0,
            )
            buffer.add_rewards(
                arrangements=blue_arr,
                is_newly_terminal=env.current_is_newly_terminal,
                rewards=-env.current_reward_pl0,
            )

        # Test 1-step is consistent with GAE(0)
        buffer.process_data(td_lambda=0.0, gae_lambda=0.0, reg_temp=0.0, reg_norm=1.0)
        returns_gae = buffer.val_est.clone()[buffer.ready_flags]
        sum_probs = returns_gae.sum(dim=-1)
        self.assertTrue(
            torch.allclose(sum_probs, torch.ones_like(sum_probs, device="cuda"), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
