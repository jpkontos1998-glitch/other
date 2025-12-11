import unittest

import torch

from pyengine.core.env import Stratego
from pyengine import utils

pystratego = utils.get_pystratego()


class RewardBugTest(unittest.TestCase):
    def test_terminal_reward_chase_on(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            continuous_chasing_rule=True,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            (
                ["MCD" + 37 * "A"],
                ["MCD" + 37 * "A"],
            )
        )
        env.reset()

        def my_apply_action(abs_from, abs_to):
            action_tensor = torch.tensor(
                pystratego.util.abs_coordinates_to_actions(
                    [(abs_from, abs_to)], env.current_player
                ),
                device="cuda",
                dtype=torch.int32,
            )
            env.apply_actions(action_tensor)

        my_apply_action(2, 8)
        my_apply_action(97, 96)
        my_apply_action(8, 98)
        my_apply_action(96, 98)
        reward_before_dummy_action = env.current_reward_pl0.clone()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        env.sample_random_legal_action(action_tensor)
        env.apply_actions(action_tensor)
        reward_after_dummy_action = env.current_reward_pl0.clone()
        self.assertTrue(reward_before_dummy_action == reward_after_dummy_action)

    def test_terminal_reward_chase_off(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            continuous_chasing_rule=False,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            (
                ["MCD" + 37 * "A"],
                ["MCD" + 37 * "A"],
            )
        )
        env.reset()

        def my_apply_action(abs_from, abs_to):
            action_tensor = torch.tensor(
                pystratego.util.abs_coordinates_to_actions(
                    [(abs_from, abs_to)], env.current_player
                ),
                device="cuda",
                dtype=torch.int32,
            )
            env.apply_actions(action_tensor)

        my_apply_action(2, 8)
        my_apply_action(97, 96)
        my_apply_action(8, 98)
        my_apply_action(96, 98)
        reward_before_dummy_action = env.current_reward_pl0.clone()
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        env.sample_random_legal_action(action_tensor)
        env.apply_actions(action_tensor)
        reward_after_dummy_action = env.current_reward_pl0.clone()
        self.assertTrue(reward_before_dummy_action == reward_after_dummy_action)


if __name__ == "__main__":
    unittest.main()
