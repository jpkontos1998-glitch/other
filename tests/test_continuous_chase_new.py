import os
import unittest
from random import uniform

import torch

from pyengine.core.env import Stratego
from pyengine import utils


pystratego = utils.get_pystratego()


cwd = os.path.dirname(__file__)
continuous_chase_games = sorted(os.listdir(os.path.join(cwd, "continuous_chase_games_new")))
violations = {"initial_boards": [], "action_sequences": [], "fn": []}

for game_file in continuous_chase_games[:10]:  # Only test first 10 game to speed up
    with open(os.path.join(cwd, "continuous_chase_games_new", game_file), "r") as f:
        game_data = f.readlines()
        init_board = game_data[0].strip()
        assert len(init_board) == 100
        violations["initial_boards"].append(init_board)
        violations["action_sequences"].append([int(game_data[i]) for i in range(1, len(game_data))])
        violations["fn"].append(game_file)


class ContinuousChaseNewTest(unittest.TestCase):
    def test_continuous_chase(self):
        num_envs = 1
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=1,
            continuous_chasing_rule=True,
            max_num_moves_between_attacks=200,
        )
        clean_env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=1,
            continuous_chasing_rule=False,
            max_num_moves_between_attacks=200,
        )
        action_tensor = torch.zeros(env.num_envs, device="cuda", dtype=torch.int32)
        for init_board, action_seq in list(
            zip(violations["initial_boards"], violations["action_sequences"])
        ):
            env.change_reset_behavior_to_initial_board(init_board)
            clean_env.change_reset_behavior_to_initial_board(init_board)
            env.reset()
            clean_env.reset()
            for a in action_seq[:-1]:
                action_tensor[:] = a
                self.assertTrue(env.current_legal_action_mask[0, a])
                env.apply_actions(action_tensor)
                clean_env.apply_actions(action_tensor)
            action_tensor[:] = action_seq[-1]
            self.assertTrue(clean_env.current_legal_action_mask[0, action_seq[-1]])
            self.assertFalse(env.current_legal_action_mask[0, action_seq[-1]])

    def test_change_reset_behavior(self):
        for two_square in [True, False]:
            num_envs = 2
            env = Stratego(
                num_envs=num_envs,
                traj_len_per_player=1,
                barrage=True,
                full_info=False,
                two_square_rule=two_square,
                max_num_moves_between_attacks=200,
            )
            counter = 0
            init_board = violations["initial_boards"][counter]
            env.change_reset_behavior_to_initial_board(init_board)
            env.reset()
            action_seq = violations["action_sequences"][counter]
            while counter < len(violations["initial_boards"]):
                self.assertTrue(torch.all(env.current_num_moves == 0))
                action_seq = violations["action_sequences"][counter]
                if counter < len(violations["initial_boards"]) - 1:
                    next_init_board = violations["initial_boards"][counter + 1]
                    env.change_reset_behavior_to_initial_board(next_init_board)
                action_tensor = torch.zeros(env.num_envs, device="cuda", dtype=torch.int32)
                for i, a in enumerate(action_seq[:-1]):
                    action_tensor[:] = a
                    self.assertTrue(env.current_legal_action_mask.cpu().numpy()[:, a].all())
                    env.apply_actions(action_tensor)
                self.assertFalse(
                    env.current_legal_action_mask.cpu().numpy()[:, action_seq[-1]].all()
                )
                while not env.current_has_just_reset.all():
                    env.sample_first_legal_action(action_tensor)
                    env.apply_actions(action_tensor)
                counter += 1

    def test_change_state(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=1,
        )
        other_env = Stratego(
            num_envs=2,
            traj_len_per_player=1,
        )
        for i in range(min(len(violations["initial_boards"]), 10)):
            init_board = violations["initial_boards"][i]
            action_seq = violations["action_sequences"][i]
            env.change_reset_behavior_to_initial_board(init_board)
            env.reset()
            action_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
            other_action_tensor = torch.zeros(2, device="cuda", dtype=torch.int32)
            for j, a in enumerate(action_seq[:-1]):
                action_tensor[:] = a
                env.apply_actions(action_tensor)
                env_state = env.current_state
                env_state.tile(2)
                other_env.change_reset_behavior_to_env_state(env_state)

                # Test randomly roughly once a game
                if uniform(0, 1) > 1 / len(action_seq):
                    continue
                for k, a_ in list(enumerate(action_seq))[j + 1 :]:
                    other_action_tensor[:] = a_
                    if k == len(action_seq) - 1:
                        self.assertFalse(
                            other_env.current_legal_action_mask.cpu().numpy()[:, a_].any()
                        )
                        break
                    other_env.apply_actions(other_action_tensor)

    def test_parallel(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=1,
        )
        state = None
        for init_board in violations["initial_boards"]:
            env.change_reset_behavior_to_initial_board(init_board)
            env.reset()
            if state is None:
                state = env.current_state
            else:
                state = state.cat(env.current_state)
        env = Stratego(
            num_envs=state.num_envs,
            traj_len_per_player=1,
        )
        env.change_reset_behavior_to_env_state(state)
        action_tensor = torch.zeros(state.num_envs, device="cuda", dtype=torch.int32)
        for i in range(env.conf.max_num_moves):
            env.sample_random_legal_action(action_tensor)
            if (env.current_num_moves_since_last_attack > 200).any():
                break
            for j, action_seq in enumerate(violations["action_sequences"]):
                if i > len(action_seq) - 1:
                    continue
                a = action_seq[i]
                if i < len(action_seq) - 1:
                    action_tensor[j] = a
                    self.assertTrue(env.current_legal_action_mask.cpu().numpy()[j, a])
                if i == len(action_seq) - 1:
                    self.assertFalse(env.current_legal_action_mask.cpu().numpy()[j, a])
            env.apply_actions(action_tensor)


if __name__ == "__main__":
    unittest.main()
