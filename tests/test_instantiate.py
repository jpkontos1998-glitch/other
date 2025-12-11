import unittest
import os
import random

import torch

from pyengine.belief.uniform import uniform_belief
from pyengine.core.env import Stratego
from pyengine import utils


pystratego = utils.get_pystratego()

with open(f"{os.path.dirname(__file__)}/instantiate_fail_game.txt", "r") as f:
    lines = f.readlines()


class InstantiateTest(unittest.TestCase):
    def compare_tensors(self, tensor1, tensor2):
        if tensor1.shape != tensor2.shape:
            raise AssertionError(
                f"Tensors have different shapes: {tensor1.shape} vs {tensor2.shape}"
            )
        if not torch.allclose(tensor1, tensor2):
            raise AssertionError(
                f"Tensors are not close. Argwhere: {torch.argwhere(tensor1 - tensor2)}"
            )

    def test_instantiate_fail(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
        )
        env.change_reset_behavior_to_initial_board(lines[0].strip())
        env.reset()
        other_num_envs = 256
        other_env = Stratego(num_envs=other_num_envs, traj_len_per_player=100)
        action_tensor = torch.zeros(env.num_envs, device="cuda", dtype=torch.int32)
        for i in range(1, len(lines)):
            action_tensor[:] = int(lines[i])
            env.apply_actions(action_tensor)
        legal_action_mask = env.current_legal_action_mask
        state = env.current_state
        samples = uniform_belief.generate(
                n_sample=other_num_envs,
                unknown_piece_position_onehot=env.current_unknown_piece_position_onehot.squeeze(0),
                unknown_piece_has_moved=env.current_unknown_piece_has_moved.squeeze(0),
                unknown_piece_counts=env.current_unknown_piece_counts.squeeze(0),
        )
        new_env_state = pystratego.util.assign_opponent_hidden_pieces(
            state, samples.to(torch.uint8)
        )
        new_env_state_legacy = pystratego.util.legacy_assign_opponent_hidden_pieces(
            state, samples.to(torch.uint8)
        )
        self.assertEqual(new_env_state.num_envs, new_env_state_legacy.num_envs)
        self.assertEqual(new_env_state.to_play, new_env_state_legacy.to_play)

        for attr in (
            "boards",
            "zero_boards",
            "num_moves",
            "num_moves_since_last_attack",
            "terminated_since",
            "has_legal_movement",
            "flag_captured",
            "action_history",
            "board_history",
            "move_summary_history",
        ):
            self.compare_tensors(getattr(new_env_state, attr), getattr(new_env_state_legacy, attr))

        for i in (0, 1):
            self.compare_tensors(
                new_env_state.chase_state.last_src_pos[i],
                new_env_state_legacy.chase_state.last_src_pos[i],
            )
            self.compare_tensors(
                new_env_state.chase_state.last_dst_pos[i],
                new_env_state_legacy.chase_state.last_dst_pos[i],
            )
            self.compare_tensors(
                new_env_state.chase_state.chase_length[i],
                new_env_state_legacy.chase_state.chase_length[i],
            )

        self.assertTrue(torch.allclose(new_env_state.boards, new_env_state_legacy.boards))
        other_env.change_reset_behavior_to_env_state(new_env_state)
        new_legal_action_mask = other_env.current_legal_action_mask
        self.assertTrue(torch.allclose(legal_action_mask, new_legal_action_mask))

    def test_random_instantiates(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
        )
        other_num_envs = 256
        other_env = Stratego(num_envs=other_num_envs, traj_len_per_player=100)
        for i in range(1, 10000):
            env.apply_actions(env.sample_random_legal_action())
            if random.random() > 0.01:
                continue
            infostate = env.current_infostate_tensor
            legal_action_mask = env.current_legal_action_mask
            num_moves = env.current_num_moves
            unknown_piece_position_onehot = env.current_unknown_piece_position_onehot
            unknown_piece_counts = env.current_unknown_piece_counts
            unknown_piece_has_moved = env.current_unknown_piece_has_moved
            state = env.current_state
            samples = uniform_belief.generate(
                n_sample=other_num_envs,
                unknown_piece_position_onehot=env.current_unknown_piece_position_onehot.squeeze(0),
                unknown_piece_has_moved=env.current_unknown_piece_has_moved.squeeze(0),
                unknown_piece_counts=env.current_unknown_piece_counts.squeeze(0),
            )
            new_env_state = pystratego.util.assign_opponent_hidden_pieces(
                state, samples.to(torch.uint8)
            )
            new_env_state_legacy = pystratego.util.legacy_assign_opponent_hidden_pieces(
                state, samples.to(torch.uint8)
            )
            self.assertEqual(new_env_state.num_envs, new_env_state_legacy.num_envs)
            self.assertEqual(new_env_state.to_play, new_env_state_legacy.to_play)

            for attr in (
                "boards",
                "zero_boards",
                "num_moves",
                "num_moves_since_last_attack",
                "terminated_since",
                "has_legal_movement",
                "flag_captured",
                "action_history",
                "board_history",
                "move_summary_history",
            ):
                self.compare_tensors(
                    getattr(new_env_state, attr), getattr(new_env_state_legacy, attr)
                )

            for i in (0, 1):
                self.compare_tensors(
                    new_env_state.chase_state.last_src_pos[i],
                    new_env_state_legacy.chase_state.last_src_pos[i],
                )
                self.compare_tensors(
                    new_env_state.chase_state.last_dst_pos[i],
                    new_env_state_legacy.chase_state.last_dst_pos[i],
                )
                self.compare_tensors(
                    new_env_state.chase_state.chase_length[i],
                    new_env_state_legacy.chase_state.chase_length[i],
                )

            self.assertTrue(torch.allclose(new_env_state.boards, new_env_state_legacy.boards))
            other_env.change_reset_behavior_to_env_state(new_env_state)
            new_legal_action_mask = other_env.current_legal_action_mask
            self.assertTrue(torch.allclose(legal_action_mask, new_legal_action_mask))
            new_infostate = other_env.current_infostate_tensor
            self.assertTrue(torch.allclose(infostate, new_infostate))
            new_num_moves = other_env.current_num_moves
            self.assertTrue(torch.allclose(num_moves, new_num_moves))
            new_unknown_piece_position_onehot = other_env.current_unknown_piece_position_onehot
            self.assertTrue(
                torch.allclose(unknown_piece_position_onehot, new_unknown_piece_position_onehot)
            )
            new_unknown_piece_counts = other_env.current_unknown_piece_counts
            self.assertTrue(torch.allclose(unknown_piece_counts, new_unknown_piece_counts))
            new_unknown_piece_has_moved = other_env.current_unknown_piece_has_moved
            self.assertTrue(torch.allclose(unknown_piece_has_moved, new_unknown_piece_has_moved))


if __name__ == "__main__":
    unittest.main()
