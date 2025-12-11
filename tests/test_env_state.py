import os
import glob
import unittest
import torch
from pyengine.core.env import Stratego
from importlib.machinery import ExtensionFileLoader

from pyengine.utils.constants import N_PLAYER

root = os.path.dirname(os.path.abspath(__file__))
pystratego_path = glob.glob(f"{root}/../build/pystratego*.so")[0]
pystratego = ExtensionFileLoader("pystratego", pystratego_path).load_module()


class EnvStateTest(unittest.TestCase):
    def test_run(self):
        env = Stratego(num_envs=1, traj_len_per_player=50, move_memory=20)
        env.env.seed_action_sampler(44)
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        buf_size = env.env.buf_size

        for _ in range(buf_size * 40 + 10):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        T = env.current_step
        for t in range(T - buf_size + 1 + env.conf.move_memory, T + 1):
            state = env.snapshot_state(t)
            self.assertEqual(env.board_strs(t), state.board_strs())

        env.env.change_reset_behavior(pystratego.ResetBehavior.RANDOM_JB_BARRAGE_BOARD)
        env.reset()

        for _ in range(buf_size * 40 + 10):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        T = env.current_step
        for t in range(T - buf_size + 1 + env.conf.move_memory, T + 1):
            state = env.snapshot_state(t)
            self.assertEqual(env.board_strs(t), state.board_strs())

        while True:
            if env.env.get_terminated_since(env.current_step).max() == 0:
                break
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        env_state = env.snapshot_state(T)
        env_state_infostate = env.infostate_tensor(T).clone()
        env.change_reset_behavior_to_env_state(env_state)
        env.reset()

        self.assertEqual(
            env.zero_board_strs(env.current_step)[0],
            env_state.zero_board_strs()[0],
        )

        for t in range(buf_size * 40 + 11):
            for env_idx in range(env.num_envs):
                self.assertEqual(
                    env.zero_board_strs(env.current_step)[env_idx],
                    env_state.zero_board_strs()[env_idx],
                )
                if env.num_moves(env.current_step)[env_idx] == env_state.num_moves[env_idx]:
                    self.assertEqual(env.num_moves_since_reset(env.current_step)[env_idx], 0)
                    self.assertEqual(
                        env.board_strs(env.current_step)[env_idx],
                        env_state.board_strs()[env_idx],
                    )
                    self.assertTrue(
                        torch.allclose(
                            env.infostate_tensor(t)[env_idx, ...],
                            env_state_infostate[env_idx, ...],
                        )
                    )
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        T = env.current_step
        for t in range(T - buf_size + 1 + env.conf.move_memory, T + 1):
            state = env.snapshot_state(t)
            self.assertEqual(env.board_strs(t), state.board_strs())

        while True:
            if env.env.get_terminated_since(env.current_step).max() == 0:
                break
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        env_state = env.snapshot_state(T)
        env_state_infostate = env.infostate_tensor(T).clone()
        env.change_reset_behavior_to_env_state(env_state)
        env.reset()

        for t in range(buf_size * 40 + 11):
            for env_idx in range(env.num_envs):
                if env.num_moves(env.current_step)[env_idx] == env_state.num_moves[env_idx]:
                    self.assertEqual(env.num_moves_since_reset(env.current_step)[env_idx], 0)
                    self.assertEqual(
                        env.board_strs(env.current_step)[env_idx],
                        env_state.board_strs()[env_idx],
                    )
                    self.assertTrue(
                        torch.allclose(
                            env.current_infostate_tensor[env_idx, ...],
                            env_state_infostate[env_idx, ...],
                        )
                    )
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        T = env.current_step
        for t in range(T - buf_size + 1 + env.conf.move_memory, T + 1):
            state = env.snapshot_state(t)
            self.assertEqual(env.board_strs(t), state.board_strs())

    def test_overlap(self):
        """This test checks that the snapshots at consecutive states have the correct overlap"""
        env = Stratego(
            num_envs=256,
            traj_len_per_player=32,
            move_memory=30,
        )

        for t in range(1024):
            self.assertEqual(env.current_step, t)
            actions = env.sample_random_legal_action()
            env.apply_actions(actions)
            new = env.snapshot_state(t + 1)
            old = env.snapshot_state(t)

            steps_since_reset = env.current_num_moves_since_reset.cpu()
            for i in range(env.num_envs):
                if steps_since_reset[i] == 0:
                    old.action_history[:, i] = 0
                    old.board_history[:, i, :] = 0

            self.assertTrue(torch.allclose(new.action_history[:-1], old.action_history[1:]))
            self.assertTrue(torch.allclose(new.board_history[:-1], old.board_history[1:]))

    def test_reset_to_pl1_state(self):
        """Test resetting to a state of player 1"""
        # Initialize two envs
        env1 = Stratego(num_envs=1, traj_len_per_player=50, move_memory=20)
        env1.change_reset_behavior_to_random_initial_arrangement(
            [
                [pystratego.JB_INIT_BOARDS_BARRAGE[42]],
                [pystratego.JB_INIT_BOARDS_BARRAGE[43]],
            ],
        )
        env1.reset()
        env2 = Stratego(num_envs=1, traj_len_per_player=54, move_memory=20)
        env2.change_reset_behavior_to_random_initial_arrangement(
            [
                [pystratego.JB_INIT_BOARDS_BARRAGE[42]],
                [pystratego.JB_INIT_BOARDS_BARRAGE[43]],
            ],
        )
        env2.reset()

        action_tensor = torch.zeros(env1.num_envs, dtype=torch.int32, device="cuda")

        # Roll them out for a few steps using random actions to a player 1 state
        for _ in range(env1.env.buf_size * 40 + 11):
            env1.sample_random_legal_action(action_tensor)
            env1.apply_actions(action_tensor)
            env2.apply_actions(action_tensor)
        assert env2.current_player == 1

        while True:
            if (
                env2.env.get_terminated_since(env2.current_step).max() == 0
                and env2.current_player == 1
            ):
                break
            env1.sample_random_legal_action(action_tensor)
            env1.apply_actions(action_tensor)
            env2.apply_actions(action_tensor)

        # Roll out a bit and then reset
        T = env2.current_step
        env_state = env2.snapshot_state(T)
        env2.change_reset_behavior_to_env_state(env_state)
        env2.reset()
        for _ in range(10):
            env2.sample_random_legal_action(action_tensor)
            env2.apply_actions(action_tensor)
        env2.reset()
        self.assertEqual(
            env1.board_strs(env1.current_step),
            env2.board_strs(env2.current_step),
        )

        # Check that things are equal when we rollout
        while env1.env.get_terminated_since(env1.current_step).item() <= 1:
            env1.sample_random_legal_action(action_tensor)
            env1.apply_actions(action_tensor)
            env2.apply_actions(action_tensor)
            self.assertEqual(
                env1.board_strs(env1.current_step),
                env2.board_strs(env2.current_step),
            )
            self.assertTrue((env1.current_infostate_tensor == env2.current_infostate_tensor).all())
            self.assertEqual(env1.current_reward_pl0, env2.current_reward_pl0)
            self.assertEqual(
                env1.env.get_terminated_since(env1.current_step),
                env2.env.get_terminated_since(env2.current_step),
            )

    def test_replicate_env(self):
        """Test replicate index"""
        env = Stratego(num_envs=100, traj_len_per_player=100, move_memory=20)
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        infostate_tensors = []
        env_states = []
        for _ in range(100):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            env_state = env.snapshot_state(env.current_step)
            board_strs = env_state.board_strs()
            env_state.replicate_env(0)
            for env_idx in range(env.num_envs):
                self.assertEqual(board_strs[0], env_state.board_strs()[env_idx])
            env_states.append(env.snapshot_state(env.current_step))
            infostate_tensors.append(env.current_infostate_tensor.clone())
        for infostate, env_state in zip(infostate_tensors, env_states):
            if env_state.terminated_since.max() > 0:
                continue
            env.change_reset_behavior_to_env_state(env_state)
            env.reset()
            self.assertTrue(torch.allclose(infostate, env.current_infostate_tensor))

    def test_robustness_to_mutation(self):
        """Example of non-robustness to mutation"""
        env = Stratego(traj_len_per_player=50, num_envs=5, move_memory=20)
        env_state = env.snapshot_state(env.current_step)
        board_strs = env_state.board_strs()
        env.change_reset_behavior_to_env_state(env_state)
        env.reset()

        self.assertEqual(board_strs, env_state.board_strs())
        # Now mutate the state that was passed to `change_reset_behavior`
        env_state.replicate_env(0)
        self.assertNotEqual(board_strs, env_state.board_strs())

        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for _ in range(100):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
        env.reset()

        self.assertNotEqual(env_state.board_strs(), env.board_strs(env.current_step))
        self.assertEqual(board_strs, env.board_strs(env.current_step))

    def test_tiling(self):
        """Test tiling"""
        num_envs = 50
        env1 = Stratego(num_envs=num_envs, traj_len_per_player=30, move_memory=20)
        env2 = Stratego(
            num_envs=2 * num_envs,
            traj_len_per_player=30,
            move_memory=20,
        )
        action_tensor = torch.zeros(env1.num_envs, dtype=torch.int32, device="cuda")
        for _ in range(100):
            env1.sample_random_legal_action(action_tensor)
            env1.apply_actions(action_tensor)

        while True:
            if env1.env.get_terminated_since(env1.current_step).max() == 0:
                break
            env1.sample_random_legal_action(action_tensor)
            env1.apply_actions(action_tensor)

        env1.env.get_terminated_since(env1.current_step)
        env_state = env1.snapshot_state(env1.current_step)
        env_state.tile(2)
        env2.change_reset_behavior_to_env_state(env_state)
        env2.reset()
        for env_idx in range(num_envs):
            self.assertEqual(
                env1.board_strs(env1.current_step)[env_idx],
                env2.board_strs(env2.current_step)[env_idx],
            )
            self.assertEqual(
                env1.board_strs(env1.current_step)[env_idx],
                env2.board_strs(env2.current_step)[num_envs + env_idx],
            )
            self.assertTrue(
                torch.allclose(
                    env1.current_infostate_tensor[env_idx, ...],
                    env2.current_infostate_tensor[env_idx, ...],
                )
            )
            self.assertTrue(
                torch.allclose(
                    env1.current_infostate_tensor[env_idx, ...],
                    env2.current_infostate_tensor[num_envs + env_idx, ...],
                )
            )
            self.assertEqual(
                env1.current_reward_pl0[env_idx],
                env2.current_reward_pl0[env_idx],
            )
            self.assertEqual(
                env1.current_reward_pl0[env_idx],
                env2.current_reward_pl0[num_envs + env_idx],
            )

    def test_overlap_deterministic(self):
        """This test checks that the snapshots at consecutive states have the correct overlap"""
        env = Stratego(
            num_envs=8,
            traj_len_per_player=32,
            move_memory=30,
            custom_inits=[
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ],
        )
        for t in range(256):
            self.assertEqual(env.current_step, t)
            actions = env.sample_first_legal_action()
            env.apply_actions(actions)
            new = env.snapshot_state(t + 1)
            old = env.snapshot_state(t)

            steps_since_reset = env.current_num_moves_since_reset.cpu()
            for i in range(env.num_envs):
                if steps_since_reset[i] == 0:
                    old.action_history[:, i] = 0
                    old.board_history[:, i, :] = 0

            self.assertTrue(torch.allclose(new.action_history[:-1], old.action_history[1:]))
            self.assertTrue(torch.allclose(new.board_history[:-1], old.board_history[1:]))

    def test_slice(self):
        """Test slicing"""
        env = Stratego(num_envs=10, traj_len_per_player=10, move_memory=20)
        env.reset()
        env_state = env.snapshot_state(env.current_step)
        sliced_env_state = env_state.slice(5, 10)
        self.assertEqual(sliced_env_state.num_envs, 5)
        self.assertEqual(sliced_env_state.to_play, env_state.to_play)
        self.assertTrue(torch.allclose(sliced_env_state.boards, env_state.boards[5:10]))
        self.assertTrue(torch.allclose(sliced_env_state.zero_boards, env_state.zero_boards[5:10]))
        self.assertTrue(torch.allclose(sliced_env_state.num_moves, env_state.num_moves[5:10]))
        self.assertTrue(
            torch.allclose(
                sliced_env_state.num_moves_since_last_attack,
                env_state.num_moves_since_last_attack[5:10],
            )
        )
        self.assertTrue(
            torch.allclose(sliced_env_state.terminated_since, env_state.terminated_since[5:10])
        )
        self.assertTrue(
            torch.allclose(sliced_env_state.has_legal_movement, env_state.has_legal_movement[5:10])
        )
        self.assertTrue(
            torch.allclose(sliced_env_state.flag_captured, env_state.flag_captured[5:10])
        )
        self.assertTrue(
            torch.allclose(sliced_env_state.action_history, env_state.action_history[:, 5:10])
        )
        self.assertTrue(
            torch.allclose(sliced_env_state.board_history, env_state.board_history[:, 5:10])
        )
        self.assertTrue(
            torch.allclose(
                sliced_env_state.move_summary_history, env_state.move_summary_history[:, 5:10]
            )
        )
        for i in range(N_PLAYER):
            self.assertTrue(
                torch.allclose(
                    sliced_env_state.chase_state.last_dst_pos[i],
                    env_state.chase_state.last_dst_pos[i][5:10],
                )
            )
            self.assertTrue(
                torch.allclose(
                    sliced_env_state.chase_state.last_src_pos[i],
                    env_state.chase_state.last_src_pos[i][5:10],
                )
            )
            self.assertTrue(
                torch.allclose(
                    sliced_env_state.chase_state.chase_length[i],
                    env_state.chase_state.chase_length[i][5:10],
                )
            )


if __name__ == "__main__":
    unittest.main()
