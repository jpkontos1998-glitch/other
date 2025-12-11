import unittest
import random

import torch as th

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()


class HistorySnapshotTest(unittest.TestCase):
    def test_small(self):
        def compare_tensors(A, B, env=None):
            if A.shape != B.shape:
                self.fail("Shape mismatch: {} vs {}".format(A.shape, B.shape))
            if not th.allclose(A, B):
                print(A)
                print(B)
                print(env)
                self.fail()

        num_envs = 1
        num_steps = 2
        env = Stratego(num_envs, num_steps // 2)

        action_tensor = th.zeros(env.num_envs, dtype=th.int32, device="cuda")
        env_idx = th.randint(0, num_envs, (1,)).item()
        for t in range(num_steps):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        query_step = 1
        (even_states, odd_states) = env.snapshot_env_history(query_step, env_idx)
        nm = env.num_moves(query_step)[env_idx]
        expected_even, expected_odd = 0, 0
        for i, t in enumerate(range(query_step - nm, query_step + 1)):
            nmt = env.num_moves(t)[env_idx]
            if nmt % 2 == 0:
                expected_even += 1
            else:
                expected_odd += 1
            self.assertEqual(nmt, i)

        self.assertEqual(even_states.num_envs, expected_even)
        self.assertEqual(odd_states.num_envs, expected_odd)

        even_env = Stratego(
            even_states.num_envs,
            1,
            reset_state=even_states,
            reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
        )
        odd_env = Stratego(
            odd_states.num_envs,
            1,
            reset_state=odd_states,
            reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
        )
        for nmt, t in enumerate(range(query_step - nm, query_step + 1)):
            tgt = even_states if nmt % 2 == 0 else odd_states
            tgt_env = even_env if nmt % 2 == 0 else odd_env

            self.assertEqual(tgt_env.current_board_strs[nmt // 2], env.board_strs(t)[env_idx])
            self.assertEqual(tgt_env.current_player, env.acting_player(t))
            compare_tensors(env.env.get_board_tensor(t)[env_idx], tgt.boards[nmt // 2])
            compare_tensors(
                env.env.get_board_tensor(t)[env_idx], tgt_env.env.get_board_tensor(0)[nmt // 2]
            )
            for i, ch in enumerate(env.INFOSTATE_CHANNEL_DESCRIPTION):
                compare_tensors(
                    tgt_env.current_infostate_tensor[nmt // 2][i],
                    env.infostate_tensor(t)[env_idx][i],
                )

    def test_random(self):
        """
        This tests:
        - We correctly remember information from multiple games ago
        - We correctly remember information from the current game
        - We snapshot correctly when the step query is is deeply terminated
        - We snapshot correctly when the step query is in the middle of a game
        - Everything still works after the buffer is overwritten
        """

        def compare_tensors(A, B, env=None):
            if A.shape != B.shape:
                self.fail("Shape mismatch: {} vs {}".format(A.shape, B.shape))
            if not th.allclose(A, B):
                print(A)
                print(B)
                print(env)
                self.fail()

        num_envs = 20
        num_steps = 2000
        traj_len_per_player = num_steps
        max_num_moves = num_steps // 2
        env = Stratego(num_envs, traj_len_per_player, max_num_moves=max_num_moves)

        action_tensor = th.zeros(env.num_envs, dtype=th.int32, device="cuda")
        for _ in range(
            5
        ):  # Run through a few times to check it still works when the buffer is rewritten
            last_steps = []
            env_idx = th.randint(0, num_envs, (1,)).item()
            for t in range(num_steps):
                if env.current_is_pl0_first_move[env_idx]:
                    if env.current_step > 0:
                        last_steps.append(
                            env.current_step - 1
                        )  # Include the full game from the last termination step
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)
            last_steps.append(env.current_step)  # Also test the current game

            for i in range(len(last_steps)):
                last_step = last_steps[i]
                if random.random() < 0.5:  # Sometimes test in the middle of a game
                    last_step = random.randint(
                        last_step - env.num_moves(last_step)[env_idx] + 1, last_step - 1
                    )
                (even_states, odd_states) = env.snapshot_env_history(last_step, env_idx)
                nm = env.num_moves(last_step)[env_idx]
                expected_even, expected_odd = 0, 0
                for i, t in enumerate(range(last_step - nm, last_step + 1)):
                    nmt = env.num_moves(t)[env_idx]
                    if nmt % 2 == 0:
                        expected_even += 1
                    else:
                        expected_odd += 1
                    self.assertEqual(nmt, i)

                self.assertEqual(even_states.num_envs, expected_even)
                self.assertEqual(odd_states.num_envs, expected_odd)

                even_env = Stratego(
                    even_states.num_envs,
                    2,
                    reset_state=even_states,
                    reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
                    max_num_moves=max_num_moves,
                )
                odd_env = Stratego(
                    odd_states.num_envs,
                    2,
                    reset_state=odd_states,
                    reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
                    max_num_moves=max_num_moves,
                )
                for nmt, t in enumerate(range(last_step - nm, last_step + 1)):
                    tgt = even_states if nmt % 2 == 0 else odd_states
                    tgt_env = even_env if nmt % 2 == 0 else odd_env

                    self.assertEqual(
                        tgt_env.current_board_strs[nmt // 2], env.board_strs(t)[env_idx]
                    )

                    self.assertEqual(tgt_env.current_player, env.acting_player(t))
                    compare_tensors(tgt_env.current_piece_ids[nmt // 2], env.piece_ids(t)[env_idx])
                    compare_tensors(
                        tgt_env.current_legal_action_mask[nmt // 2],
                        env.legal_action_mask(t)[env_idx],
                    )

                    self.assertEqual(env.num_moves(t)[env_idx], tgt.num_moves[nmt // 2])

                    self.assertEqual(
                        env.num_moves_since_last_attack(t)[env_idx],
                        tgt.num_moves_since_last_attack[nmt // 2],
                    )

                    self.assertEqual(
                        env.env.get_terminated_since(t)[env_idx], tgt.terminated_since[nmt // 2]
                    )

                    self.assertEqual(
                        env.env.get_has_legal_movement(t)[env_idx], tgt.has_legal_movement[nmt // 2]
                    )

                    self.assertEqual(
                        env.env.get_flag_captured(t)[env_idx], tgt.flag_captured[nmt // 2]
                    )

                    compare_tensors(env.env.get_board_tensor(t)[env_idx], tgt.boards[nmt // 2])
                    self.assertEqual(tgt.board_strs()[nmt // 2], env.board_strs(t)[env_idx])

                    compare_tensors(
                        env.snapshot_state(t).action_history[:, env_idx],
                        tgt.action_history[:, nmt // 2],
                    )
                    compare_tensors(
                        env.snapshot_state(t).move_summary_history[:, env_idx],
                        tgt.move_summary_history[:, nmt // 2],
                    )
                    compare_tensors(
                        env.snapshot_state(t).board_history[:, env_idx],
                        tgt.board_history[:, nmt // 2],
                    )

    def test_border_cases(self):
        """
        This tests:
        - We correctly handle the case where the step query is 0
        - We correctly handle the case where the step query is on boundary of buffer
        - We correctly handle the case where the step query is the last step
        - We still correctly handle these cases after the buffer is overwritten
        """

        def compare_tensors(A, B, env=None):
            if A.shape != B.shape:
                self.fail("Shape mismatch: {} vs {}".format(A.shape, B.shape))
            if not th.allclose(A, B):
                print(A)
                print(B)
                print(env)
                self.fail()

        num_envs = 20
        num_steps = 2000
        max_num_moves = num_steps // 2
        env = Stratego(num_envs, num_steps, max_num_moves=max_num_moves)

        action_tensor = th.zeros(env.num_envs, dtype=th.int32, device="cuda")
        for _ in range(
            5
        ):  # Run through a few times to check it still works when the buffer is rewritten
            last_steps = []
            env_idx = th.randint(0, num_envs, (1,)).item()
            for t in range(num_steps):
                if env.current_step % env.env.buf_size in [
                    0,
                    1,
                    env.env.buf_size - 2,
                    env.env.buf_size - 1,
                ]:
                    last_steps.append(env.current_step)
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)

            for i in range(len(last_steps)):
                last_step = last_steps[i]
                (even_states, odd_states) = env.snapshot_env_history(last_step, env_idx)
                nm = env.num_moves(last_step)[env_idx]
                expected_even, expected_odd = 0, 0
                for i, t in enumerate(range(last_step - nm, last_step + 1)):
                    nmt = env.num_moves(t)[env_idx]
                    if nmt % 2 == 0:
                        expected_even += 1
                    else:
                        expected_odd += 1
                    self.assertEqual(nmt, i)

                self.assertEqual(even_states.num_envs, expected_even)
                self.assertEqual(odd_states.num_envs, expected_odd)

                even_env = (
                    Stratego(
                        even_states.num_envs,
                        2,
                        reset_state=even_states,
                        reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
                        max_num_moves=max_num_moves,
                    )
                    if even_states.num_envs != 0
                    else None
                )
                odd_env = (
                    Stratego(
                        odd_states.num_envs,
                        2,
                        reset_state=odd_states,
                        reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
                        max_num_moves=max_num_moves,
                    )
                    if odd_states.num_envs != 0
                    else None
                )
                for nmt, t in enumerate(range(last_step - nm, last_step + 1)):
                    tgt = even_states if nmt % 2 == 0 else odd_states
                    tgt_env = even_env if nmt % 2 == 0 else odd_env

                    compare_tensors(
                        tgt_env.current_infostate_tensor[nmt // 2], env.infostate_tensor(t)[env_idx]
                    )
                    compare_tensors(tgt_env.current_piece_ids[nmt // 2], env.piece_ids(t)[env_idx])
                    compare_tensors(
                        tgt_env.current_legal_action_mask[nmt // 2],
                        env.legal_action_mask(t)[env_idx],
                    )

                    self.assertEqual(env.num_moves(t)[env_idx], tgt.num_moves[nmt // 2])

                    self.assertEqual(
                        env.num_moves_since_last_attack(t)[env_idx],
                        tgt.num_moves_since_last_attack[nmt // 2],
                    )
                    self.assertEqual(tgt_env.current_player, env.acting_player(t))
                    self.assertEqual(
                        env.env.get_terminated_since(t)[env_idx], tgt.terminated_since[nmt // 2]
                    )

                    self.assertEqual(
                        env.env.get_has_legal_movement(t)[env_idx], tgt.has_legal_movement[nmt // 2]
                    )

                    self.assertEqual(
                        env.env.get_flag_captured(t)[env_idx], tgt.flag_captured[nmt // 2]
                    )

                    compare_tensors(env.env.get_board_tensor(t)[env_idx], tgt.boards[nmt // 2])
                    self.assertEqual(tgt.board_strs()[nmt // 2], env.board_strs(t)[env_idx])

                    compare_tensors(
                        env.snapshot_state(t).action_history[:, env_idx],
                        tgt.action_history[:, nmt // 2],
                    )
                    compare_tensors(
                        env.snapshot_state(t).move_summary_history[:, env_idx],
                        tgt.move_summary_history[:, nmt // 2],
                    )
                    compare_tensors(
                        env.snapshot_state(t).board_history[:, env_idx],
                        tgt.board_history[:, nmt // 2],
                    )


if __name__ == "__main__":
    unittest.main()
