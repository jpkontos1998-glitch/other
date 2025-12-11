import unittest

import torch as th

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego, set_seed_everywhere

pystratego = get_pystratego()

red_chars = ["C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "B"]
blue_chars = ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "N"]

EMPTY_PIECE_ID = 255


def encode_piece(char, move):
    assert char.upper() in red_chars or char.upper() in blue_chars or char == "a"
    assert move in (".", "@")

    encoding = 0
    if char.upper() in red_chars:
        encoding += red_chars.index(char.upper())
    elif char.upper() in blue_chars:
        encoding += blue_chars.index(char.upper())
    else:
        assert char == "a"
        encoding += 13

    if char.upper() != char:
        encoding += 16  # visibility bit on
    if move == ".":
        encoding += 32  # mobility bit on
    return encoding


def is_scout_move(coords):
    fr = coords[0]
    to = coords[1]
    from_row = int(fr // 10)
    from_col = int(fr % 10)
    to_row = int(to // 10)
    to_col = int(to % 10)
    return abs(from_row - to_row) > 1 or abs(from_col - to_col) > 1


class MoveSummaryTest(unittest.TestCase):
    def test_cells_in_get(self):
        env = Stratego(
            num_envs=16,
            traj_len_per_player=100,
        )

        actions = {}
        for i in range(1024):
            a = env.sample_random_legal_action()

            for j in range(1, env.conf.move_memory + 1):
                step = env.current_step - j
                if step < 0:
                    break

                self.assertTrue(th.allclose(env.played_actions(step), actions[step]))
                coords = th.tensor(
                    pystratego.util.actions_to_abs_coordinates(env.played_actions(step), 0),
                    dtype=th.uint8,
                ).reshape(-1, 2)
                self.assertTrue(th.allclose(env.move_summary(step)[:, 0:2].cpu(), coords))
            self.assertTrue(env.current_step not in actions)
            actions[env.current_step] = a.clone()
            env.apply_actions(a)

    def test_pieces_in_get(self):
        env = Stratego(
            num_envs=16,
            traj_len_per_player=100,
        )

        def adjust_bits(n: int) -> int:
            return ((n & 0b11000000) >> 2) + (n & 0b00001111)

        for i in range(100):
            a = env.sample_random_legal_action()
            snap = env.snapshot_state(env.current_step)
            self.assertEqual(
                snap.board_history.shape, (max(210, env.conf.move_memory), env.num_envs, 1920)
            )
            self.assertEqual(env.current_board_strs, snap.board_strs())
            for j in range(1, env.conf.move_memory + 1):
                step = env.current_step - j
                if step < 0:
                    break

                coords = th.tensor(
                    pystratego.util.actions_to_abs_coordinates(
                        env.played_actions(step), env.acting_player(step)
                    ),
                    dtype=th.uint8,
                ).reshape(-1, 2)
                strs = env.board_strs(step)
                pieces = env.move_summary(step).cpu()[:, 2:4]
                piece_ids = env.move_summary(step).cpu()[:, 4:6]
                for idx in range(env.num_envs):
                    if j <= env.current_num_moves[idx]:
                        if (
                            adjust_bits(
                                snap.board_history[-j, idx, 16 * (int(coords[idx, 1]))].item()
                            )
                            != 29
                        ):
                            self.assertEqual(
                                pieces[idx, 0].item(),
                                adjust_bits(
                                    snap.board_history[-j, idx, 16 * int(coords[idx, 0])].item()
                                ),
                            )
                            self.assertEqual(
                                pieces[idx, 0],
                                encode_piece(
                                    strs[idx][2 * coords[idx, 0]], strs[idx][2 * coords[idx, 0] + 1]
                                ),
                            )
                        else:
                            # Either the piece is visible and revealed, or hidden and exported as HIDDEN_PIECE
                            self.assertTrue(
                                pieces[idx, 0].item() == 0b00001111  # Never moved hidden unknown
                                or pieces[idx, 0].item() == 0b00101111  # Moved hidden unknown
                                or (
                                    pieces[idx, 0].item() & 0b00010000 == 0b00010000
                                    and pieces[idx, 0].item() & 15 < 15
                                )  # Visible and revealed
                            )
                            if (
                                pieces[idx, 0].item() & 0b00010000 == 0b00010000
                                and pieces[idx, 0].item() & 15 < 15
                            ):  # piece is revealed and should match board string
                                self.assertEqual(
                                    pieces[idx, 0],
                                    encode_piece(
                                        strs[idx][2 * coords[idx, 0]],
                                        strs[idx][2 * coords[idx, 0] + 1],
                                    ),
                                )
                            else:  # board string should agree piece is hidden
                                self.assertEqual(
                                    strs[idx][2 * coords[idx, 0] + 1],
                                    strs[idx][2 * coords[idx, 0] + 1].upper(),
                                )
                                if (
                                    pieces[idx, 0].item() == 0b00001111
                                ):  # board string should agree piece has not moved
                                    self.assertEqual(strs[idx][2 * coords[idx, 0] + 1], "@")
                                else:  # board string should agree piece has moved
                                    self.assertEqual(strs[idx][2 * coords[idx, 0] + 1], ".")
                        self.assertEqual(
                            pieces[idx, 1].item(),
                            adjust_bits(
                                snap.board_history[-j, idx, 16 * int(coords[idx, 1])].item()
                            ),
                        )
                        self.assertEqual(
                            piece_ids[idx, 0].item(),
                            snap.board_history[-j, idx, 16 * int(coords[idx, 0]) + 1].item(),
                        )
                        self.assertEqual(
                            piece_ids[idx, 1].item(),
                            snap.board_history[-j, idx, 16 * int(coords[idx, 1]) + 1].item(),
                        )
                    self.assertEqual(
                        pieces[idx, 1],
                        encode_piece(
                            strs[idx][2 * coords[idx, 1]], strs[idx][2 * coords[idx, 1] + 1]
                        ),
                    )

            env.apply_actions(a)

    def test_inception(self):
        num_envs = 8
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=100,
        )
        actions = []
        num_moves = []
        num_levels = 100
        snap = env.snapshot_state(env.current_step)
        snaps = []
        env.change_reset_behavior_to_env_state(snap)
        next_is_terminal = []
        for i in range(num_levels):
            snaps.append(env.current_state)
            a = env.sample_random_legal_action()
            actions.append(a)
            num_moves.append(env.current_num_moves)
            set_seed_everywhere(i)
            env.apply_actions(a)
            next_is_terminal.append(env.current_is_terminal)
        inception_break = False
        max_inception = 0
        cur_inception = 0
        for i in range(num_levels):
            if snap.terminated_since.max() == 0 and not inception_break:
                env.change_reset_behavior_to_env_state(snap)
                self.assertTrue(th.allclose(num_moves[i], env.current_num_moves))
                set_seed_everywhere(i)
                env.apply_actions(actions[i])
                snap = env.current_state
                cur_inception += 1
            elif snaps[i].terminated_since.max() == 0 and inception_break:
                env.change_reset_behavior_to_env_state(snaps[i])
                self.assertTrue(th.allclose(num_moves[i], env.current_num_moves))
                set_seed_everywhere(i)
                env.apply_actions(actions[i])
                snap = env.current_state
                inception_break = False
                max_inception = max(max_inception, cur_inception)
                cur_inception = 0
            else:
                inception_break = True
                a = env.sample_random_legal_action()
                actions.append(a)


if __name__ == "__main__":
    unittest.main()
