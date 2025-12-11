import unittest

import torch

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

from pyengine.utils.constants import NOTMOVE_ID, EMPTY_PIECE_ID, N_BOARD_CELL

pystratego = get_pystratego()


class LastMovesTest(unittest.TestCase):
    def test_cells(self):
        env = Stratego(
            num_envs=16,
            traj_len_per_player=100,
        )

        for i in range(1024):
            a = env.sample_random_legal_action()

            for j in range(1, env.conf.move_memory + 1):
                step = env.current_step - j
                if step < 0:
                    break

                last_two_moves = env.last_two_moves(step)
                if step - 1 < 0:
                    prev_terminals = torch.ones(env.num_envs, dtype=torch.bool)
                else:
                    prev_terminals = env.is_terminal(step - 1)
                self.assertTrue((last_two_moves[prev_terminals, 0:2] == NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_terminals, 0:2] != NOTMOVE_ID).all())
                if step - 2 < 0:
                    prev_prev_terminals = torch.ones(env.num_envs, dtype=torch.bool)
                else:
                    prev_prev_terminals = env.is_terminal(step - 2)
                self.assertTrue((last_two_moves[prev_prev_terminals, 6:8] == NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_prev_terminals, 6:8] != NOTMOVE_ID).all())

                prev_terminals = prev_terminals.cpu()
                prev_prev_terminals = prev_prev_terminals.cpu()

                if step - 1 < 0:
                    continue
                coords = torch.tensor(
                    pystratego.util.actions_to_abs_coordinates(
                        env.played_actions(step - 1), env.acting_player(step - 1)
                    ),
                    dtype=torch.uint8,
                ).reshape(-1, 2)
                if env.acting_player(step) == 1:
                    coords = 99 - coords
                self.assertTrue(
                    torch.allclose(
                        last_two_moves[~prev_terminals, 0:2].cpu(), coords[~prev_terminals]
                    )
                )
                if step - 2 < 0:
                    continue
                coords = torch.tensor(
                    pystratego.util.actions_to_abs_coordinates(
                        env.played_actions(step - 2), env.acting_player(step - 2)
                    ),
                    dtype=torch.uint8,
                ).reshape(-1, 2)
                if env.acting_player(step) == 1:
                    coords = 99 - coords
                self.assertTrue(
                    torch.allclose(
                        last_two_moves[~prev_prev_terminals, 6:8].cpu(),
                        coords[~prev_prev_terminals],
                    )
                )
            env.apply_actions(a)

    def test_ids(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
        )

        for i in range(1024):
            a = env.sample_random_legal_action()

            for j in range(1, env.conf.move_memory + 1):
                step = env.current_step - j
                if step < 0:
                    break

                last_two_moves = env.last_two_moves(step)
                if step - 1 < 0:
                    prev_terminals = torch.ones(env.num_envs, dtype=torch.bool)
                else:
                    prev_terminals = env.is_terminal(step - 1)
                self.assertTrue((last_two_moves[prev_terminals, 4:6] == NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_terminals, 4:6] != NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_terminals, 4] != EMPTY_PIECE_ID).all())
                if step - 2 < 0:
                    prev_prev_terminals = torch.ones(env.num_envs, dtype=torch.bool)
                else:
                    prev_prev_terminals = env.is_terminal(step - 2)
                self.assertTrue((last_two_moves[prev_prev_terminals, 10:12] == NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_prev_terminals, 10:12] != NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_prev_terminals, 10] != EMPTY_PIECE_ID).all())

                prev_terminals = prev_terminals.cpu()
                prev_prev_terminals = prev_prev_terminals.cpu()

                if step - 1 < 0:
                    continue
                coords = (
                    torch.tensor(
                        pystratego.util.actions_to_abs_coordinates(
                            env.played_actions(step - 1), env.acting_player(step - 1)
                        ),
                        dtype=torch.uint8,
                    )
                    .reshape(-1, 2)
                    .long()
                )
                if env.acting_player(step) == 1:
                    coords = 99 - coords
                piece_ids = env.piece_ids(step - 1).flatten(start_dim=-2).cpu()
                is_piece = piece_ids < N_BOARD_CELL
                piece_ids[is_piece] = 99 - piece_ids[is_piece]
                piece_ids = piece_ids.flip(-1)
                move_ids = piece_ids.gather(-1, coords)
                # Src ID should belong to opponent
                self.assertTrue((60 <= last_two_moves[~prev_terminals, 4]).all())
                self.assertTrue((last_two_moves[~prev_terminals, 4] < 100).all())
                self.assertTrue(
                    torch.allclose(
                        last_two_moves[~prev_terminals, 4:6].cpu(), move_ids[~prev_terminals]
                    )
                )

                if step - 2 < 0:
                    continue
                coords = (
                    torch.tensor(
                        pystratego.util.actions_to_abs_coordinates(
                            env.played_actions(step - 2), env.acting_player(step - 2)
                        ),
                        dtype=torch.uint8,
                    )
                    .reshape(-1, 2)
                    .long()
                )
                if env.acting_player(step) == 1:
                    coords = 99 - coords
                piece_ids = env.piece_ids(step - 2).flatten(start_dim=-2).cpu()
                move_ids = piece_ids.gather(-1, coords)
                # Src ID should belong to us
                self.assertTrue((0 <= last_two_moves[~prev_prev_terminals, 10]).all())
                self.assertTrue((last_two_moves[~prev_prev_terminals, 10] < 40).all())
                self.assertTrue(
                    torch.allclose(
                        move_ids[~prev_prev_terminals],
                        last_two_moves[~prev_prev_terminals, 10:12].cpu(),
                    )
                )

            env.apply_actions(a)

    def test_types(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
        )

        for i in range(1024):
            a = env.sample_random_legal_action()

            for j in range(1, env.conf.move_memory + 1):
                step = env.current_step - j
                if step < 0:
                    break

                last_two_moves = env.last_two_moves(step)
                if step - 1 < 0:
                    prev_terminals = torch.ones(env.num_envs, dtype=torch.bool)
                else:
                    prev_terminals = env.is_terminal(step - 1)
                self.assertTrue((last_two_moves[prev_terminals, 2:4] == NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_terminals, 2:4] != NOTMOVE_ID).all())
                if step - 2 < 0:
                    prev_prev_terminals = torch.ones(env.num_envs, dtype=torch.bool)
                else:
                    prev_prev_terminals = env.is_terminal(step - 2)
                self.assertTrue((last_two_moves[prev_prev_terminals, 8:10] == NOTMOVE_ID).all())
                self.assertTrue((last_two_moves[~prev_prev_terminals, 8:10] != NOTMOVE_ID).all())

            env.apply_actions(a)

    def test_sanity(self):
        num_envs = 1
        max_num_moves = 1000
        env = Stratego(
            num_envs=num_envs, traj_len_per_player=max_num_moves, max_num_moves=max_num_moves
        )
        for _ in range(5000):
            if env.current_is_pl0_first_move:
                last_moves_ls = {0: [], 1: []}
            last_moves_ls[env.current_player].append(env.current_last_two_moves)
            env.apply_actions(env.sample_random_legal_action())
            if env.current_is_newly_terminal:
                for i, state in enumerate(env.snapshot_env_history(env.current_step - 1, 0)):
                    seq_env = Stratego(
                        state.num_envs,
                        2,
                        quiet=2,
                        reset_state=state,
                        reset_behavior=pystratego.ResetBehavior.CUSTOM_ENV_STATE,
                        max_num_moves_between_attacks=env.conf.max_num_moves_between_attacks,
                        max_num_moves=env.conf.max_num_moves,
                    )
                    assert not (seq_env.current_last_two_moves == NOTMOVE_ID).all()

                    for j in range(seq_env.current_last_two_moves.size(0)):
                        if not torch.allclose(
                            seq_env.current_last_two_moves[j],
                            last_moves_ls[i][j],
                        ):
                            print(f"Mismatch at env {i}, step {j}")
                            print(seq_env.current_last_two_moves[j])
                            print(last_moves_ls[i][j])
                    assert torch.allclose(
                        seq_env.current_last_two_moves, torch.cat(last_moves_ls[i])
                    )


if __name__ == "__main__":
    unittest.main()
