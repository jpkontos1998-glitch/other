import sys
import os
import unittest

import torch


from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego
from pyengine.utils.init_helpers import BLUE_CHAR_TO_RED_CHAR

path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "external")
sys.path.append(path)

from our_tensors import parse_game_data, make_filled_initial_board  # noqa E402

pystratego = get_pystratego()

rollout_env = Stratego(num_envs=1, traj_len_per_player=2000, full_info=0, barrage=1)
reconstruction_env = Stratego(num_envs=1, traj_len_per_player=2000, full_info=0, barrage=1)


def write_data(data, dummy_fn):
    if os.path.exists(dummy_fn):
        os.remove(dummy_fn)
    with open(dummy_fn, "w") as f:
        for row in data:
            f.write(f"{row}\n")


def reconstruct_env(fn):
    with open(fn, "r") as f:
        data = f.readlines()
    our_player, initial_board, actions = parse_game_data(data)
    filled_initial_board, piece_cells = make_filled_initial_board(
        our_player, initial_board, actions
    )
    env = Stratego(num_envs=1, barrage=True, traj_len_per_player=2000)
    arr_pl0 = filled_initial_board[:40].upper()
    arr_pl1 = filled_initial_board[-40:][::-1].upper()
    remapped_arr_pl1 = []
    for c in arr_pl1:
        if c in BLUE_CHAR_TO_RED_CHAR:
            remapped_arr_pl1.append(BLUE_CHAR_TO_RED_CHAR[c])
        else:
            remapped_arr_pl1.append(c)
    arr_pl1 = "".join(remapped_arr_pl1)

    env.change_reset_behavior_to_random_initial_arrangement(([arr_pl0], [arr_pl1]))
    env.reset()
    for t, src, dst, attacker_type, defender_type in actions:
        a = pystratego.util.abs_coordinates_to_actions([(src, dst)], t % 2)
        action_tensor = torch.tensor(a, device=env.conf.cuda_device, dtype=torch.int32)
        env.apply_actions(action_tensor)
    return env


def check_tensors(env1, env2):
    truth_values = []
    truth_values.append(env1.current_step == env2.current_step)
    truth_values.append(env1.current_player == env2.current_player)
    truth_values.append((env1.current_infostate_tensor == env2.current_infostate_tensor).all())
    truth_values.append((env1.current_legal_action_mask == env2.current_legal_action_mask).all())
    truth_values.append((env1.current_is_terminal == env2.current_is_terminal).all())
    truth_values.append(
        (env1.current_move_summary_history_tensor == env2.current_move_summary_history_tensor).all()
    )
    truth_values.append((env1.current_num_moves == env2.current_num_moves).all())
    return all(truth_values)


def human(piece_encoding):
    return {
        0: "spy",
        1: "scout",
        2: "miner",
        3: "sergeant",
        4: "lieutenant",
        5: "captain",
        6: "major",
        7: "colonel",
        8: "general",
        9: "marshal",
        10: "flag",
        11: "bomb",
        # 12: "lake",  # not possible
        13: "empty",
    }[piece_encoding % 16]


class ReconstructionTest(unittest.TestCase):
    def test_tensor_reconstruction_pl0(self):
        env = Stratego(num_envs=1, barrage=True, traj_len_per_player=2000)
        controlled_side = 0

        board = list(env.current_board_strs[0][::2])
        for i in range(60, 100):
            if board[i] == board[i].upper():  # Hidden piece
                board[i] = "!"

        dummy_fn = "tmp.txt"
        data = [controlled_side, "".join(board)]
        write_data(data, dummy_fn)

        reconstructed_env = reconstruct_env(dummy_fn)

        self.assertTrue(check_tensors(env, reconstructed_env))

        for t in range(500):
            if env.current_is_terminal.item():
                break

            a = env.sample_random_legal_action()
            env.apply_actions(a)
            ms = env.move_summary(env.current_step - 1).cpu()[0]
            src_cell = ms[0].item()
            dst_cell = ms[1].item()
            src_piece = ms[2].item()
            dst_piece = ms[3].item()

            if env.current_player == 0:
                src_cell = 99 - src_cell
                dst_cell = 99 - dst_cell

            was_battle = human(dst_piece) != "empty"
            if not was_battle:
                data.append(f"{env.current_step:3}  {src_cell:2}  {dst_cell:2}")
            else:
                data.append(
                    f"{env.current_step:3}  {src_cell:2}  {dst_cell:2}  {human(src_piece):10}  {human(dst_piece):10}"
                )

            write_data(data, dummy_fn)

            if env.current_player == controlled_side:
                reconstructed_env = reconstruct_env(dummy_fn)
                self.assertTrue(check_tensors(env, reconstructed_env))

        os.remove(dummy_fn)

    def test_tensor_reconstruction_pl1(self):
        env = Stratego(num_envs=1, barrage=True, traj_len_per_player=2000)
        controlled_side = 1

        board = list(env.current_board_strs[0][::2])
        for i in range(40):
            if board[i] == board[i].upper():  # Hidden piece
                board[i] = "?"

        dummy_fn = "tmp.txt"
        data = [controlled_side, "".join(board)]
        write_data(data, dummy_fn)

        reconstructed_env = reconstruct_env(dummy_fn)

        for t in range(500):
            if env.current_is_terminal.item():
                break

            a = env.sample_random_legal_action()
            env.apply_actions(a)
            ms = env.move_summary(env.current_step - 1).cpu()[0]
            src_cell = ms[0].item()
            dst_cell = ms[1].item()
            src_piece = ms[2].item()
            dst_piece = ms[3].item()

            if env.current_player == 0:
                src_cell = 99 - src_cell
                dst_cell = 99 - dst_cell

            was_battle = human(dst_piece) != "empty"
            if not was_battle:
                data.append(f"{env.current_step:3}  {src_cell:2}  {dst_cell:2}")
            else:
                data.append(
                    f"{env.current_step:3}  {src_cell:2}  {dst_cell:2}  {human(src_piece):10}  {human(dst_piece):10}"
                )

            write_data(data, dummy_fn)

            if env.current_player == controlled_side:
                reconstructed_env = reconstruct_env(dummy_fn)
                self.assertTrue(check_tensors(env, reconstructed_env))

        os.remove(dummy_fn)


if __name__ == "__main__":
    unittest.main()
