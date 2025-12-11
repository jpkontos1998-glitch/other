import unittest

import torch
from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()

letters_pl0 = ["c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "b"]
letters_pl1 = ["o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "n"]
non_pieces = ["a", "_"]
letters = [letters_pl0, letters_pl1]

EMPTY_PIECE_ID = 255

SPY_TYPE = 0
SCOUT_TYPE = 1
MINER_TYPE = 2
MARSHAL_TYPE = 9
FLAG_TYPE = 10
BOMB_TYPE = 11

ATTACKER_WON = 1
KAMIKAZE = 0
DEFENDER_WON = -1


def battle_outcome(attacker_type, defender_type):
    if attacker_type == defender_type:
        return KAMIKAZE
    if defender_type == BOMB_TYPE and attacker_type == MINER_TYPE:
        return ATTACKER_WON
    if attacker_type == SPY_TYPE and defender_type == MARSHAL_TYPE:
        return ATTACKER_WON
    if attacker_type > defender_type:
        return ATTACKER_WON
    return DEFENDER_WON


class PieceIDTest(unittest.TestCase):
    def test_uniqueness(self):
        num_envs = 32
        num_steps = 2000
        for barrage in [True, False]:
            env = Stratego(num_envs=num_envs, traj_len_per_player=100, barrage=barrage)
            action_tensor = torch.zeros(num_envs, dtype=torch.int32, device="cuda")

            for _ in range(num_steps):
                env.sample_random_legal_action(action_tensor)
                print(action_tensor)
                env.apply_actions(action_tensor)
                piece_ids = env.current_piece_ids.cpu()

                unique, counts = torch.unique(piece_ids, return_counts=True)
                non_empty_counts = counts[unique != EMPTY_PIECE_ID]
                self.assertTrue(torch.all(non_empty_counts <= num_envs))

    def test_range(self):
        num_envs = 32
        num_steps = 2000
        env = Stratego(num_envs=num_envs, traj_len_per_player=100, barrage=True)
        action_tensor = torch.zeros(num_envs, dtype=torch.int32, device="cuda")
        for i in range(num_steps):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            piece_ids = env.current_piece_ids.cpu()
            unique, counts = torch.unique(piece_ids, return_counts=True)
            self.assertTrue(
                torch.all(
                    (0 <= unique)
                    | (unique <= 39)
                    | (60 <= unique)
                    | (unique <= 99)
                    | (unique == EMPTY_PIECE_ID)
                )
            )

    def test_board_str_consistency(self):
        num_envs = 32
        num_steps = 200
        env = Stratego(
            num_envs=num_envs,
            traj_len_per_player=100,
        )
        action_tensor = torch.zeros(num_envs, dtype=torch.int32, device="cuda")
        for i in range(num_steps):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
            board_strs = env.current_board_strs
            piece_ids = env.current_piece_ids.cpu()
            for j in range(num_envs):
                current_player = env.current_player
                env_board_str = board_strs[j][::2].lower()  # strip movement & visibility info
                env_piece_ids = piece_ids[j].flatten()
                if current_player == 1:
                    env_board_str = env_board_str[::-1]
                for k in range(100):
                    if env_board_str[k] in non_pieces:
                        self.assertEqual(env_piece_ids[k], EMPTY_PIECE_ID)
                    else:
                        if env_board_str[k] in letters[current_player]:
                            self.assertTrue(0 <= env_piece_ids[k] <= 39)
                        else:
                            self.assertTrue(60 <= env_piece_ids[k] <= 99)

    def test_board_id_temporal_consistency(self):
        num_traj = 1000
        env = Stratego(num_envs=1, traj_len_per_player=100, barrage=True)
        for _ in range(num_traj):
            env.reset()
            action_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
            board = torch.empty(10**2, dtype=torch.int32, device="cuda")
            initial_board = env.current_board_strs[0][::2]
            for i in range(100):
                if i <= 39 and initial_board[i] != "a":
                    board[i] = i
                elif i >= 60 and initial_board[i] != "a":
                    board[i] = i
                else:
                    board[i] = EMPTY_PIECE_ID
            board = board
            while True:
                env.sample_random_legal_action(action_tensor)
                src, dst = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )[0]
                if board[dst] == EMPTY_PIECE_ID:
                    board[dst] = board[src]
                    board[src] = EMPTY_PIECE_ID
                else:
                    src_type = (
                        env.current_piece_type_onehot.view(100, -1)[src].int().argmax().item()
                    )
                    dst_type = (
                        env.current_piece_type_onehot.view(100, -1)[dst].int().argmax().item()
                    )
                    if env.current_player == 1:
                        src = 99 - src
                        dst = 99 - dst
                    outcome = battle_outcome(src_type, dst_type)
                    if outcome == ATTACKER_WON:
                        board[dst] = board[src]
                        board[src] = EMPTY_PIECE_ID
                    elif outcome == DEFENDER_WON:
                        board[src] = EMPTY_PIECE_ID
                    else:
                        board[src] = EMPTY_PIECE_ID
                        board[dst] = EMPTY_PIECE_ID
                env.apply_actions(action_tensor)
                if env.is_terminal:
                    break
                if env.current_player == 0:
                    self.assertTrue(torch.all(board == env.current_piece_ids[0].flatten()))
                else:
                    self.assertTrue(
                        torch.all(board == 99 - env.current_piece_ids[0].flatten()[::-1])
                    )


if __name__ == "__main__":
    unittest.main()
