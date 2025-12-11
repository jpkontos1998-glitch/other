import unittest
from dataclasses import dataclass

import torch

from pyengine.core.env import Stratego
from pyengine.utils import set_seed_everywhere, get_pystratego

pystratego = get_pystratego()

RED_PIECES = [
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "B",
]
BLUE_PIECES = [
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "N",
]


@dataclass
class Piece:
    piece_id: int
    piece_type: str
    has_moved: bool

    @property
    def visible(self):
        return self.piece_type.islower() or self.piece_type == "_"

    @property
    def color(self):
        if self.piece_type == "a":
            return 0
        elif self.piece_type.upper() in RED_PIECES:
            return 1
        elif self.piece_type.upper() in BLUE_PIECES:
            return 2
        else:
            assert self.piece_type == "_"
            return 3

    @property
    def rank(self):
        if self.color == 0:
            return 13
        elif self.color == 1:
            return RED_PIECES.index(self.piece_type.upper())
        elif self.color == 2:
            return BLUE_PIECES.index(self.piece_type.upper())
        else:
            return 12

    def beats(self, other: "Piece"):
        assert self.color in (1, 2)
        assert other.color in (3 - self.color, 0)
        if self.rank == 10:  # Flag can never win
            return False
        if other.color == 0:
            return True
        else:  # opponent piece
            return (
                (self.rank >= other.rank)
                or (other.rank == 10)  # Flag always loses
                or (self.rank == 2 and other.rank == 11)  # Miner -> bomb
                or (self.rank == 0 and other.rank == 9)  # Spy -> Marshal
            )


class StrategoChecker:
    def __init__(self, board_str, piece_ids):
        # print(board_str)
        # print(piece_ids)
        initial_board = [[None for _ in range(10)] for _ in range(10)]
        used_piece_ids = set()
        for i in range(100):
            assert board_str[2 * i + 1] in (".", "@")
            piece_id = piece_ids.view(-1)[i].item()
            initial_board[i // 10][i % 10] = Piece(
                piece_id=piece_id,
                piece_type=board_str[2 * i],
                has_moved=True if board_str[2 * i + 1] == "." else False,
            )

            if i < 40:
                assert initial_board[i // 10][i % 10].color in (0, 1)
            elif i >= 60:
                assert initial_board[i // 10][i % 10].color in (0, 2)
            else:
                assert initial_board[i // 10][i % 10].color in (0, 3)

            if board_str[2 * i] in ("A", "_"):
                assert piece_id == 255

            if piece_id != 255:
                assert 0 <= piece_id < 100
                assert initial_board[i // 10][i % 10].piece_id not in used_piece_ids
                used_piece_ids.add(initial_board[i // 10][i % 10].piece_id)
        assert len(used_piece_ids) in (80, 16)
        self.board = initial_board
        self.is_dead = [False for _ in range(100)]
        self.next_player = 1

    def apply_action(self, action):
        src, dst = pystratego.util.actions_to_abs_coordinates(
            torch.tensor([action], dtype=torch.int32), self.next_player - 1
        )[0]
        src_piece = self.board[src // 10][src % 10]
        dst_piece = self.board[dst // 10][dst % 10]
        assert src_piece.color == self.next_player
        assert dst_piece.color in (0, 3 - self.next_player)

        self.board[src // 10][src % 10] = Piece(piece_id=255, piece_type="a", has_moved=False)
        if dst_piece.color == 0:  # Move to empty, do not reveal
            self.board[dst // 10][dst % 10] = src_piece
            self.board[dst // 10][dst % 10].has_moved = True
            if abs(dst // 10 - src // 10) + abs(dst % 10 - src % 10) > 1:
                # Mark scout visible
                self.board[dst // 10][dst % 10].piece_type = src_piece.piece_type.lower()
        else:
            assert not self.is_dead[src_piece.piece_id]
            assert not self.is_dead[dst_piece.piece_id]

            if src_piece.rank == dst_piece.rank:
                self.is_dead[src_piece.piece_id] = True
                self.is_dead[dst_piece.piece_id] = True
                self.board[dst // 10][dst % 10] = Piece(
                    piece_id=255, piece_type="a", has_moved=False
                )
            elif src_piece.beats(dst_piece):
                self.is_dead[dst_piece.piece_id] = True
                self.board[dst // 10][dst % 10] = src_piece
                self.board[dst // 10][dst % 10].piece_type = src_piece.piece_type.lower()
                self.board[dst // 10][dst % 10].has_moved = True
            else:
                assert dst_piece.beats(src_piece)
                self.is_dead[src_piece.piece_id] = True
                self.board[dst // 10][dst % 10].piece_type = dst_piece.piece_type.lower()

        self.next_player = 3 - self.next_player

    def backend_piece_ids(self):
        piece_ids = torch.zeros((10, 10), dtype=torch.uint8)
        for i in range(100):
            piece_id = self.board[i // 10][i % 10].piece_id
            if piece_id == 255:
                piece_ids[i // 10, i % 10] = 255
            else:
                assert piece_id < 100
                piece_ids[i // 10, i % 10] = piece_id if piece_id < 40 else 99 - piece_id
        return piece_ids

    def backend_is_dead(self):
        out = torch.zeros((80), dtype=torch.uint8)
        for piece_id in range(40):
            out[piece_id] = self.is_dead[piece_id]
        for piece_id in range(99, 59, -1):
            out[40 + 99 - piece_id] = self.is_dead[piece_id]
        return out


class StrategoEnvTest(unittest.TestCase):
    def test_game_terminal_and_reward(self):
        env = Stratego(
            num_envs=32,
            continuous_chasing_rule=False,
            traj_len_per_player=100,
            full_info=True,
            verbose=False,
        )
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        while env.stats["num_finished_games"] < 1000:
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

        num_finished_games = env.stats["num_finished_games"]
        non_tie = env.stats["num_flag_capture"] + env.stats["num_wipe_out"]
        print(f"num terminated games: {num_finished_games}, non-tie terminals: {non_tie}")
        print(f"non-tie ratio: {100 * non_tie/num_finished_games:.2f}")
        assert non_tie / num_finished_games > 0.01

    def test_prior_failure(self):
        def compare_tensors(A, B):
            assert A.shape == B.shape
            if not torch.allclose(A, B):
                print(A)
                print(B)
                self.fail()

        env = Stratego(1, 100, max_num_moves=20)
        env.change_reset_behavior_to_initial_board(
            "FFDBMBCEDGDFJBBKLDJGBDIHIHHHIBEDEFGDEGDEaa__aa__aaaa__aa__aaNPTTUPUTTNRUSRWXSSSRNVQQQOQQVNYNPPPPPPNR"
        )
        env.reset()
        checker = StrategoChecker(
            env.current_board_strs[0],
            env.current_piece_ids[0],
        )
        actions = [431, 534, 339, 335, 221, 345, 449, 335, 559]
        for t in range(len(actions)):
            compare_tensors(
                env.env.get_board_tensor(env.current_step)[0, 1:1600:16].view(10, 10).cpu(),
                checker.backend_piece_ids(),
            )
            compare_tensors(
                env.env.get_board_tensor(env.current_step)[0, 1641 : 1641 + 160 : 2].cpu() & 1,
                checker.backend_is_dead(),
            )
            env.apply_actions(torch.tensor([actions[t]], dtype=torch.int32, device="cuda"))
            checker.apply_action(actions[t])

    def test_random(self):
        def compare_tensors(A, B, env=None):
            assert A.shape == B.shape
            if not torch.allclose(A, B):
                print(A)
                print(B)
                print(env)
                self.fail()

        set_seed_everywhere(0)
        num_envs = 2048
        env = Stratego(num_envs, 100, max_num_moves=20)
        # env.seed_action_sampler(42)

        num_steps = 25
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        checkers = [None for _ in range(env.num_envs)]
        for t in range(num_steps):
            for X in range(env.num_envs):
                if env.current_num_moves_since_reset[X] == 0:
                    checkers[X] = StrategoChecker(
                        env.current_board_strs[X],
                        env.current_piece_ids[X],
                    )

                compare_tensors(
                    env.env.get_board_tensor(env.current_step)[X, 1:1600:16].view(10, 10).cpu(),
                    checkers[X].backend_piece_ids(),
                    env=X,
                )
                compare_tensors(
                    env.env.get_board_tensor(env.current_step)[X, 1641 : 1641 + 160 : 2].cpu() & 1,
                    checkers[X].backend_is_dead(),
                    env=X,
                )

            env.sample_random_legal_action(action_tensor)
            for X in range(env.num_envs):
                if not env.current_is_terminal[X]:
                    checkers[X].apply_action(action_tensor[X].item())
            action_tensor.cpu().numpy().tofile(f"/tmp/ac{t}.npy")
            env.apply_actions(action_tensor)


if __name__ == "__main__":
    unittest.main()
