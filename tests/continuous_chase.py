from argparse import ArgumentParser
from enum import Enum
import os

import torch

from pyengine.core.env import Stratego
from pyengine import utils
from pyengine.utils.loading import load_rl_model

pystratego = utils.get_pystratego()


class States(Enum):
    DORMANT = "DORMANT"
    MAYBE_THREAT = "MAYBE_THREAT"
    EVADE = "EVADE"
    CHASE = "CHASE"


def is_adjacent(src: int, dst: int) -> bool:
    assert src is not None and dst is not None
    return dst in (src - 1, src + 1, src - 10, src + 10)


def pair_to_int(x: int, y: int) -> int:
    if y == x - 1:
        return x
    elif y == x + 1:
        return 100 + x
    elif y == x - 10:
        return 200 + x
    elif y == x + 10:
        return 300 + x
    else:
        raise ValueError("Invalid pair")


def compute_state(src, dst, is_chaser):
    if src is None:
        return States.DORMANT
    elif dst is None:
        return States.MAYBE_THREAT
    elif is_chaser:
        return States.EVADE
    else:
        return States.CHASE


class StateMachine:
    def __init__(self, tag):
        self.tag = tag
        self.clear()

    def update(self, src: int, dst: int, was_battle: bool, is_opp: bool) -> bool:
        if is_opp:
            needs_reset = self.evader_pos is not None and self.evader_pos != src
            needs_reset = needs_reset or (
                self.chaser_pos is not None and not is_adjacent(self.chaser_pos, src)
            )
        else:
            needs_reset = self.chaser_pos != src or (
                self.evader_pos is not None and not is_adjacent(dst, self.evader_pos)
            )
        needs_reset = needs_reset or was_battle

        if needs_reset:
            self.clear()

        ret = False
        if is_opp:
            if self.chaser_pos is not None and self.evader_pos is None:
                # An evasion begins
                self.pair_history[pair_to_int(self.chaser_pos, src)] = True
            self.evader_pos = dst
        else:
            if (
                not is_opp
                and self.evader_pos is not None
                and self.chaser_pos is not None
                and self.last_chaser_pos != dst
            ):
                ret = self.pair_history[pair_to_int(dst, self.evader_pos)]

            if self.chaser_pos is not None and self.evader_pos is not None:
                assert is_adjacent(self.evader_pos, dst)
                self.pair_history[pair_to_int(dst, self.evader_pos)] = True
            self.last_chaser_pos = self.chaser_pos
            self.chaser_pos = dst

        return ret

    def clear(self):
        self.pair_history = 400 * [False]
        self.chaser_pos = None
        self.evader_pos = None
        self.last_chaser_pos = None

    def print(self):
        def int_to_pair(z: int) -> tuple:
            if z < 100:
                return z, z - 1
            elif z < 200:
                return z - 100, z - 100 + 1
            elif z < 300:
                return z - 200, z - 200 - 10
            else:
                return z - 300, z - 300 + 10

        print("Tag:", self.tag)
        print("Pair history:", [int_to_pair(i) for i, p in enumerate(self.pair_history) if p])
        print("Chaser pos:", self.chaser_pos)
        print("Evader pos:", self.evader_pos)
        print("Last chaser pos:", self.last_chaser_pos)


class MinimalGameStateMachine:
    def __init__(self):
        self.state_machine1 = StateMachine("RED")
        self.state_machine2 = StateMachine("BLUE")

    def update(self, src, dst, was_battle, acting_player):
        return self.state_machine1.update(
            src, dst, was_battle, acting_player == 0
        ) or self.state_machine2.update(src, dst, was_battle, acting_player == 1)


def write_game(data, fn):
    with open(fn, "w") as f:
        for d in data:
            f.write(f"{str(d)}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--num_games", type=int, default=100)
    args = parser.parse_args()
    utils.set_seed_everywhere(0)
    env = Stratego(
        num_envs=1,
        traj_len_per_player=100,
        barrage=True,
        full_info=False,
        continuous_chasing_rule=False,
    )

    rl_model, arrs = load_rl_model(args.model_path)
    env.change_reset_behavior_to_random_initial_arrangement(arrs)
    violations = 0
    continuous_chase_probabilities = []
    i = 0
    while violations < args.num_games:
        i += 1
        env.reset()
        data = []
        board_hist = []
        board_str = env.current_board_strs[0][::2]
        data.append(board_str)
        sm = MinimalGameStateMachine()
        while not env.current_is_terminal:
            action_tensor, values, log_probs = rl_model(
                env.current_infostate_tensor,
                env.current_piece_ids,
                env.current_legal_action_mask,
            )
            coords = pystratego.util.actions_to_abs_coordinates(action_tensor, env.current_player)[
                0
            ]
            was_battle = (
                env.current_board_strs[0].replace("@", "").replace(".", "")[coords[1]] != "a"
            )
            data.append(action_tensor.item())
            env.apply_actions(action_tensor)
            assert len(coords) == 2
            is_violation = sm.update(coords[0], coords[1], was_battle, env.current_player)
            if is_violation:
                violations += 1
                print("Continuous chase action probability:", log_probs.exp().item())
                write_game(data, f"tests/continuous_chase_games/game{violations}.txt")
                break
        print(f"{violations} out of {i+1} games have continuous chase violations")

    print("Testing that saved games trigger violations")
    cwd = os.path.dirname(__file__)
    continuous_chase_games = os.listdir(os.path.join(cwd, "continuous_chase_games"))
    violations = {"initial_boards": [], "action_sequences": []}

    for game_file in continuous_chase_games:
        with open(os.path.join(cwd, "continuous_chase_games", game_file), "r") as f:
            game_data = f.readlines()
            init_board = game_data[0].strip()
            assert len(init_board) == 100
            violations["initial_boards"].append(init_board)
            violations["action_sequences"].append(
                [int(game_data[i]) for i in range(1, len(game_data))]
            )

    for i in range(args.num_games):
        init_board = violations["initial_boards"][i]
        action_sequence = violations["action_sequences"][i]
        env.change_reset_behavior_to_initial_board(init_board)
        env.reset()
        sm = MinimalGameStateMachine()
        action_tensor = action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for action in action_sequence:
            action_tensor[0] = action
            coords = pystratego.util.actions_to_abs_coordinates(action_tensor, env.current_player)[
                0
            ]
            was_battle = (
                env.current_board_strs[0].replace("@", "").replace(".", "")[coords[1]] != "a"
            )
            is_violation = sm.update(coords[0], coords[1], was_battle, 1 - env.current_player)
            env.apply_actions(action_tensor)
        assert is_violation, "Game did not trigger continuous chase violation"
        print(f"Game {i+1} passed")
