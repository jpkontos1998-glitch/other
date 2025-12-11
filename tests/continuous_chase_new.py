from argparse import ArgumentParser
from enum import Enum
from pathlib import Path


from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego, set_seed_everywhere
from pyengine.utils.loading import load_rl_model

pystratego = get_pystratego()


class State(Enum):
    DORMANT = "DORMANT"
    THREATENED = "THREATENED"
    EVADING = "EVADING"
    CHASED = "CHASED"


class PlayerStateMachine:
    def __init__(self):
        """The state machine is shown below:
        + -> +---+ ---> +---+      +---+ ---> +---+
        |    | D |      | T | ---> | E |      | C |
        + -- +---+ <--- +---+      +---+ <--- +---+
                ^   ^                   |          |
                |   |                   |          |
                |   --------------------+          |
                -----------------------------------+
        """
        self.state = State.DORMANT
        self.chase_len = 0

    def register_threat(self, active: bool):
        assert self.state in [State.DORMANT, State.EVADING]
        if not active:
            self.chase_len = 0
            self.state = State.DORMANT
        else:
            self.chase_len += 1
            if self.state == State.DORMANT:
                self.state = State.THREATENED
            else:
                self.state = State.CHASED

    def register_evade(self, active: bool):
        assert self.state in [State.DORMANT, State.THREATENED, State.CHASED]
        if self.state == State.DORMANT or not active:
            self.chase_len = 0
            self.state = State.DORMANT
        else:
            self.chase_len += 1
            self.state = State.EVADING


LETTERS = (
    ["c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "b"],
    ["o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "n"],
)


class GameStateMachine:
    def __init__(self, initial_board_str: str):
        self.reset(initial_board_str)

    def step(self, board_str, action_abs_coords) -> bool:
        # Assert src has correct owner
        assert self.current_board_str[action_abs_coords[0]] in LETTERS[self.current_player]

        # Log piece-to-move
        last_moved = self.current_position[action_abs_coords[0]]

        # Compute new position
        current_position = list(self.current_position)
        current_position[action_abs_coords[0]] = -1  # Source cell necessarily becomes empty
        if self.current_board_str[action_abs_coords[1]] == "a":  # no battle
            current_position[action_abs_coords[1]] = self.current_position[
                action_abs_coords[0]
            ]  # Destination cell value assumes source cell value
        else:  # Can reset after battle b/c previous board states can't be reached
            self.reset(board_str, 1 - self.current_player)
            return False

        # Update states
        # ---
        # This method updates each player's continuous-chase state.
        self.update(action_abs_coords)

        # Check continuous chase is occurring
        # ---
        # We track continuous chases by the pieces being chased, so if there
        # is a continuous chase, it will be registered in the opponents's state.
        if self.states[1 - self.current_player].state == State.CHASED:
            # Check if board state has been visited during chase
            lower_index = len(self.past_positions) - self.states[1 - self.current_player].chase_len
            if tuple(current_position) in self.past_positions[lower_index:]:
                # Check if piece is returning to previous position (always allowed)
                # ---
                # -1 is board state before our current move
                # -2 is board state before opponent's last move
                # -3 is board state before our previous move
                if (
                    current_position[action_abs_coords[1]]
                    != self.past_positions[-3][action_abs_coords[1]]
                ):
                    return True

        # Filter movement and visibility bits
        board_str = board_str[::2].lower()
        # Assert difference makes sense
        assert sum(s_ != s for s_, s in zip(self.current_board_str, board_str)) == 2

        # Update fields
        self.current_board_str = board_str
        self.current_position = tuple(current_position)
        self.past_positions.append(self.current_position)
        self.check_valid()
        self.current_player = 1 - self.current_player
        self.last_moved = last_moved

        return False

    def reset(self, board_str, current_player=0):
        self.states = [PlayerStateMachine(), PlayerStateMachine()]
        # Filter movement and visibility bits
        board_str = board_str[::2].lower()
        self.current_board_str = board_str
        # Assign each piece a unique index
        i = 0
        pos = []
        for s in self.current_board_str:
            if s == "a":
                pos.append(-1)
            elif s == "_":
                pos.append(-2)
            else:
                pos.append(i)
                i += 1
        self.current_position = tuple(pos)
        self.past_positions = [self.current_position]
        self.check_valid()
        self.current_player = current_player
        self.last_moved = None

    def check_valid(self):
        other = []
        for s, v in zip(self.current_board_str, self.current_position):
            if s == "a":
                assert v == -1
            elif s == "_":
                assert v == -2
            else:
                other.append(v)
        assert len(set(other)) == len(other)

    def update(self, coords):
        src, dst = coords

        # Evasion
        # ---
        # The piece-to-move is evading if the opponent's last-moved piece threatened it.
        # In other words, this last-moved piece is adjacent to the source cell.
        adjacent_positions = (src - 1, src + 1, src - 10, src + 10)
        adjacent_positions = [
            adj_pos for adj_pos in adjacent_positions if adj_pos >= 0 and adj_pos < 100
        ]
        is_evading = False
        for adj_pos in adjacent_positions:
            if self.last_moved == self.current_position[adj_pos]:
                assert adj_pos != dst, "This shouldn't be reachable"
                is_evading = True
        self.states[self.current_player].register_evade(is_evading)

        # Threatening
        # ---
        # Any opponent piece adjacent to the destination cell is under threat.
        adjacent_positions = (dst - 1, dst + 1, dst - 10, dst + 10)
        adjacent_positions = [
            adj_pos for adj_pos in adjacent_positions if adj_pos >= 0 and adj_pos < 100
        ]
        is_threatening = False
        for adj_pos in adjacent_positions:
            if (
                self.current_board_str[adj_pos] in LETTERS[1 - self.current_player]
            ):  # adj_pos is threatened
                is_threatening = True
        self.states[1 - self.current_player].register_threat(is_threatening)


def write_game(data, fn):
    with open(fn, "w") as f:
        for d in data:
            f.write(f"{str(d)}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--save_dir")
    args = parser.parse_args()
    set_seed_everywhere(0)
    env = Stratego(
        num_envs=1, traj_len_per_player=100, two_square_rule=True, continuous_chasing_rule=True
    )
    sm = GameStateMachine(env.current_board_strs[0])

    rl_model, arrs = load_rl_model(args.model_path)
    env.change_reset_behavior_to_random_initial_arrangement(arrs)
    violations = 0
    counter = 0
    continuous_chase_probabilities = []
    while violations < args.num_games:
        counter += 1
        env.reset()
        data = []
        board_hist = []
        board_str = env.current_board_strs[0][::2]
        data.append(board_str)
        sm.reset(env.current_board_strs[0])
        while not env.current_is_terminal:
            action_tensor, values, log_probs = rl_model(
                env.current_infostate_tensor,
                env.current_piece_ids,
                env.current_legal_action_mask,
            )
            coords = pystratego.util.actions_to_abs_coordinates(action_tensor, env.current_player)[
                0
            ]
            data.append(action_tensor.item())
            env.apply_actions(action_tensor)
            assert len(coords) == 2
            board_hist.append((env.current_board_strs_pretty[0], coords))
            is_violation = sm.step(env.current_board_strs[0], coords)
            if is_violation:
                for j, (b, move) in enumerate(board_hist):
                    print(f"Move {j}: {move}")
                    print(b.replace("a", "."))
                write_game(
                    data,
                    f"{Path(__file__).resolve().parent}/continuous_chase_games_new/game{violations}.txt",
                )
                violations += 1
                break
            assert sm.current_board_str == env.current_board_strs[0][::2].lower()
        print(f"{violations} out of {counter+1} games have continuous chase violations")
        print("Violation probabilities", continuous_chase_probabilities)
