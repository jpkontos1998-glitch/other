import torch
import unittest
from collections import deque

import numpy as np

from pyengine.core.env import Stratego
from pyengine import utils

pystratego = utils.get_pystratego()


def move(cellfrom, cellto, player):
    moves = pystratego.util.abs_coordinates_to_actions([(cellfrom, cellto)], player=player)
    return torch.tensor(moves, dtype=torch.int32, device="cuda:0")


def twosquare_applies(env):
    env.env.compute_two_square_rule_applies(env.current_step)
    return env.env.two_square_rule_applies.item()


def print_board(env):
    print(
        "\n".join(
            [
                env.board_strs(env.current_step())[0]
                .replace("@", " ")
                .replace(".", " ")[i * 20 : (i + 1) * 20]
                for i in range(10)
            ]
        )
    )


class TwosquareTests(unittest.TestCase):
    def run_basic1(self, two_square_rule):
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=two_square_rule)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()

        self.assertFalse(twosquare_applies(env))
        env.apply_actions(move(39, 49, 0))
        self.assertEqual(env.env.get_terminated_since(env.current_step), 0)
        self.assertFalse(twosquare_applies(env))
        env.apply_actions(move(60, 50, 1))
        self.assertEqual(env.env.get_terminated_since(env.current_step), 0)
        self.assertFalse(twosquare_applies(env))
        env.apply_actions(move(49, 39, 0))
        self.assertEqual(env.env.get_terminated_since(env.current_step), 0)
        self.assertFalse(twosquare_applies(env))
        env.apply_actions(move(50, 60, 1))
        self.assertEqual(env.env.get_terminated_since(env.current_step), 0)
        self.assertFalse(twosquare_applies(env))
        env.apply_actions(move(39, 49, 0))
        self.assertEqual(env.env.get_terminated_since(env.current_step), 0)
        self.assertFalse(twosquare_applies(env))
        env.apply_actions(move(60, 50, 1))
        self.assertEqual(env.env.get_terminated_since(env.current_step), 0)
        self.assertTrue(twosquare_applies(env))
        if two_square_rule:
            # We should be able to move left and up but not down
            self.assertEqual(env.current_legal_action_mask[0, move(49, 39, 0)], False)  # Down
        else:
            self.assertEqual(env.current_legal_action_mask[0, move(49, 39, 0)], True)  # Down
        self.assertEqual(env.current_legal_action_mask[0, move(49, 59, 0)], True)  # Up
        self.assertEqual(env.current_legal_action_mask[0, move(49, 48, 0)], True)  # Left

    def run_basic2(self):
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=True)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        # Move piece into position
        env.apply_actions(move(20, 30, 0))
        env.apply_actions(move(99, 98, 1))
        env.apply_actions(move(30, 40, 0))
        env.apply_actions(move(98, 97, 1))
        env.apply_actions(move(40, 50, 0))
        env.apply_actions(move(97, 96, 1))
        # move back and forth
        env.apply_actions(move(50, 51, 0))
        env.apply_actions(move(96, 95, 1))
        env.apply_actions(move(51, 50, 0))
        env.apply_actions(move(95, 94, 1))
        env.apply_actions(move(50, 51, 0))
        env.apply_actions(move(94, 93, 1))
        self.assertFalse(env.current_legal_action_mask[0, move(51, 50, 0).item()])

    def test_basic_prune(self):
        self.run_basic1(True)
        self.run_basic2()

    def test_basic_noprune(self):
        self.run_basic1(False)

    def test_attack_move(self):
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=True)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()

        # Move piece into position
        env.apply_actions(move(20, 30, 0))
        env.apply_actions(move(99, 98, 1))
        env.apply_actions(move(30, 40, 0))
        env.apply_actions(move(98, 97, 1))
        env.apply_actions(move(40, 50, 0))
        env.apply_actions(move(97, 96, 1))
        # move back and forth
        env.apply_actions(move(50, 51, 0))
        env.apply_actions(move(96, 95, 1))
        env.apply_actions(move(51, 50, 0))
        env.apply_actions(move(95, 94, 1))
        env.apply_actions(move(50, 51, 0))
        # move opponent into position
        env.apply_actions(move(60, 50, 1))
        self.assertFalse(env.current_legal_action_mask[0, move(51, 50, 0).item()])

    def test_move_in_different_dir(self):
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=True)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        # Move piece into position
        env.apply_actions(move(20, 30, 0))
        env.apply_actions(move(99, 98, 1))
        env.apply_actions(move(30, 40, 0))
        env.apply_actions(move(98, 97, 1))
        env.apply_actions(move(40, 50, 0))
        env.apply_actions(move(97, 96, 1))
        # move back and forth
        env.apply_actions(move(50, 51, 0))
        env.apply_actions(move(96, 95, 1))
        env.apply_actions(move(51, 50, 0))
        env.apply_actions(move(95, 94, 1))
        env.apply_actions(move(50, 40, 0))
        env.apply_actions(move(94, 93, 1))
        for dest in [50, 41, 30]:
            self.assertTrue(env.current_legal_action_mask[0, move(40, dest, 0).item()])

    def test_scout_behavior_deterministic(self):
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=True)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        # Move piece into position
        env.apply_actions(move(20, 50, 0))
        env.apply_actions(move(99, 98, 1))
        env.apply_actions(move(50, 30, 0))
        env.apply_actions(move(98, 97, 1))
        env.apply_actions(move(30, 50, 0))
        env.apply_actions(move(97, 96, 1))
        for dest in [20, 30, 40]:
            self.assertFalse(env.current_legal_action_mask[0, move(50, dest, 0).item()])
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=False)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        # Move piece into position
        env.apply_actions(move(20, 50, 0))
        env.apply_actions(move(99, 98, 1))
        env.apply_actions(move(50, 30, 0))
        env.apply_actions(move(98, 97, 1))
        env.apply_actions(move(30, 50, 0))
        env.apply_actions(move(97, 96, 1))
        for dest in [20, 30, 40]:
            self.assertTrue(env.current_legal_action_mask[0, move(50, dest, 0).item()])

    def test_scout_behavior_random(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=1,
            two_square_rule=True,
            max_num_moves_between_attacks=200,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        miner_pos = 11
        opponent_move_left = True
        opp_pos = 99
        current_pos = 20
        borders_crossed = deque([set(), set(), set()], maxlen=3)
        for i in range(200):
            if i % 2 == 0:  # player 0
                places = [i for i in range(20, 30) if i != current_pos]
                legals = []
                for place in places:
                    env_legality = env.current_legal_action_mask[
                        0, move(current_pos, place, 0).item()
                    ]
                    legals.append(env_legality)
                    min_pos = min(current_pos, place)
                    max_pos = max(current_pos, place)
                    border_idx = set(range(min_pos, max_pos))
                    for borders in borders_crossed:
                        border_idx = border_idx.intersection(borders)
                    env_legality2 = len(border_idx) == 0
                    self.assertEqual(env_legality, env_legality2)
                legal_places = [places[i] for i in range(len(places)) if legals[i]]
                if len(legal_places) == 0:
                    borders_crossed.append(set())
                    if miner_pos == 11:
                        env.apply_actions(move(11, 10, 0))
                        miner_pos = 10
                    else:
                        env.apply_actions(move(10, 11, 0))
                        miner_pos = 11
                else:
                    next_pos = np.random.choice(legal_places)
                    min_pos = min(current_pos, next_pos)
                    max_pos = max(current_pos, next_pos)
                    border_idx = set(range(min_pos, max_pos))
                    borders_crossed.append(border_idx)
                    env.apply_actions(move(current_pos, next_pos, 0))
                    current_pos = next_pos
            else:  # player 1
                if (
                    opp_pos > 90
                    and opponent_move_left
                    and env.current_legal_action_mask[0, move(opp_pos, opp_pos - 1, 1).item()]
                ):
                    env.apply_actions(move(opp_pos, opp_pos - 1, 1))
                    opp_pos -= 1
                elif (
                    opp_pos > 90
                    and opponent_move_left
                    and (not env.current_legal_action_mask[0, move(opp_pos, opp_pos - 1, 1).item()])
                ):
                    opponent_move_left = False
                    env.apply_actions(move(opp_pos, opp_pos + 1, 1))
                    opp_pos += 1
                elif (
                    opp_pos < 99
                    and (not opponent_move_left)
                    and env.current_legal_action_mask[0, move(opp_pos, opp_pos + 1, 1).item()]
                ):
                    env.apply_actions(move(opp_pos, opp_pos + 1, 1))
                    opp_pos += 1
                else:
                    opponent_move_left = True
                    env.apply_actions(move(opp_pos, opp_pos - 1, 1))
                    opp_pos -= 1

    def test_termination_behavior(self):
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=True)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["DAAAAAAAAAAECAAAABLAKAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        env.apply_actions(move(0, 10, 0))
        env.apply_actions(move(60, 50, 1))
        env.apply_actions(move(39, 29, 0))
        env.apply_actions(move(50, 40, 1))
        env.apply_actions(move(10, 0, 0))
        env.apply_actions(move(40, 30, 1))
        env.apply_actions(move(0, 10, 0))
        env.apply_actions(move(79, 78, 1))
        env.apply_actions(move(10, 0, 0))
        env.apply_actions(move(30, 31, 1))
        env.apply_actions(move(0, 10, 0))

        # do same thing without termination to make sure it counts as illegal
        env = Stratego(num_envs=1, traj_len_per_player=1, two_square_rule=True)
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["DAAAAAAAAAAECAAAABLAKAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        env.apply_actions(move(0, 10, 0))
        env.apply_actions(move(60, 50, 1))
        env.apply_actions(move(39, 29, 0))
        env.apply_actions(move(50, 40, 1))
        env.apply_actions(move(10, 0, 0))
        env.apply_actions(move(40, 30, 1))
        env.apply_actions(move(0, 10, 0))
        env.apply_actions(move(79, 78, 1))
        env.apply_actions(move(10, 0, 0))
        env.apply_actions(move(78, 77, 1))
        self.assertFalse(env.current_legal_action_mask[0, move(0, 10, 0).item()])

    def test_payoff_behavior(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=1,
            two_square_rule=True,
            custom_inits=(
                ["DABAAAAAAAMBAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
                ["DBMAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
            ),
        )

        env.apply_actions(move(0, 1, 0))
        env.apply_actions(move(99, 89, 1))
        env.apply_actions(move(1, 0, 0))
        env.apply_actions(move(89, 79, 1))
        env.apply_actions(move(0, 1, 0))
        # self.assertEqual(env.current_reward_pl0, 0)
        action_tensor = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        env.sample_random_legal_action(action_tensor)
        env.apply_actions(action_tensor)
        # self.assertEqual(env.current_reward_pl0, -1)

    def test_flag(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=1,
            two_square_rule=False,
            custom_inits=(
                ["DABAAAAAAAMBAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
                ["DBMAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
            ),
        )

        env.apply_actions(move(0, 1, 0))
        env.apply_actions(move(99, 89, 1))
        env.apply_actions(move(1, 0, 0))
        env.apply_actions(move(89, 79, 1))
        env.apply_actions(move(0, 1, 0))
        env.apply_actions(move(79, 78, 1))
        self.assertTrue(twosquare_applies(env))

    def test_action_reduction(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=1,
            two_square_rule=True,
            max_num_moves_between_attacks=200,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env.reset()
        env2 = Stratego(
            num_envs=1,
            traj_len_per_player=1,
            two_square_rule=False,
            max_num_moves_between_attacks=200,
        )
        env2.change_reset_behavior_to_random_initial_arrangement(
            [
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
                ["KAAAAAALAAAECAAAABAADAAAAAAAAAAMAAAAAAAD"],
            ]
        )
        env2.reset()
        miner_pos = 11
        opponent_move_left = True
        opp_pos = 99
        current_pos = 20
        for i in range(200):
            if i % 2 == 0:  # player 0
                places = [i for i in range(20, 30) if i != current_pos]
                legals = []
                for place in places:
                    env_legality = env.current_legal_action_mask[
                        0, move(current_pos, place, 0).item()
                    ]
                    legals.append(env_legality)
                legal_places = [places[i] for i in range(len(places)) if legals[i]]
                self.assertEqual(
                    env2.current_legal_action_mask.sum() - (len(places) - len(legal_places)),
                    env.current_legal_action_mask.sum(),
                )
                if len(legal_places) == 0:
                    if miner_pos == 11:
                        env.apply_actions(move(11, 10, 0))
                        env2.apply_actions(move(11, 10, 0))
                        miner_pos = 10
                    else:
                        env.apply_actions(move(10, 11, 0))
                        env2.apply_actions(move(10, 11, 0))
                        miner_pos = 11
                else:
                    next_pos = np.random.choice(legal_places)
                    env.apply_actions(move(current_pos, next_pos, 0))
                    env2.apply_actions(move(current_pos, next_pos, 0))
                    current_pos = next_pos
            else:  # player 1
                if (
                    opp_pos > 90
                    and opponent_move_left
                    and env.current_legal_action_mask[0, move(opp_pos, opp_pos - 1, 1).item()]
                ):
                    env.apply_actions(move(opp_pos, opp_pos - 1, 1))
                    env2.apply_actions(move(opp_pos, opp_pos - 1, 1))
                    opp_pos -= 1
                elif (
                    opp_pos > 90
                    and opponent_move_left
                    and (not env.current_legal_action_mask[0, move(opp_pos, opp_pos - 1, 1).item()])
                ):
                    opponent_move_left = False
                    env.apply_actions(move(opp_pos, opp_pos + 1, 1))
                    env2.apply_actions(move(opp_pos, opp_pos + 1, 1))
                    opp_pos += 1
                elif (
                    opp_pos < 99
                    and (not opponent_move_left)
                    and env.current_legal_action_mask[0, move(opp_pos, opp_pos + 1, 1).item()]
                ):
                    env.apply_actions(move(opp_pos, opp_pos + 1, 1))
                    env2.apply_actions(move(opp_pos, opp_pos + 1, 1))
                    opp_pos += 1
                else:
                    opponent_move_left = True
                    env.apply_actions(move(opp_pos, opp_pos - 1, 1))
                    env2.apply_actions(move(opp_pos, opp_pos - 1, 1))
                    opp_pos -= 1

    def test_snapshot(self):
        env = Stratego(
            num_envs=8,
            traj_len_per_player=1,
            two_square_rule=True,
            max_num_moves_between_attacks=200,
        )
        two_square_states = []
        snapshot_states = []
        action_tensor = torch.zeros(env.num_envs, device="cuda:0", dtype=torch.int32)
        # Generate sequence of states and two square state machines
        for i in range(512):
            # The call to .cpu() is VERY important because the tensor is only
            # stealing a pointer from the underlying machine.
            machine = env.env.get_twosquare_state(env.current_step)
            two_square_states.append((machine[0].cpu(), machine[1].cpu()))

            snapshot_states.append(env.snapshot_state(env.current_step))
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
        # Test that reloading the states produces the correct twosquare state machines
        for ts, state in zip(two_square_states, snapshot_states):
            if state.terminated_since.max() == 0:
                env.change_reset_behavior_to_env_state(state)
                env.reset()
                ts_state = (
                    env.env.get_twosquare_state(env.current_step)[0].cpu(),
                    env.env.get_twosquare_state(env.current_step)[1].cpu(),
                )
                for p in range(2):
                    terminated = env.env.get_terminated_since(0).cpu()
                    for i in range(env.num_envs):
                        if terminated[i] == 0:
                            self.assertTrue(torch.allclose(ts[p][i], ts_state[p][i]))


if __name__ == "__main__":
    unittest.main()
