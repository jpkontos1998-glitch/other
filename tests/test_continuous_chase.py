import os
import unittest
from copy import deepcopy
from random import uniform

import torch

from pyengine.core.env import Stratego
from pyengine import utils

from continuous_chase import MinimalGameStateMachine


pystratego = utils.get_pystratego()


cwd = os.path.dirname(__file__)
continuous_chase_games = sorted(os.listdir(os.path.join(cwd, "continuous_chase_games")))
violations = {"initial_boards": [], "action_sequences": [], "fn": []}

for game_file in continuous_chase_games[
    :10
]:  # Only load the first 10 games to make the test faster
    with open(os.path.join(cwd, "continuous_chase_games", game_file), "r") as f:
        game_data = f.readlines()
        init_board = game_data[0].strip()
        assert len(init_board) == 100
        violations["initial_boards"].append(init_board)
        violations["action_sequences"].append([int(game_data[i]) for i in range(1, len(game_data))])
        violations["fn"].append(game_file)


def check_actions(legal_action_mask1, legal_action_mask2, sm, board_string, current_player):
    assert len(legal_action_mask1.shape) == len(legal_action_mask2.shape) == 1
    action_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
    eq = legal_action_mask1 == legal_action_mask2
    if eq.all():
        return True
    for i, b in enumerate(eq):
        if b:
            continue
        sm_ = deepcopy(sm)
        action_tensor[0] = i
        coords = pystratego.util.actions_to_abs_coordinates(action_tensor, current_player)[0]
        was_battle = board_string.replace("@", "").replace(".", "")[coords[1]] != "a"
        to_be_violation = sm_.update(coords[0], coords[1], was_battle, 1 - current_player)
        if not to_be_violation:
            return False
    return True


class ContinuousChaseTest(unittest.TestCase):
    def test_continuous_chase(self):
        for two_square in [True, False]:
            for chase_on in [True, False]:
                num_envs = 1
                env = Stratego(
                    num_envs=num_envs,
                    traj_len_per_player=100,
                    barrage=True,
                    full_info=False,
                    continuous_chasing_rule=chase_on,
                    two_square_rule=two_square,
                    max_num_moves_between_attacks=200,
                )
                clean_env = Stratego(
                    num_envs=num_envs,
                    traj_len_per_player=100,
                    barrage=True,
                    full_info=False,
                    continuous_chasing_rule=False,
                    two_square_rule=two_square,
                    max_num_moves_between_attacks=200,
                )
                for i, (init_board, action_seq, fn) in enumerate(
                    zip(
                        violations["initial_boards"],
                        violations["action_sequences"],
                        violations["fn"],
                    )
                ):  # Only use the first 10 games to make the test faster
                    env.change_reset_behavior_to_initial_board(init_board)
                    clean_env.change_reset_behavior_to_initial_board(init_board)
                    env.reset()
                    clean_env.reset()
                    sm = MinimalGameStateMachine()
                    action_tensor = torch.zeros(env.num_envs, device="cuda", dtype=torch.int32)
                    for j, a in enumerate(action_seq):
                        action_tensor[:] = a
                        coords = pystratego.util.actions_to_abs_coordinates(
                            action_tensor, env.current_player
                        )[0]
                        was_battle = (
                            env.current_board_strs[0].replace("@", "").replace(".", "")[coords[1]]
                            != "a"
                        )
                        agrees = check_actions(
                            env.current_legal_action_mask[0],
                            clean_env.current_legal_action_mask[0],
                            deepcopy(sm),
                            env.current_board_strs[0],
                            env.current_player,
                        )
                        if not agrees:
                            print(j)
                            raise
                        self.assertTrue(agrees)
                        to_be_violation = sm.update(
                            coords[0], coords[1], was_battle, 1 - env.current_player
                        )
                        if env.current_num_moves_since_last_attack > 200:
                            break
                        if to_be_violation:
                            if chase_on:
                                self.assertFalse(
                                    env.current_legal_action_mask.cpu().numpy()[:, a].any()
                                )
                                break
                            else:
                                self.assertTrue(
                                    torch.allclose(
                                        env.current_legal_action_mask,
                                        clean_env.current_legal_action_mask,
                                    )
                                )
                            break
                        else:
                            self.assertTrue(env.current_legal_action_mask.cpu().numpy()[:, a].all())
                            self.assertTrue(
                                clean_env.current_legal_action_mask.cpu().numpy()[:, a].all()
                            )
                            env.apply_actions(action_tensor)
                            clean_env.apply_actions(action_tensor)
                    else:
                        assert False, "Should not reach here"

    def test_long_chase(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            continuous_chasing_rule=True,
            max_num_moves_between_attacks=200,
            two_square_rule=False,
        )
        env.change_reset_behavior_to_initial_board(
            "CMaaaaaaaa"
            + "KLaaDaaaaa"
            + "EBaaaaaaaa"
            + "Daaaaaaaaa"
            + "aa__aa__aa"
            + "aa__aa__aa"
            + "aaaaPaaaaa"
            + "aaaaaaaaaa"
            + "aaaWaaaaPO"
            + "QaaaaaXaNY"
        )
        env.reset()

        def my_apply_action(abs_from, abs_to):
            action_tensor = torch.tensor(
                pystratego.util.abs_coordinates_to_actions(
                    [(abs_from, abs_to)], env.current_player
                ),
                device="cuda",
                dtype=torch.int32,
            )
            env.apply_actions(action_tensor)

        my_apply_action(30, 80)
        for _ in range(50):
            my_apply_action(90, 91)
            my_apply_action(80, 81)
            my_apply_action(91, 90)
            my_apply_action(81, 80)
        self.assertTrue(env.current_is_terminal)
        my_apply_action(99, 89)

    # def test_snapshot_past(self):
    """This test 'passed' but is disabled because I don't know how to catch a core dump in python"""
    #     for two_square in [True, False]:
    #         env = Stratego(num_envs=1, traj_len_per_player=100, barrage=True, full_info=False, two_square_rule=two_square)
    #         for init_board, action_seq in zip(
    #             violations["initial_boards"], violations["action_sequences"]
    #         ):
    #             env.change_reset_behavior_to_initial_board(init_board)
    #             env.reset()
    #             sm = MinimalGameStateMachine()
    #             action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
    #             for a in action_seq:
    #                 action_tensor[0] = a
    #                 coords = pystratego.util.actions_to_abs_coordinates(
    #                     action_tensor, env.current_player
    #                 )[0]
    #                 was_battle = (
    #                     env.current_board_strs[0].replace("@", "").replace(".", "")[coords[1]] != "a"
    #                 )
    #                 assert len(coords) == 2
    #                 to_be_violation = sm.update(
    #                     coords[0], coords[1], was_battle, 1 - env.current_player
    #                 )
    #                 if to_be_violation:
    #                     self.assertFalse(env.current_legal_action_mask[0].cpu().numpy()[a])
    #                     env_state = env.snapshot_state(env.current_step - 1)
    #                     with self.assertRaises(AssertionError):
    #                         env.change_reset_behavior_to_env_state(env_state)
    #                     break
    #                 else:
    #                     self.assertTrue(env.current_legal_action_mask[0].cpu().numpy()[a])
    #                     env.apply_actions(action_tensor)

    def test_change_reset_behavior(self):
        for two_square in [True, False]:
            num_envs = 2
            env = Stratego(
                num_envs=num_envs,
                traj_len_per_player=100,
                barrage=True,
                full_info=False,
                two_square_rule=two_square,
                max_num_moves_between_attacks=200,
            )
            counter = 0
            init_board = violations["initial_boards"][counter]
            env.change_reset_behavior_to_initial_board(init_board)
            env.reset()
            action_seq = violations["action_sequences"][counter]
            while counter < len(
                violations["initial_boards"][:10]
            ):  # Only use the first 10 games to make the test faster
                self.assertTrue(torch.all(env.current_num_moves == 0))
                action_seq = violations["action_sequences"][counter]
                if counter < len(violations["initial_boards"]) - 1:
                    next_init_board = violations["initial_boards"][counter + 1]
                    env.change_reset_behavior_to_initial_board(next_init_board)
                sm = MinimalGameStateMachine()
                action_tensor = torch.zeros(env.num_envs, device="cuda", dtype=torch.int32)
                for i, a in enumerate(action_seq):
                    action_tensor[:] = a
                    coords = pystratego.util.actions_to_abs_coordinates(
                        action_tensor, env.current_player
                    )[0]
                    was_battle = (
                        env.current_board_strs[0].replace("@", "").replace(".", "")[coords[1]]
                        != "a"
                    )
                    to_be_violation = sm.update(
                        coords[0], coords[1], was_battle, 1 - env.current_player
                    )
                    if (env.current_num_moves_since_last_attack > 200).any():
                        while not env.current_is_terminal[0]:
                            env.sample_random_legal_action(action_tensor)
                            action_tensor[:] = action_tensor[0]
                            env.apply_actions(action_tensor)
                        while env.current_is_terminal[0]:
                            env.sample_random_legal_action(action_tensor)
                            action_tensor[:] = action_tensor[0]
                            env.apply_actions(action_tensor)
                        counter += 1
                        break
                    if i == len(action_seq) - 1:
                        self.assertTrue(to_be_violation)
                    if to_be_violation:
                        self.assertFalse(env.current_legal_action_mask.cpu().numpy()[:, a].any())
                        while not env.current_is_terminal[0]:
                            env.sample_random_legal_action(action_tensor)
                            action_tensor[:] = action_tensor[0]
                            env.apply_actions(action_tensor)
                        while env.current_is_terminal[0]:
                            env.sample_random_legal_action(action_tensor)
                            action_tensor[:] = action_tensor[0]
                            env.apply_actions(action_tensor)
                        counter += 1
                        break
                    else:
                        self.assertTrue(env.current_legal_action_mask.cpu().numpy()[:, a].all())
                        env.apply_actions(action_tensor)
                else:
                    assert False, "Should not reach here"

    def test_change_state(self):
        for two_square in [True, False]:
            env = Stratego(
                num_envs=1,
                traj_len_per_player=100,
                barrage=True,
                full_info=False,
                two_square_rule=two_square,
                max_num_moves_between_attacks=200,
            )
            other_env = Stratego(
                num_envs=2,
                traj_len_per_player=100,
                barrage=True,
                full_info=False,
                two_square_rule=two_square,
                max_num_moves_between_attacks=200,
            )
            for i in range(min(len(violations["initial_boards"]), 10)):
                init_board = violations["initial_boards"][i]
                action_seq = violations["action_sequences"][i]
                env.change_reset_behavior_to_initial_board(init_board)
                env.reset()
                action_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
                other_action_tensor = torch.zeros(2, device="cuda", dtype=torch.int32)
                sm = MinimalGameStateMachine()
                for j, a in enumerate(action_seq[:-1]):
                    action_tensor[:] = a
                    coords = pystratego.util.actions_to_abs_coordinates(
                        action_tensor, env.current_player
                    )[0]
                    was_battle = (
                        env.current_board_strs[0].replace("@", "").replace(".", "")[coords[1]]
                        != "a"
                    )
                    to_be_violation = sm.update(
                        coords[0], coords[1], was_battle, 1 - env.current_player
                    )
                    if (env.current_num_moves_since_last_attack >= 200).any():
                        break
                    env.apply_actions(action_tensor)
                    env_state = env.current_state
                    env_state.tile(2)
                    other_env.change_reset_behavior_to_env_state(env_state)

                    sm_ = deepcopy(sm)
                    # Test randomly roughly once a game
                    if uniform(0, 1) > 1 / len(action_seq):
                        continue
                    for k, a_ in list(enumerate(action_seq))[j + 1 :]:
                        other_action_tensor[:] = a_
                        coords = pystratego.util.actions_to_abs_coordinates(
                            other_action_tensor, other_env.current_player
                        )[0]
                        was_battle = (
                            other_env.current_board_strs[0]
                            .replace("@", "")
                            .replace(".", "")[coords[1]]
                            != "a"
                        )
                        to_be_violation = sm_.update(
                            coords[0], coords[1], was_battle, 1 - other_env.current_player
                        )
                        if (other_env.current_num_moves_since_last_attack > 200).any():
                            break
                        if to_be_violation:
                            self.assertTrue(k == len(action_seq) - 1)
                            self.assertFalse(
                                other_env.current_legal_action_mask.cpu().numpy()[:, a_].any()
                            )
                            break
                        other_env.apply_actions(other_action_tensor)

    def test_two_illegal_actions(self):
        for chasing in [True, False]:
            env = Stratego(
                num_envs=1,
                traj_len_per_player=100,
                barrage=True,
                full_info=True,
                continuous_chasing_rule=chasing,
                max_num_moves_between_attacks=200,
            )
            env.change_reset_behavior_to_initial_board(
                "CMaaaaaaaa"
                + "KLaaDaaaaa"
                + "EBaaaaaaaa"
                + "Daaaaaaaaa"
                + "aa__aa__aa"
                + "aa__aa__aa"
                + "aaaaPaaaaa"
                + "aaaaaaaaaa"
                + "aaaWaaaaPO"
                + "QaaaaaXaNY"
            )
            env.reset()

            def my_apply_action(abs_from, abs_to):
                action_tensor = torch.tensor(
                    pystratego.util.abs_coordinates_to_actions(
                        [(abs_from, abs_to)], env.current_player
                    ),
                    device="cuda",
                    dtype=torch.int32,
                )
                env.apply_actions(action_tensor)

            my_apply_action(30, 40)
            my_apply_action(64, 34)
            my_apply_action(40, 50)
            my_apply_action(34, 32)
            my_apply_action(50, 60)
            my_apply_action(32, 12)
            my_apply_action(60, 70)
            my_apply_action(12, 13)
            my_apply_action(14, 15)
            my_apply_action(13, 14)
            my_apply_action(15, 25)
            my_apply_action(14, 15)
            my_apply_action(25, 24)
            my_apply_action(15, 25)
            my_apply_action(24, 14)
            my_apply_action(25, 24)
            my_apply_action(14, 4)
            my_apply_action(24, 14)
            my_apply_action(4, 5)
            my_apply_action(14, 4)
            my_apply_action(5, 6)
            my_apply_action(4, 5)
            my_apply_action(6, 16)
            my_apply_action(5, 6)
            my_apply_action(16, 26)
            my_apply_action(6, 16)
            my_apply_action(26, 36)
            my_apply_action(16, 26)
            my_apply_action(36, 35)
            my_apply_action(26, 36)
            my_apply_action(35, 34)
            my_apply_action(36, 35)
            my_apply_action(34, 33)
            my_apply_action(35, 34)
            my_apply_action(33, 23)
            my_apply_action(34, 33)
            my_apply_action(23, 13)
            my_apply_action(33, 23)
            my_apply_action(13, 14)
            env.current_legal_action_mask
            down = torch.tensor(
                pystratego.util.abs_coordinates_to_actions([(23, 13)], env.current_player),
                device="cuda",
                dtype=torch.int32,
            )
            across = torch.tensor(
                pystratego.util.abs_coordinates_to_actions([(23, 24)], env.current_player),
                device="cuda",
                dtype=torch.int32,
            )
            if chasing:
                self.assertFalse(env.current_legal_action_mask[0].cpu().numpy()[down])
                self.assertFalse(env.current_legal_action_mask[0].cpu().numpy()[across])
            else:
                self.assertTrue(env.current_legal_action_mask[0].cpu().numpy()[down])
                self.assertTrue(env.current_legal_action_mask[0].cpu().numpy()[across])

    def test_parallel(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            max_num_moves_between_attacks=200,
        )
        state = None
        for init_board in violations["initial_boards"]:
            env.change_reset_behavior_to_initial_board(init_board)
            env.reset()
            if state is None:
                state = env.current_state
            else:
                state = state.cat(env.current_state)
        env = Stratego(
            num_envs=state.num_envs,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            max_num_moves_between_attacks=200,
        )
        env.change_reset_behavior_to_env_state(state)
        action_tensor = torch.zeros(state.num_envs, device="cuda", dtype=torch.int32)
        for i in range(env.conf.max_num_moves):
            env.sample_random_legal_action(action_tensor)
            if (env.current_num_moves_since_last_attack > 200).any():
                break
            for j, action_seq in enumerate(violations["action_sequences"]):
                if i > len(action_seq) - 1:
                    continue
                a = action_seq[i]
                if i < len(action_seq) - 1:
                    action_tensor[j] = a
                    self.assertTrue(env.current_legal_action_mask.cpu().numpy()[j, a])
                if i == len(action_seq) - 1:
                    self.assertFalse(env.current_legal_action_mask.cpu().numpy()[j, a])
            env.apply_actions(action_tensor)

    def test_edge_case1(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            max_num_moves_between_attacks=200,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            (
                ["MEAAAAAAAABAAAAAAAAAAAAAAAAAAALDAAKCAADA"],
                ["DAAMAABAAAAAAAAAAAAAAKAAAEAAAAAAAADAAALC"],
            )
        )
        env.reset()
        actions = [
            335,
            338,
            334,
            448,
            1431,
            558,
            930,
            225,
            444,
            339,
            554,
            434,
            664,
            449,
            774,
            559,
            1284,
            669,
            883,
            554,
            1738,
            1264,
            839,
            1779,
            1001,
            778,
            1002,
            1688,
            1001,
            1687,
            2,
            888,
            1012,
            1698,
            11,
            897,
            1,
            887,
            11,
            897,
        ]
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        for a in actions:
            action_tensor[:] = a
            env.apply_actions(action_tensor)
            self.assertTrue(env.current_legal_action_mask.any())

    def test_edge_case2(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            max_num_moves_between_attacks=200,
        )
        env.change_reset_behavior_to_random_initial_arrangement(
            (
                ["AAAAAAAAEMAAAAAAAAALAAAAAAAAAAABAACKAADD"],
                ["AAAAAAADDMAAAAAACKLBAAAAAAAAAEAAAAAAAAAA"],
            )
        )
        env.reset()
        actions = [
            334,
            229,
            335,
            1416,
            445,
            115,
            444,
            1517,
            555,
            116,
            1354,
            226,
            1355,
            1739,
            1354,
            338,
            455,
            1436,
            1465,
            1325,
            445,
            1507,
            1566,
            448,
            667,
            558,
            1677,
            1668,
            678,
            667,
            1668,
            1577,
            455,
            1476,
            1345,
            775,
            667,
            1385,
            1344,
            1284,
            345,
            335,
            777,
            445,
            1587,
            555,
            1335,
            1183,
            1586,
            1365,
            787,
            1264,
            677,
            663,
            1567,
            1173,
            1566,
            906,
            667,
            1300,
            777,
            882,
            1587,
            705,
            886,
            1072,
            1338,
            1071,
            134,
            1072,
            1496,
            971,
            739,
            892,
            119,
            1082,
            129,
            881,
            1719,
            224,
            1395,
            334,
            1294,
            1618,
            18,
            444,
            1489,
            554,
            385,
            17,
            893,
            7,
            1283,
            1517,
            884,
            16,
            894,
            6,
            884,
            16,
            1394,
            1406,
            895,
            5,
            785,
            115,
            1475,
            1325,
            776,
            124,
            786,
            114,
            776,
            224,
            935,
            234,
            886,
            124,
            8,
            14,
            1596,
            1204,
            897,
            3,
            787,
            113,
            777,
            123,
            887,
            1213,
            1597,
            14,
            896,
            4,
            786,
            114,
            776,
            124,
            1586,
            1214,
            787,
            113,
            1677,
            1123,
            778,
            1022,
            1688,
            221,
            787,
            1031,
            1577,
            1132,
            1476,
            1233,
            775,
            808,
            18,
            1398,
            1385,
            694,
            1330,
            334,
            8,
            444,
            1618,
            554,
            117,
            664,
            127,
            1374,
            1517,
            1375,
            116,
            774,
            126,
            784,
            116,
            774,
            1284,
            1384,
            126,
            785,
            1416,
            1475,
            1315,
            676,
            114,
            1566,
            1224,
            1667,
            1123,
            568,
            222,
            458,
            1032,
            348,
            1183,
            238,
            231,
            128,
            882,
            18,
            892,
            8,
            882,
            18,
            1192,
            1608,
            893,
            7,
            883,
            17,
            893,
            7,
            783,
            117,
            773,
            127,
            883,
            1517,
            1193,
            1516,
            1092,
            17,
            891,
            1607,
            1081,
            1608,
            1082,
            1607,
            1081,
        ]
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        for a in actions:
            action_tensor[:] = a
            env.apply_actions(action_tensor)
            self.assertTrue(env.current_legal_action_mask.any())

    def test_attack_edge_case(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=100,
            barrage=True,
            full_info=True,
            max_num_moves_between_attacks=200,
        )
        with open(os.path.join(cwd, "strategus_games/attack_chase.txt"), "r") as f:
            game_data = f.readlines()
            initial_board = game_data[0].strip()
            assert len(init_board) == 100
            action_sequence = [int(game_data[i]) for i in range(1, 377)]
        env.change_reset_behavior_to_initial_board(initial_board)
        env.reset()
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        for a in action_sequence:
            action_tensor[:] = a
            env.apply_actions(action_tensor)
        illegal_action = pystratego.util.abs_coordinates_to_actions([(64, 65)], env.current_player)[
            0
        ]
        self.assertFalse(env.current_legal_action_mask[0, illegal_action])


if __name__ == "__main__":
    unittest.main()
