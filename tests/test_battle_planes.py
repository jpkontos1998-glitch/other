import unittest

import torch

from pyengine.core.env import Stratego
from pyengine import utils


pystratego = utils.get_pystratego()

letters_pl0 = ["c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "b"]
letters_pl1 = ["o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "n"]
flag_or_bomb_letters = ["m", "b", "y", "n"]
letter_to_index = {
    **{c: i for i, c in enumerate(letters_pl0)},
    **{c: i for i, c in enumerate(letters_pl1)},
}


def is_attacker_wins(attacker, defender):
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    if atk_val == 0 and def_val == 9:  # spy beats marshal
        return True
    elif atk_val == 2 and def_val == 11:  # miner beats bomb
        return True
    return atk_val > def_val


def is_defender_wins(attacker, defender):
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    if atk_val == 0 and def_val == 9:  # spy beats marshal
        return False
    elif atk_val == 2 and def_val == 11:  # miner beats bomb
        return False
    return def_val > atk_val


def is_attacker_dies(attacker, defender):
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    if def_val == 10:  # flag
        return False
    elif (atk_val == 0) and (def_val == 9):  # spy takes marsh
        return False
    elif (atk_val == 2) and (def_val == 11):  # minter takes bomb
        return False
    return atk_val <= def_val


def is_defender_dies(attacker, defender):
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    if def_val == 10:  # flag
        return True
    elif (atk_val == 0) and (def_val == 9):  # spy takes marsh
        return True
    elif (atk_val == 2) and (def_val == 11):  # minter takes bomb
        return True
    return def_val <= atk_val


def is_tie(attacker, defender):
    atk_val = letter_to_index[attacker.lower()]
    def_val = letter_to_index[defender.lower()]
    return atk_val == def_val


def is_attacked_visible_stronger(board_str, coords):
    board_str = board_str[0][::2]
    src, dst = coords[0]
    if board_str[dst] == "a":  # not battle
        return False
    if board_str[dst].isupper():  # defender hidden
        return False
    return is_defender_wins(board_str[src], board_str[dst])


def is_attacked_visible_tie(board_str, coords):
    board_str = board_str[0][::2]
    src, dst = coords[0]
    if board_str[dst] == "a":  # not battle
        return False
    if board_str[dst].isupper():  # defender hidden
        return False
    return is_tie(board_str[src], board_str[dst])


def is_attacked_hidden(board_str, coords):
    board_str = board_str[0][::2]
    src, dst = coords[0]
    if board_str[dst] == "a":  # not battle
        return False
    if board_str[dst].islower():  # defender visible
        return False
    return is_attacker_dies(board_str[src], board_str[dst])


def is_defended_visible_weaker(board_str, coords):
    board_str = board_str[0][::2]
    src, dst = coords[0]
    if board_str[dst] == "a":  # not battle
        return False
    if board_str[dst].isupper():  # defender hidden
        return False
    return is_attacker_wins(board_str[src], board_str[dst])


def is_defended_visible_tie(board_str, coords):
    board_str = board_str[0][::2]
    src, dst = coords[0]
    if board_str[dst] == "a":  # not battle
        return False
    if board_str[dst].isupper():  # defender hidden
        return False
    return is_tie(board_str[src], board_str[dst])


def is_defended_hidden(board_str, coords):
    board_str = board_str[0][::2]
    src, dst = coords[0]
    if board_str[dst] == "a":  # not battle
        return False
    if board_str[dst].islower():  # defender visible
        return False
    return is_defender_dies(board_str[src], board_str[dst])


class BattleTest(unittest.TestCase):
    def test_our_attacked_visible_stronger(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=10,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_attacked_visible_stronger_spy")
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index(
                "our_deathstatus_attacked_visible_stronger_marshal"
            )
            + 1
        )
        self.assertEqual(s, 131)
        self.assertEqual(e, 141)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_attacked_visible_stronger(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][0]].lower()]
                    if env.current_player == 0:
                        pl0_planes[val, dst] = 1
                    else:
                        pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    for p in range(10):
                        if not torch.allclose(planes[p], pl0_planes[p]):
                            print(
                                f"=== Acting player 0 at cell {99-coords[0][1]}: Mismatch for piece type {p}"
                            )
                            print(planes[p].view(-1, 10, 10))
                            print(pl0_planes[p].view(-1, 10, 10))
                            self.fail()
                else:
                    for p in range(10):
                        if not torch.allclose(planes[p], pl1_planes[p]):
                            print(
                                f"=== Acting player 1 at cell {coords[0][1]}: Mismatch for piece type {p}"
                            )
                            print(planes[p].view(-1, 10, 10))
                            print(pl1_planes[p].view(-1, 10, 10))
                            self.fail()

    def test_our_attacked_visible_tie(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=10,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_attacked_visible_tie_spy")
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_attacked_visible_tie_marshal")
            + 1
        )
        self.assertEqual(s, 141)
        self.assertEqual(e, 151)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_attacked_visible_tie(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][0]].lower()]
                    if env.current_player == 0:
                        pl0_planes[val, dst] = 1
                    else:
                        pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())

    def test_our_attacked_hidden(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_attacked_hidden_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_attacked_hidden_marshal") + 1
        self.assertEqual(s, 151)
        self.assertEqual(e, 161)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_attacked_hidden(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][0]].lower()]
                    if env.current_player == 0:
                        pl0_planes[val, dst] = 1
                    else:
                        pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())

    def test_our_defended_visible_weaker(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_visible_defended_weaker_spy")
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index(
                "our_deathstatus_visible_defended_weaker_marshal"
            )
            + 1
        )
        self.assertEqual(s, 161)
        self.assertEqual(e, 171)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_defended_visible_weaker(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][1]].lower()]
                    if val < 10:  # don't encode bomb or flag deaths in battle planes
                        if env.current_player == 1:  # opposite player is defending
                            pl0_planes[val, dst] = 1
                        else:
                            pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    for p in range(10):
                        if not torch.allclose(planes[p], pl0_planes[p]):
                            print(
                                f"=== Acting player 0 at cell {99-coords[0][1]}: Mismatch for piece type {p}"
                            )
                            print("Infostate planes")
                            print(planes[p].view(-1, 10, 10))
                            print("My planes")
                            print(pl0_planes[p].view(-1, 10, 10))
                            self.fail()
                else:
                    for p in range(10):
                        if not torch.allclose(planes[p], pl1_planes[p]):
                            print(
                                f"=== Acting player 1 at cell {coords[0][1]}: Mismatch for piece type {p}"
                            )
                            print("Infostate planes")
                            print(planes[p].view(-1, 10, 10))
                            print("My planes")
                            print(pl1_planes[p].view(-1, 10, 10))
                            self.fail()

    def test_our_defended_visible_tie(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_visible_defended_tie_spy")
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_visible_defended_tie_marshal")
            + 1
        )
        self.assertEqual(s, 171)
        self.assertEqual(e, 181)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_defended_visible_tie(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][1]].lower()]
                    if val < 10:  # don't encode bombs or flags
                        if env.current_player == 1:
                            pl0_planes[val, dst] = 1
                        else:
                            pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())

    def test_our_defended_hidden(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_hidden_defended_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_deathstatus_hidden_defended_marshal") + 1
        self.assertEqual(s, 181)
        self.assertEqual(e, 191)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_defended_hidden(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][1]].lower()]
                    if val < 10:  # don't record bombs or flags
                        if env.current_player == 1:
                            pl0_planes[val, dst] = 1
                        else:
                            pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    print(planes[:, -1])
                    print(pl1_planes[:, -1])
                    self.assertTrue((planes == pl1_planes).all())

    def test_their_attacked_visible_stronger(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=10,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index(
            "their_deathstatus_attacked_visible_stronger_spy"
        )
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index(
                "their_deathstatus_attacked_visible_stronger_marshal"
            )
            + 1
        )
        self.assertEqual(s, 191)
        self.assertEqual(e, 201)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_attacked_visible_stronger(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][0]].lower()]
                    if env.current_player == 1:
                        pl0_planes[val, dst] = 1
                    else:
                        pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    for p in range(10):
                        if not torch.allclose(planes[p], pl0_planes[p]):
                            print(
                                f"=== Acting player 0 at cell {99-coords[0][1]}: Mismatch for piece type {p}"
                            )
                            print("Infostate planes")
                            print(planes[p].view(-1, 10, 10))
                            print("My planes")
                            print(pl0_planes[p].view(-1, 10, 10))
                            for i in range(80):
                                print(
                                    i,
                                    env.env.get_board_tensor(env.current_step)[
                                        0,
                                        (800 + 24 + 2 + 2 + 1 + 1 + 1 + 10 + 2 * i) : (
                                            800 + 24 + 2 + 2 + 1 + 1 + 1 + 10 + 2 * i + 2
                                        ),
                                    ].cpu(),
                                )

                            self.fail()
                else:
                    for p in range(10):
                        if not torch.allclose(planes[p], pl1_planes[p]):
                            print(
                                f"=== Acting player 1 at cell {coords[0][1]}: Mismatch for piece type {p}"
                            )
                            print("Infostate planes")
                            print(planes[p].view(-1, 10, 10))
                            print("My planes")
                            print(pl1_planes[p].view(-1, 10, 10))

                            for i in range(80):
                                print(
                                    i,
                                    env.env.get_board_tensor(env.current_step)[
                                        0,
                                        (800 + 24 + 2 + 2 + 1 + 1 + 1 + 10 + 2 * i) : (
                                            800 + 24 + 2 + 2 + 1 + 1 + 1 + 10 + 2 * i + 2
                                        ),
                                    ].cpu(),
                                )
                            self.fail()

    def test_their_attacked_visible_tie(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=10,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_deathstatus_attacked_visible_tie_spy")
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index(
                "their_deathstatus_attacked_visible_tie_marshal"
            )
            + 1
        )
        self.assertEqual(s, 201)
        self.assertEqual(e, 211)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_attacked_visible_tie(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][0]].lower()]
                    if env.current_player == 1:
                        pl0_planes[val, dst] = 1
                    else:
                        pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())

    def test_their_attacked_hidden(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_deathstatus_attacked_hidden_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_deathstatus_attacked_hidden_marshal") + 1
        self.assertEqual(s, 211)
        self.assertEqual(e, 221)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_attacked_hidden(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][0]].lower()]
                    if env.current_player == 1:
                        pl0_planes[val, dst] = 1
                    else:
                        pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())

    def test_their_defended_visible_weaker(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_deathstatus_visible_defended_weaker_spy")
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index(
                "their_deathstatus_visible_defended_weaker_marshal"
            )
            + 1
        )
        self.assertEqual(s, 221)
        self.assertEqual(e, 231)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_defended_visible_weaker(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][1]].lower()]
                    if val < 10:  # don't record flag or bomb
                        if env.current_player == 0:
                            pl0_planes[val, dst] = 1
                        else:
                            pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())

    def test_their_defended_visible_tie(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_deathstatus_visible_defended_tie_spy")
        e = (
            env.INFOSTATE_CHANNEL_DESCRIPTION.index(
                "their_deathstatus_visible_defended_tie_marshal"
            )
            + 1
        )
        self.assertEqual(s, 231)
        self.assertEqual(e, 241)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_defended_visible_tie(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][1]].lower()]
                    if val < 10:  # don't record flag or bomb
                        if env.current_player == 0:
                            pl0_planes[val, dst] = 1
                        else:
                            pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())

    def test_their_defended_hidden(self):
        env = Stratego(num_envs=1, traj_len_per_player=10)
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_deathstatus_hidden_defended_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_deathstatus_hidden_defended_marshal") + 1
        self.assertEqual(s, 241)
        self.assertEqual(e, 251)
        action_tensor = torch.tensor([0], device="cuda", dtype=torch.int32)
        planes = env.current_infostate_tensor[0, s:e].view(10, 100)
        for _ in range(10):
            env.reset()
            pl0_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            pl1_planes = torch.zeros(10, 100, device="cuda")  # piece type x height x width
            while not env.current_is_terminal:
                env.sample_random_legal_action(action_tensor)
                coords = pystratego.util.actions_to_abs_coordinates(
                    action_tensor, env.current_player
                )
                if is_defended_hidden(env.current_board_strs, coords):
                    dst = coords[0][1]
                    val = letter_to_index[env.current_board_strs[0][::2][coords[0][1]].lower()]
                    if val < 10:  # don't record flag or bomb
                        if env.current_player == 0:
                            pl0_planes[val, dst] = 1
                        else:
                            pl1_planes[val, 99 - dst] = 1
                env.apply_actions(action_tensor)
                planes = env.current_infostate_tensor[0, s:e].view(10, 100)
                if env.current_player == 0:
                    self.assertTrue((planes == pl0_planes).all())
                else:
                    self.assertTrue((planes == pl1_planes).all())


if __name__ == "__main__":
    unittest.main()
