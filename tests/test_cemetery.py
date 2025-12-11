import unittest

import torch
from torch.nn.functional import one_hot

from pyengine.core.env import Stratego
from pyengine import utils


pystratego = utils.get_pystratego()


class CemeteryTest(unittest.TestCase):
    def test_our_cemetery(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=10,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_dead_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("our_dead_bomb") + 1
        self.assertEqual(s, 109)
        self.assertEqual(e, 120)
        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for _ in range(1):
            env.reset()
            mask = torch.tensor(10 * [True] + [False] + [True], device="cuda")
            zero_arrangement_pl0 = one_hot(
                env.current_zero_arrangements[0][0].long(), num_classes=12
            )[:, mask]
            zero_arrangement_pl1 = one_hot(
                env.current_zero_arrangements[1][0].long(), num_classes=12
            )[:, mask]
            while not env.current_is_terminal:
                our_cemetery = env.current_infostate_tensor[0, s:e].flatten(start_dim=1)
                self.assertTrue((our_cemetery[:, 40:] == torch.tensor([0], device="cuda")).all())
                for piece_id in range(40):
                    is_alive = (env.current_piece_ids[0].flatten() == piece_id).any()
                    if is_alive:
                        self.assertTrue(
                            (our_cemetery[:, piece_id] == torch.tensor([0], device="cuda")).all()
                        )
                        continue
                    if env.current_player == 0:
                        self.assertTrue(
                            (our_cemetery[:, piece_id] == zero_arrangement_pl0[piece_id]).all()
                        )
                    else:
                        self.assertTrue(
                            (our_cemetery[:, piece_id] == zero_arrangement_pl1[piece_id]).all()
                        )
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)

    def test_their_cemetery(self):
        env = Stratego(
            num_envs=1,
            traj_len_per_player=10,
        )
        s = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_dead_spy")
        e = env.INFOSTATE_CHANNEL_DESCRIPTION.index("their_dead_bomb") + 1
        self.assertEqual(s, 120)
        self.assertEqual(e, 131)
        action_tensor = torch.tensor([0], dtype=torch.int32, device="cuda")
        for _ in range(1):
            env.reset()
            mask = torch.tensor(10 * [True] + [False] + [True], device="cuda")
            zero_arrangement_pl0 = one_hot(
                env.current_zero_arrangements[0][0].long(), num_classes=12
            )[:, mask]
            zero_arrangement_pl1 = one_hot(
                env.current_zero_arrangements[1][0].long(), num_classes=12
            )[:, mask]
            while not env.current_is_terminal:
                their_cemetery = env.current_infostate_tensor[0, s:e].flatten(start_dim=1)
                self.assertTrue((their_cemetery[:, :60] == torch.tensor([0], device="cuda")).all())
                for piece_id in range(60, 100):
                    is_alive = (env.current_piece_ids[0].flatten() == piece_id).any()
                    if is_alive:
                        self.assertTrue(
                            (their_cemetery[:, piece_id] == torch.tensor([0], device="cuda")).all()
                        )
                        continue
                    if env.current_player == 0:
                        self.assertTrue(
                            (
                                their_cemetery[:, piece_id] == zero_arrangement_pl1[99 - piece_id]
                            ).all()
                        )
                    else:
                        self.assertTrue(
                            (
                                their_cemetery[:, piece_id] == zero_arrangement_pl0[99 - piece_id]
                            ).all()
                        )
                env.sample_random_legal_action(action_tensor)
                env.apply_actions(action_tensor)


if __name__ == "__main__":
    unittest.main()
