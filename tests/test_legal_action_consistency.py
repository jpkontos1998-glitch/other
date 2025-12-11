import unittest

import torch
from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()


class LegalActionTests(unittest.TestCase):
    def test_wrapper(self):
        legal_actions = {}
        total_steps = 100
        env = Stratego(512, total_steps, False, True, two_square_rule=True)
        for i in range(total_steps):
            legal_actions[env.current_step] = env.current_legal_action_mask
            action = env.sample_random_legal_action()
            env.apply_actions(action)
            # recomputed_legal_action = env.legal_action_mask(i)
            # self.assertTrue((legal_actions[i] == recomputed_legal_action).all())
        print("====")
        for i in range(total_steps):
            recomputed_legal_action = env.legal_action_mask(i)
            self.assertTrue((legal_actions[i] == recomputed_legal_action).all())

    def test_naked(self):
        legal_actions = {}
        total_steps = 1000
        env = pystratego.StrategoRolloutBuffer(
            total_steps + 2,
            512,
            reset_behavior=pystratego.ResetBehavior.RANDOM_JB_BARRAGE_BOARD,
            two_square_rule=True,
        )
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")
        for i in range(total_steps):
            self.assertEqual(i, env.current_step())
            # print(i, env.get_twosquare_state(i))
            env.compute_legal_action_mask(i)
            legal_actions[env.current_step()] = env.legal_action_mask.clone()
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)
        print("====")
        for i in range(total_steps):
            # print(i, env.get_twosquare_state(i))
            env.compute_legal_action_mask(i)
            self.assertTrue((legal_actions[i] == env.legal_action_mask).all())


if __name__ == "__main__":
    unittest.main()
