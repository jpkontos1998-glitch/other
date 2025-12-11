import os
import glob
from importlib.machinery import ExtensionFileLoader
import torch
import unittest

from pyengine.core.env import Stratego

root = os.path.dirname(os.path.abspath(__file__))
pystratego_path = glob.glob(f"{root}/../build/pystratego*.so")[0]
pystratego = ExtensionFileLoader("pystratego", pystratego_path).load_module()

num_rows = 100
num_envs = 8


class SamplerTest(unittest.TestCase):
    def test_samplers(self):
        env = Stratego(num_envs, num_rows)
        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        for _ in range(num_rows):
            mask = env.current_legal_action_mask.to("cpu")

            env.sample_first_legal_action(action_tensor)
            first_action = action_tensor.to("cpu")
            for env_idx in range(env.num_envs):
                found = False
                for i in range(pystratego.NUM_ACTIONS):
                    if mask[env_idx][i]:
                        found = True
                        self.assertEqual(first_action[env_idx], i)
                        break
                self.assertTrue(found)

            env.sample_random_legal_action(action_tensor)

            # FIXME: Check these are indeed uniform

            # Double check that the actions selected are indeed valid
            self.assertTrue(mask[range(env.num_envs), action_tensor.to("cpu")].all())

            env.apply_actions(action_tensor)


if __name__ == "__main__":
    unittest.main()
