import os
import unittest
import random
import glob
from importlib.machinery import ExtensionFileLoader

import torch

root = os.path.dirname(os.path.abspath(__file__))
pystratego_path = glob.glob(f"{root}/../build/pystratego*.so")[0]
pystratego = ExtensionFileLoader("pystratego", pystratego_path).load_module()


class RewardTest(unittest.TestCase):
    def test_equalpieces_equalcounts(self):
        piece_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40]
        c = random.randint(1, 30)
        piece_counter[random.randint(0, 9)] += c
        piece_counter[-1] -= c
        bs = pystratego.PieceArrangementGenerator(piece_counter)
        arrangements = bs.generate_string_arrangements([69, 420])

        env = pystratego.StrategoRolloutBuffer(
            1024,
            32,
            reset_behavior=pystratego.ResetBehavior.RANDOM_CUSTOM_INITIAL_ARRANGEMENT,
            move_memory=32,
            max_num_moves=20000,
            initial_arrangements=[arrangements, arrangements],
        )

        action_tensor = torch.zeros(env.num_envs, dtype=torch.int32, device="cuda")

        num_terminated = 0
        for _ in range(10000):
            env.sample_random_legal_action(action_tensor)
            env.apply_actions(action_tensor)

            t = env.current_step()
            env.compute_reward_pl0(t)

            terminated_rewards = env.reward_pl0[env.get_terminated_since(t) == 1]
            self.assertTrue(
                torch.allclose(terminated_rewards, torch.zeros_like(terminated_rewards))
            )
            num_terminated += terminated_rewards.numel()

        print("Num terminated:", num_terminated)


if __name__ == "__main__":
    unittest.main()
