import os
import glob
from importlib.machinery import ExtensionFileLoader
import torch


root = os.path.dirname(os.path.abspath(__file__))
pystratego_path = glob.glob(f"{root}/../build/pystratego*.so")[0]
pystratego = ExtensionFileLoader("pystratego", pystratego_path).load_module()


def run(num_envs, num_steps=1024, num_reps=10):
    env = pystratego.StrategoRolloutBuffer(num_steps + 1, num_envs, move_memory=32)
    action_tensor = torch.zeros(num_envs, dtype=torch.int32, device="cuda")

    for _ in range(num_reps):
        env.reset()
        for t in range(num_steps):
            # env.sample_random_legal_action(action_tensor)
            env.sample_first_legal_action(action_tensor)

            # FIXME: Optionally double check that the actions selected are indeed valid
            # assert(legal_action_mask[range(num_envs), action_tensor].all())

            env.apply_actions(action_tensor)
            env.compute_infostate_tensor(t)


run(1024)
