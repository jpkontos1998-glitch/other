import unittest
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from pyengine.core.env import Stratego
import torch as th
import numpy as np


class FreeMemoryTest(unittest.TestCase):
    def test_free(self):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)

        As, Bs, Cs = [], [], []
        for t in range(30):
            info = nvmlDeviceGetMemoryInfo(h)
            if t > 3:
                As.append(info.used)
            env = Stratego(
                num_envs=128,
                traj_len_per_player=200,
            )
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            if t > 3:
                Bs.append(info.used)
            del env
            th.cuda.empty_cache()
            th.cuda.synchronize()
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            if t > 3:
                Cs.append(info.used)

        print(
            f"Memory before creating env: {np.mean(As) / 1024**2:.2f} ± {np.std(As) / 1024**2:.2f} MB"
        )
        print(
            f"Memory after env creation: {np.mean(Bs) / 1024**2:.2f} ± {np.std(Bs) / 1024**2:.2f} MB"
        )
        print(
            f"Memory after deleting env: {np.mean(Cs) / 1024**2:.2f} ± {np.std(Cs) / 1024**2:.2f} MB"
        )

        self.assertTrue(np.std(As) <= 1)
        self.assertTrue(np.std(Bs) <= 1)
        self.assertTrue(np.std(Cs) <= 1)


if __name__ == "__main__":
    unittest.main()
