import unittest
import random

from pyengine.utils import get_pystratego
import psutil
import os

pystratego = get_pystratego()


class MemoryTest(unittest.TestCase):
    def test_increase_barrage(self):
        generator = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.BARRAGE)

        N = 100000
        K = generator.num_possible_arrangements()

        ids = [random.randint(0, K - 1) for _ in range(N)]
        x = generator.generate_arrangements(ids)
        x = x.cpu()

        process = psutil.Process(os.getpid())
        for i in range(100):
            generator.arrangement_ids(x)
            if i == 10:
                initial_num_bytes = process.memory_info().rss
        final_num_bytes = process.memory_info().rss
        self.assertLess(final_num_bytes - initial_num_bytes, 2 * 10**6)  # At most 2 MB difference

    def test_increase_classic(self):
        generator = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.CLASSIC)

        N = 100000
        K = generator.num_possible_arrangements()

        ids = [random.randint(0, K - 1) for _ in range(N)]
        x = generator.generate_arrangements(ids)
        x = x.cpu()

        process = psutil.Process(os.getpid())
        for i in range(100):
            generator.arrangement_ids(x)
            if i == 10:
                initial_num_bytes = process.memory_info().rss
        final_num_bytes = process.memory_info().rss
        self.assertLess(final_num_bytes - initial_num_bytes, 2 * 10**6)  # At most 2 MB difference


if __name__ == "__main__":
    unittest.main()
