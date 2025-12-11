import os
import glob
import unittest
import torch
import random
from math import factorial as fac

from pyengine.utils import get_pystratego


from importlib.machinery import ExtensionFileLoader

pystratego = get_pystratego()


class FullPieceArrangementGeneratorTest(unittest.TestCase):
    def test_first_and_last(self):
        gen = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.CLASSIC)

        self.assertEqual(
            gen.num_possible_arrangements(),
            fac(40) // (fac(8) * fac(5) * fac(4) * fac(4) * fac(4) * fac(3) * fac(2) * fac(6)),
        )

        # fmt: off
        self.assertTrue(
            torch.allclose(
                gen.generate_arrangements([0, 1, gen.num_possible_arrangements() - 1]),
                torch.tensor([
                     [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,
                       4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  7,  7,  8,  9, 10, 11, 11,
                      11, 11, 11, 11],
                     [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,
                       4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  7,  7,  8,  9, 11, 10, 11,
                      11, 11, 11, 11],
                     [11, 11, 11, 11, 11, 11, 10,  9,  8,  7,  7,  6,  6,  6,  5,  5,  5,  5,
                       4,  4,  4,  4,  3,  3,  3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,
                       1,  1,  1,  0]
                ], dtype=torch.uint8)
            ))
        # fmt: on

    def test_encoding_decoding(self):
        gen = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.CLASSIC)

        N = 1000
        K = gen.num_possible_arrangements()

        ids = [random.randint(0, K - 1) for _ in range(N)]
        arrangements = gen.generate_arrangements(ids)
        decoded_ids = gen.arrangement_ids(arrangements)

        self.assertEqual(ids, decoded_ids)


class BarragePieceArrangementGeneratorTest(unittest.TestCase):
    def test_first_and_last(self):
        gen = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.BARRAGE)

        self.assertEqual(
            gen.num_possible_arrangements(),
            fac(40) // (fac(32) * fac(2)),
        )

        # fmt: off
        self.assertTrue(
            torch.allclose(
                gen.generate_arrangements([0, 1, gen.num_possible_arrangements() - 1]),
                torch.tensor([
                    [ 0,  1,  1,  2,  8,  9, 10, 11, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                     13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                     13, 13, 13, 13],
                    [ 0,  1,  1,  2,  8,  9, 10, 13, 11, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                     13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                     13, 13, 13, 13],
                    [13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                     13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 11, 10,  9,  8,
                     2,  1,  1,  0]
                ], dtype=torch.uint8)
            ))
        # fmt: on

    def test_encoding_decoding(self):
        gen = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.BARRAGE)

        N = 1000
        K = gen.num_possible_arrangements()

        ids = [random.randint(0, K - 1) for _ in range(N)]
        arrangements = gen.generate_arrangements(ids)
        decoded_ids = gen.arrangement_ids(arrangements)

        self.assertEqual(ids, decoded_ids)

    def test_encoding_decoding_strings(self):
        gen = pystratego.PieceArrangementGenerator(pystratego.BoardVariant.BARRAGE)

        N = 1000
        K = gen.num_possible_arrangements()

        ids = [random.randint(0, K - 1) for _ in range(N)]
        arrangements = gen.generate_arrangements(ids)

        ids2string = gen.generate_string_arrangements(ids)
        tensor2string = pystratego.util.arrangement_strings_from_tensor(arrangements.cuda())
        tensor2string_cuda = pystratego.util.arrangement_strings_from_tensor(arrangements.cuda())

        self.assertEqual(ids2string, tensor2string)
        self.assertEqual(ids2string, tensor2string_cuda)


if __name__ == "__main__":
    unittest.main()
