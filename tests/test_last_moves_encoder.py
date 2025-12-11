import unittest

import torch

from pyengine.networks.last_moves_encoder import LastMovesEncoder
from pyengine.utils.constants import LAST_MOVES_DICTIONARY_SIZE, TRAILING_LAST_MOVES_DIM


class LastMovesEncoderTest(unittest.TestCase):
    def test_matches_loop(self):
        batch_size = 10
        embed_dim = 128
        encoder = LastMovesEncoder(embed_dim=embed_dim, pos_emb_std=0.1)
        last_moves = torch.randint(
            0, LAST_MOVES_DICTIONARY_SIZE, (batch_size, TRAILING_LAST_MOVES_DIM)
        )
        output = encoder(last_moves)
        my_output = torch.zeros(batch_size, embed_dim)
        for i in range(batch_size):
            for j in range(TRAILING_LAST_MOVES_DIM):
                my_output[i] += encoder.encodings[j, last_moves[i, j], :]
        self.assertTrue(torch.allclose(output, my_output))


if __name__ == "__main__":
    unittest.main()
