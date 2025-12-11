import unittest

import torch

from pyengine.core.buffer import compress_log_probs, expand_log_probs
from pyengine.utils.constants import N_ACTION, UPPER_BOUND_N_LEGAL_ACTION


class TestLogProbCompression(unittest.TestCase):
    def test_compress_log_probs(self):
        B = 100
        log_probs = torch.randn(B, N_ACTION, device="cuda")
        # Generate legal action mask with at most UPPER_BOUND_N_LEGAL_ACTION legal actions per row
        legal_action_mask = torch.zeros(B, N_ACTION, device="cuda", dtype=torch.bool)
        for i in range(B):
            num_legal = torch.randint(1, UPPER_BOUND_N_LEGAL_ACTION + 1, (1,)).item()
            legal_indices = torch.randperm(N_ACTION)[:num_legal]
            legal_action_mask[i, legal_indices] = True
        log_probs[~legal_action_mask] = -1e10
        compressed = compress_log_probs(log_probs, legal_action_mask)
        expanded = expand_log_probs(compressed, legal_action_mask)
        self.assertTrue(torch.allclose(log_probs, expanded))
        self.assertTrue(expanded.shape == (B, N_ACTION))


if __name__ == "__main__":
    unittest.main()
