import unittest

import torch


class LogLossTest(unittest.TestCase):
    def test_equal_grad(self):
        torch.set_grad_enabled(True)
        q_raw = torch.nn.Parameter(torch.rand(5, requires_grad=True))
        q = torch.nn.functional.softmax(q_raw, dim=0)

        # Retain gradients for non-leaf tensor
        q.retain_grad()

        # Select an index for testing
        i = 2  # You can change this to test different components of q

        # Define the first loss: -log q_i
        loss1 = -torch.log(q[i])
        loss1.backward(retain_graph=True)

        # Store gradient of the first loss
        grad_log_q_i = q.grad.clone()
        q.grad.zero_()

        # Define the second loss: -q_i / q_i.detach()
        loss2 = -q[i] / q[i].detach()
        loss2.backward()

        # Store gradient of the second loss
        grad_q_over_detach = q.grad.clone()

        self.assertTrue(torch.allclose(grad_log_q_i, grad_q_over_detach))


if __name__ == "__main__":
    unittest.main()
