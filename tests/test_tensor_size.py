import torch
import unittest

from pyengine.utils import get_pystratego

pystratego = get_pystratego()


class TensorTestDefaultConf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_envs = 100
        cls.env = pystratego.StrategoRolloutBuffer(
            1024, cls.num_envs, enable_hidden_and_types_planes=True, enable_dm_planes=True
        )

    def test_legal_action_mask_shape(self):
        self.assertEqual(self.env.legal_action_mask.shape, (self.num_envs, pystratego.NUM_ACTIONS))

    def test_legal_action_mask_dtype(self):
        self.assertEqual(self.env.legal_action_mask.dtype, torch.bool)

    def test_legal_action_mask_device(self):
        self.assertTrue(self.env.legal_action_mask.is_cuda)

    def test_infostate_tensor_shape(self):
        self.assertEqual(
            self.env.NUM_INFOSTATE_CHANNELS,
            pystratego.NUM_BOARD_STATE_CHANNELS + pystratego.StrategoConf().move_memory * 6,
        )
        self.assertEqual(
            self.env.infostate_tensor.shape,
            (self.num_envs, self.env.NUM_INFOSTATE_CHANNELS, 10, 10),
        )

    def test_infostate_tensor_dtype(self):
        self.assertTrue(self.env.infostate_tensor.dtype in (torch.float16, torch.float32))

    def test_infostate_tensor_device(self):
        self.assertTrue(self.env.infostate_tensor.is_cuda)


class TensorTestCustomConf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_envs = 100
        cls.env = pystratego.StrategoRolloutBuffer(
            1024,
            cls.num_envs,
            move_memory=6,
            enable_hidden_and_types_planes=True,
            enable_dm_planes=True,
        )

    def test_legal_action_mask_shape(self):
        self.assertEqual(self.env.legal_action_mask.shape, (self.num_envs, pystratego.NUM_ACTIONS))

    def test_legal_action_mask_dtype(self):
        self.assertEqual(self.env.legal_action_mask.dtype, torch.bool)

    def test_legal_action_mask_device(self):
        self.assertTrue(self.env.legal_action_mask.is_cuda)

    def test_infostate_tensor_shape(self):
        self.assertEqual(
            self.env.NUM_INFOSTATE_CHANNELS, pystratego.NUM_BOARD_STATE_CHANNELS + 6 * 6
        )
        self.assertEqual(
            self.env.infostate_tensor.shape,
            (self.num_envs, pystratego.NUM_BOARD_STATE_CHANNELS + 6 * 6, 10, 10),
        )

    def test_infostate_tensor_dtype(self):
        self.assertTrue(self.env.infostate_tensor.dtype in (torch.float16, torch.float32))

    def test_infostate_tensor_device(self):
        self.assertTrue(self.env.infostate_tensor.is_cuda)


if __name__ == "__main__":
    unittest.main()
