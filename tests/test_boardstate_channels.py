import unittest

from pyengine.core.env import Stratego
from pyengine.utils import get_pystratego

pystratego = get_pystratego()


class TestBoardstateChannels(unittest.TestCase):
    def test_boardstate_channels(self):
        boardchannels = pystratego.BOARDSTATE_CHANNEL_DESCRIPTION
        env = Stratego(1, 1)
        infostatechannels = env.INFOSTATE_CHANNEL_DESCRIPTION
        for a, b in zip(boardchannels, infostatechannels):
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
