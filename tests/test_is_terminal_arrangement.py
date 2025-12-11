import unittest
import random

from pyengine.core.env import pretty_board
from pyengine.utils import get_pystratego

pystratego = get_pystratego()

immovable_letters = tuple(6 * ["B"] + ["M"])
movable_letters = tuple(
    1 * ["C"]
    + 8 * ["D"]
    + 5 * ["E"]
    + 4 * ["F"]
    + 4 * ["G"]
    + 4 * ["H"]
    + 3 * ["I"]
    + 2 * ["J"]
    + ["K"]
    + ["L"]
)
letters = tuple(
    1 * ["C"]
    + 8 * ["D"]
    + 5 * ["E"]
    + 4 * ["F"]
    + 4 * ["G"]
    + 4 * ["H"]
    + 3 * ["I"]
    + 2 * ["J"]
    + ["K"]
    + ["L"]
    + ["M"]
    + 6 * ["B"]
)
assert len(letters) == 40
assert len(immovable_letters) + len(movable_letters) == len(letters)

movable_locations = [30, 31, 34, 35, 38, 39]
assert len(movable_locations) == 6
immovable_locations = [i for i in range(40) if i not in movable_locations]

"""
Arrangements look like this:

00 01 02 03 04 05 06 07 08 09
10 11 12 13 14 15 16 17 18 19
20 21 22 23 24 25 26 27 28 29
30 31 32 33 34 35 36 37 38 39
|| || XX XX || || XX XX || ||

where numbers correspond to piece_ids,
|| denotes to corridors,
XX denotes to lakes.
At the first move of the game, there are
most 6 movable pieces:
30, 31, 34, 35, 38, 39.
An arrangement is terminal if and only if
all six of these piece are of immovable types 
(i.e. flag or bomb).
"""


def generate_random_arrangement():
    return "".join(random.sample(letters, k=len(letters)))


def is_terminal_arrangement(board_str):
    immovable_pieces = ("B", "M")
    return all(board_str[loc] in immovable_pieces for loc in movable_locations)


def generate_terminal_arrangement():
    immovable = random.sample(immovable_letters, k=len(immovable_letters))
    movable = random.sample(movable_letters, k=len(movable_letters))
    immovable_locs = [random.sample(immovable_locations, k=1)[0]] + [30, 31, 34, 35, 38, 39]
    arr = movable
    for loc, let in zip(immovable_locs, immovable):
        arr.insert(loc, let)
    return "".join(arr)


class IsTerminalArrangementTest(unittest.TestCase):
    def test_random_agreement(self):
        n_arr = 100
        arrs = [generate_random_arrangement() for _ in range(n_arr)]
        my_terms = [is_terminal_arrangement(arr) for arr in arrs]
        py_terms = pystratego.util.is_terminal_arrangement(arrs)
        for arr, t1, t2 in zip(arrs, my_terms, py_terms):
            if t1 != t2:
                print(arr)
                print(pretty_board("@".join(list(arr)) + "@" + "." * 120))
            self.assertEqual(t1, t2)

    def test_terminal_agreement(self):
        n_arr = 100
        arrs = [generate_terminal_arrangement() for _ in range(n_arr)]
        my_terms = [is_terminal_arrangement(arr) for arr in arrs]
        py_terms = pystratego.util.is_terminal_arrangement(arrs)
        for t1, t2 in zip(my_terms, py_terms):
            self.assertEqual(t1, t2)


if __name__ == "__main__":
    unittest.main()
