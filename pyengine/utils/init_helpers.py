def is_valid_line(line):
    s = line.strip()
    return not s.startswith("#") and len(s) == 40


# | Piece       |  char | value |
# +-------------+-------+-------+
# | SPY         |   C   |   0   |
# | SCOUT       |   D   |   1   |
# | MINER       |   E   |   2   |
# | SERGEANT    |   F   |   3   |
# | LIEUTENANT  |   G   |   4   |
# | CAPTAIN     |   H   |   5   |
# | MAJOR       |   I   |   6   |
# | COLONEL     |   J   |   7   |
# | GENERAL     |   K   |   8   |
# | MARSHAL     |   L   |   9   |
# | FLAG        |   M   |  10   |
# | BOMB        |   B   |  11   |
# +-------------+-------+-------+
# | LAKE        |   _   |  12   |
# | EMPTY       |   A   |  13   |
# +-------------+-------+-------+
STRING_TO_CHAR = {
    "SPY": "C",
    "SCOUT": "D",
    "MINER": "E",
    "SERGEANT": "F",
    "LIEUTENANT": "G",
    "CAPTAIN": "H",
    "MAJOR": "I",
    "COLONEL": "J",
    "GENERAL": "K",
    "MARSHAL": "L",
    "FLAG": "M",
    "BOMB": "B",
    "LAKE": "_",
    "EMPTY": "A",
}
CHAR_TO_VAL = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "H": 5,
    "I": 6,
    "J": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "B": 11,
    "_": 12,
    "A": 13,
}

CHAR_TO_VAL_BLUE = {
    k: i for i, k in enumerate(["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "N"])
}

VAL_TO_CHAR = sorted(CHAR_TO_VAL, key=CHAR_TO_VAL.get)

BLUE_CHAR_TO_RED_CHAR = {k: VAL_TO_CHAR[v] for k, v in CHAR_TO_VAL_BLUE.items()}

COUNTERS = {
    "classic": [
        1,  # SPY
        8,  # SCOUT
        5,  # MINER
        4,  # SERGEANT
        4,  # LIEUTENANT
        4,  # CAPTAIN
        3,  # MAJOR
        2,  # COLONEL
        1,  # GENERAL
        1,  # MARSHAL
        1,  # FLAG
        6,  # BOMB
    ],
    "barrage": [
        1,  # SPY
        2,  # SCOUT
        1,  # MINER
        0,  # SERGEANT
        0,  # LIEUTENANT
        0,  # CAPTAIN
        0,  # MAJOR
        0,  # COLONEL
        1,  # GENERAL
        1,  # MARSHAL
        1,  # FLAG
        1,  # BOMB
    ],
}


def to_byte(ch, color=0):
    assert color in (0, 1, 2)
    value = CHAR_TO_VAL[ch]
    visible = 0
    if ch == "_":
        color = 3
        visible = 1
    elif ch == "A":
        color = 0
        visible = 1
    return value + color * 16 + visible * 64


def piece_counters(s):
    ctrs = [0] * 12
    for ch in s:
        val = CHAR_TO_VAL[ch]
        if val <= 11:
            ctrs[val] += 1
    return ctrs


def remap_chars(s):
    cs = []
    for c in s:
        if c == "A":
            cs += ["A"]
        else:
            assert c >= "N" and c <= "Y"
            cs += [chr(ord(c) - ord("N") + ord("B"))]
    return "".join(cs)


def initial_boards_to_arrangements(initial_boards):
    return (
        [board[:40] for board in initial_boards],
        [remap_chars(board[99:59:-1]) for board in initial_boards],
    )
