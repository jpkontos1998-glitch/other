from types import MappingProxyType

import torch

from .load_pystratego import get_pystratego

pystratego = get_pystratego()

# Stratego players
N_PLAYER = 2

# Stratego board
BOARD_LEN = 10
N_BOARD_CELL = BOARD_LEN * BOARD_LEN
N_LAKE_CELL = 8
N_OCCUPIABLE_CELL = N_BOARD_CELL - N_LAKE_CELL

# Stratego arrangement
N_ARRANGEMENT_ROW = 4
N_ARRANGEMENT_COL = 10
ARRANGEMENT_SIZE = N_ARRANGEMENT_ROW * N_ARRANGEMENT_COL

# Stratego pieces
N_PIECE_TYPE = pystratego.NUM_PIECE_TYPES
N_PLAYER_PIECE_TYPE = 12
N_NONPLAYER_PIECE_TYPE = 2  # empty and lake
N_MOVABLE_PLAYER_PIECE_TYPE = 10
N_IMMOVABLE_PLAYER_PIECE_TYPE = 2  # flag and bomb
N_BARRAGE_PIECE = 8
N_CLASSIC_PIECE = 40

# Stratego actions
MAX_N_POSSIBLE_DST = 2 * (
    BOARD_LEN - 1
)  # ({(*, col)} \setminus {(row, col)}) \cup ({(row, *)} \setminus {(row, col)})
UPPER_BOUND_N_LEGAL_ACTION = 100
N_ACTION = 1800

# Stratego piece identifiers
N_PIECE_ID = (
    256  # Valid piece IDs are [0, 40] (our pieces), [60, 100] (opponent pieces), and 255 (empty)
)
EMPTY_PIECE_ID = 255

# Stratego move summary
NOTMOVE_ID = 254
TRAILING_LAST_MOVES_DIM = 12
LAST_MOVES_DICTIONARY_SIZE = 256

# Stratego board parameterization
ARRANGEMENT_ROW_INDICES = (0, 1, 2, 3)
CORRIDOR_COL_INDICES = (0, 1, 4, 5, 8, 9)
LAKE_INDICES = (42, 43, 46, 47, 52, 53, 56, 57)
assert len(LAKE_INDICES) == N_LAKE_CELL
PIECE_INDICES = MappingProxyType(
    {
        "spy": 0,
        "scout": 1,
        "miner": 2,
        "sergeant": 3,
        "lieutenant": 4,
        "captain": 5,
        "major": 6,
        "colonel": 7,
        "general": 8,
        "marshal": 9,
        "flag": 10,
        "bomb": 11,
        "lake": 12,
        "empty": 13,
    }
)
SPY_IDX = PIECE_INDICES["spy"]
SCOUT_IDX = PIECE_INDICES["scout"]
MINER_IDX = PIECE_INDICES["miner"]
SERGEANT_IDX = PIECE_INDICES["sergeant"]
LIEUTENANT_IDX = PIECE_INDICES["lieutenant"]
CAPTAIN_IDX = PIECE_INDICES["captain"]
MAJOR_IDX = PIECE_INDICES["major"]
COLONEL_IDX = PIECE_INDICES["colonel"]
GENERAL_IDX = PIECE_INDICES["general"]
MARSHAL_IDX = PIECE_INDICES["marshal"]
FLAG_IDX = PIECE_INDICES["flag"]
BOMB_IDX = PIECE_INDICES["bomb"]
MOVABLE_PLAYER_PIECE_TYPE_SLICE = slice(SPY_IDX, MARSHAL_IDX + 1)
IMMOVABLE_PLAYER_PIECE_TYPE_SLICE = slice(FLAG_IDX, BOMB_IDX + 1)
PLAYER_PIECE_TYPE_SLICE = slice(SPY_IDX, BOMB_IDX + 1)
IS_IMMOVABLE_PLAYER_PIECE_TYPE = tuple(10 * [False] + 2 * [True])

# Infostate
INFOSTATE_OPP_PROB_SLICE = slice(
    pystratego.BOARDSTATE_CHANNEL_DESCRIPTION.index("their_spy_prob"),
    pystratego.BOARDSTATE_CHANNEL_DESCRIPTION.index("their_bomb_prob") + 1,
)

# Categorical outcomes
CATEGORICAL_AGGREGATION = torch.tensor([-1, 0, 1], dtype=torch.float32)  # lose, tie, win
N_VF_CAT = CATEGORICAL_AGGREGATION.numel()

# Logging
UPPER_QUANTILES = torch.tensor(
    [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]
)
