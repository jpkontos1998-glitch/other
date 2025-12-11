#pragma once

#include <ATen/core/TensorBody.h>
#include <array>
#include <cstdint>
#include <string>

#include "src/util.h"

const int BOARD_SIZE = 10;
const int NUM_PLAYERS = 2;
const int NUM_PIECE_TYPES = 14; // Including `empty` and `lake`
#define IS_ADJACENT(src, dst) ((dst == src + 1 && dst % 10 != 0) || (dst == src - 1 && src % 10 != 0) || (dst == src + 10) || (dst == src - 10))

enum PieceType : uint8_t
{
    SPY = 0,        // standard movement
    SCOUT = 1,      // scout movement
    MINER = 2,      // standard movement
    SERGEANT = 3,   // standard movement
    LIEUTENANT = 4, // standard movement
    CAPTAIN = 5,    // standard movement
    MAJOR = 6,      // standard movement
    COLONEL = 7,    // standard movement
    GENERAL = 8,    // standard movement
    MARSHAL = 9,    // standard movement
    FLAG = 10,      // no movement
    BOMB = 11,      // no movement
    LAKE = 12,      // no movement
    EMPTY = 13,     // no movement
};                  // (Can be stored in 4 bits)
const uint8_t HIDDEN_PIECE = 15;

// Strados encoding used by openspiel. See also `BoardStrs`.
// The first dimension refers to the color (0 = empty, 1 = red, 2 = blue, 3 = lake),
// the second to the piece type.
const char PIECE_ENCODING[4][NUM_PIECE_TYPES] = {
    {'#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', 'a'}, // color: empty (0)
    {'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'B', '#', '#'}, // color: red   (1)
    {'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'N', '#', '#'}, // color: blue  (2)
    {'#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '_', '#'}, // color: lake  (3)
};

struct Piece
{
    PieceType type : 4;
    uint8_t color : 2; // 0 = EMPTY, 1 = RED, 2 = BLUE, 3 = LAKE
    uint8_t visible : 1;
    uint8_t has_moved : 1;
    uint8_t piece_id;

    // Piece types of the opponent that this piece has threatened
    // (i.e., moved adjacent to). The special type 15 denotes unkonwn.
    uint8_t threatened[2];
    // Piece types of the opponent that this piece has evaded, that is,
    // moved away from in the move successive to when those pieces had
    // moved adjacent.
    uint8_t evaded[2];
    // Piece types of the opponent that this pieces has been close to
    // during its turn.
    uint8_t actively_adjacent[2];

    uint8_t protected_[2];
    uint8_t protected_against[2];
    uint8_t was_protected_by[2];
    uint8_t was_protected_against[2];
};
static_assert(sizeof(Piece) == 16);
static_assert(alignof(Piece) == 1);

enum DeathReason : uint8_t
{
    /// We attacked a visible piece, which happened to be stronger
    ATTACKED_VISIBLE_STRONGER = 0,

    /// We attacked a visible piece, and tied
    ATTACKED_VISIBLE_TIE,

    /// We attacked a hidden piece, and either tied or lost
    ATTACKED_HIDDEN,

    /// We were visible, were attacked (by a visible or hidden piece),
    /// and lost because we were weaker
    VISIBLE_DEFENDED_WEAKER,

    /// We were visible, were attacked (by a visible or hidden piece),
    /// and tied
    VISIBLE_DEFENDED_TIE,

    /// We were hidden, were attacked (by a visible or hidden piece),
    /// and lost (either a tie or we were weaker)
    HIDDEN_DEFENDED,
};

struct DeathStatus
{
    bool is_dead : 1;
    DeathReason death_reason : 3;
    uint8_t piece_type : 4; // The type of the piece that died (zero if not dead)
    uint8_t death_location; // In absolute coordinates
};                          // uint16_t

static_assert(sizeof(DeathStatus) == 2);
static_assert(alignof(DeathStatus) == 1);

// Utility macros to construct pieces of a given color
#define _R(X, piece_id)                                                                          \
    {                                                                                            \
        /*type*/ X, /*color*/ 1, /*visible*/ 0, /*has_moved*/ 0, /*piece_id*/ (uint8_t)piece_id, \
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, { 0, 0 }                             \
    }
#define _B(X, piece_id)                                                                          \
    {                                                                                            \
        /*type*/ X, /*color*/ 2, /*visible*/ 0, /*has_moved*/ 0, /*piece_id*/ (uint8_t)piece_id, \
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, { 0, 0 }                             \
    }
#define _E(X)                                                                       \
    {                                                                               \
        /*type*/ X, /*color*/ 0, /*visible*/ 1, /*has_moved*/ 0, /*piece_id*/ 0xff, \
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, { 0, 0 }                \
    }
#define _L(X)                                                                       \
    {                                                                               \
        /*type*/ X, /*color*/ 3, /*visible*/ 1, /*has_moved*/ 0, /*piece_id*/ 0xff, \
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, { 0, 0 }                \
    }

struct alignas(128) StrategoBoard
{
    Piece pieces[BOARD_SIZE][BOARD_SIZE]; // 1600 bytes
    uint8_t num_hidden[2][12];            //   24 bytes -- Only up to BOMB
    uint8_t num_hidden_unmoved[2];        //    2 bytes
    uint8_t _padding1[2];                 //    2 bytes
    uint8_t prev_dst_abs;                 //    1 byte
    uint8_t prev_prev_dst_abs;            //    1 byte
    uint8_t last_moved_piece_type;        //    1 byte (HIDDEN_PIECE if not visible, 0xff if dead)
    uint8_t deaths[2][5];                 //   10 bytes (which pieces, in their starting position in [0,39], has died for each player)
    DeathStatus death_status[2][40];      //  160 bytes
    uint8_t padding2[119];

    std::string BoardString() const;
}; // Total 1920 bytes = 32 * 60

static_assert(sizeof(StrategoBoard) == 1920);
static_assert(alignof(StrategoBoard) == 128);

/// Checks that the counters match the piece counts from the `pieces` field, and that
/// the `num_hidden_unmoved` field is consistent.
void CheckIsValidInitBoard(const StrategoBoard &board);

/// Produces the string representations of an array of boards in parallel.
/// The input `d_boards` must be GPU-allocated on the given cuda device.
std::vector<std::string> BoardStrs(const StrategoBoard *d_boards,
                                   const uint32_t num_boards,
                                   const uint32_t cuda_device);

enum BoardVariant
{
    BARRAGE,
    CLASSIC
};

using StringArrangements = std::vector<std::string>;

constexpr const std::array<uint8_t, NUM_PIECE_TYPES> CLASSIC_INITIAL_COUNTS = {1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 1, 6, 0, 0};
constexpr const std::array<uint8_t, NUM_PIECE_TYPES> BARRAGE_INITIAL_COUNTS = {1, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 32};

/// This object is initialized with piece counts. It implicitly enumerates, lexicographically, all valid piece
/// arrangement with that given piece count. Given an index in this implicit lexicographic list, it is then
/// able to generate the given piece arrangement (an array of 40 `PieceType` objects).
///
/// The computation is completely performed on the CPU given CUDA's lack of support for 128-bit integers.
///
/// Limitations
/// -----------
///
/// We only support up to 2^115 unique arrangements due to the integer representation we use.
///
/// This class will check that these limits are not exceeded. The limit is satisfied by standard classic and barrage boards.
class PieceArrangementGenerator
{
public:
    /// Initialized the class with the given piece counts. The input piece count is a NUM_PIECE_TYPES-sized
    /// array.
    ///
    /// No lakes are allowed. The total sum of counts needs to be equal to 40.
    PieceArrangementGenerator(const std::array<uint8_t, NUM_PIECE_TYPES> &type_counts);

    explicit PieceArrangementGenerator(const BoardVariant variant);

    /// Generates host-allocated arrangements (as tensor) given the list of ids.
    torch::Tensor GenerateArrangements(const std::vector<uint128_t> &arrangement_ids) const;

    /// Generates host-allocated arrangements (as string) given the list of ids.
    StringArrangements GenerateStringArrangements(const std::vector<uint128_t> &arrangement_ids) const;

    /// Generates the IDs of the given arrangements.
    ///
    /// If CUDA-allocated, the arrangements are first copied onto the host.
    ///
    /// This method panics if the arrangements are not valid (e.g., they contain invalid data
    /// or do not respect the `type_counts` specified for the generator).
    std::vector<uint128_t> ArrangementIds(torch::Tensor arrangements) const;

    /// Returns the number of valid arrangements of pieces for the given
    /// `type_counts`.
    uint128_t NumPossibleArrangements() const { return num_arrangements_; }

private:
    void InitializeCache_();

    std::array<uint8_t, NUM_PIECE_TYPES> type_counts_;
    uint32_t num_pieces_; // Only 40 is supported

    /// Number of possible arrangements given the type counts.
    uint128_t num_arrangements_;

    /// Cache for (n choose k) quantities. The first index is n, the second is k.
    uint64_t cache_[41][41];
};

/// Generates host-allocated arrangements corresponding to the given string.
/// Characters should be in the range `A..M` according to the table in `stratego.h`.
///
/// This method panics if the arrangements are not valid (e.g., they contain invalid
/// characters).
torch::Tensor ArrangementTensorFromStrings(const StringArrangements &arrangements);

/// Returns a boolean vector indicating whether the given arrangement is terminal.
std::vector<bool> IsTerminalArrangement(const StringArrangements &arrangement);

/// Generates arrangements (in string format) corresponding to the given tensor.
/// Characters are given in the range `A..M` according to the table in `stratego.h`.
///
/// This method panics if the arrangements are not valid (e.g., they contain invalid
/// piece types).
std::vector<std::string> ArrangementStringsFromTensor(torch::Tensor arrangements);

/// This function constructs initialization boards (suitable for passing to
/// `StrategoEnv`) given two lists of (possibly equal) piece arrangements.
///
/// The two lists need to have the same length, and they must have matching piece and
/// piece-type counts. This needs to be true both within the lists and across the lists.
/// The function will panic if a mismatch is detected.
///
/// The arrangements are assumed to be in absolute coordinates, so the blue arrangements
/// will be rotated when placed on the board.
///
/// Boards that are terminal (that is, for which player 0 cannot move) are by definition
/// not initialization boards, and produce an error.
torch::Tensor GenerateInitializationBoards(torch::Tensor red_arrangements,
                                           torch::Tensor blue_arrangements,
                                           const int32_t cuda_device = 0);

/// This function constructs a single initialization board that matches the given string.
///
/// The function expects a 100-character string of pieces encoded in absolute coordinates,
/// according to the character encoding in `stratego.h`.
///
/// Warning: this function ignores the case of the characters.
torch::Tensor InitializationBoardFromString(std::string s, const int32_t cuda_device = 0);

/// Computes, for each cell occupied by a piece of the given player,
/// the number of available directions the piece can move. The number
/// is saturated at 2.
__global__ void SaturatedNumMovementDirectionsKernel(
    uint8_t *d_out,
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint8_t for_player);
