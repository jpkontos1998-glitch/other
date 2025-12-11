#pragma once

#include "src/env/env_state.h"
#include "src/env/game_saver.h"
#include "src/env/rules/chase_state.h"
#include "src/env/rules/twosquare_state.h"
#include "src/env/stratego_board.h"
#include "src/env/stratego_conf.h"

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <utility>
#include <vector>

/// Number of action. Each channel has the size of a board (10 x 10) representing which
/// player pieces can perform that action
///
/// Channel ordering:
/// - [0.. 8] VERTICAL   movement setting row to 0, 1, ..., 9 respectively (excluding piece row)
/// - [9..17] HORIZONTAL movement setting col to 0, 1, ..., 9 respectively (excluding piece col)
/// Only the scout can move by more than +/- 1
///
/// The action encoding is always from the point of view of the acting player.
const uint32_t NUM_ACTIONS = 1800;

/// Number of channels for each env. Each channels has the size of a board (10 x 10)
/// Channel ordering:
/// - Ch. 0 .. 11: Piece types for current player's pieces (only SPY .. BOMB)
/// - Ch. 12.. 23: Probability of each cell being of `channel` type for each of the opponent's pieces.
///                The ordering is (SPY .. BOMB opponent player)
/// - Ch. 24.. 35: Posterior for opponent over current player's piece types (same as previous 12
///                channels but from opponent's point of view)
/// - Ch. 36     : Which of the player's own pieces are hidden (0.0 = false, 1.0 = true)
/// - Ch. 37     : Which of the player's opponent's pieces are hidden (0.0 = false, 1.0 = true)
/// - Ch. 38     : Which of the cells are empty (0.0 = false, 1.0 = true)
/// - Ch. 39     : Which own pieces have moved
/// - Ch. 40     : Which opponent pieces have moved
/// - Ch. 41     : Proportion of `max_num_moves` exhausted (constant plane)
/// - Ch. 42     : Proportion of `max_num_moves_between_attacks` exhausted (constant plane)
/// - Ch. 43-53  : Our piece threatenend their piece of type X (X < FLAG, or X = UNKNOWN)
/// - Ch. 54-64  : Our piece evaded from their piece of type X (X < FLAG, or X = UNKNOWN)
/// - Ch. 65-75  : Our piece was actively adjacent to their piece of type X (X < FLAG, or X = UNKNOWN)
/// - Ch. 76-86  : Their piece threatenend our piece of type X (X < FLAG, or X = UNKNOWN)
/// - Ch. 87-97  : Their piece evaded from our piece of type X (X < FLAG, or X = UNKNOWN)
/// - Ch. 98-108 : Their piece was actively adjacent to our piece of type X (X < FLAG, or X = UNKNOWN)
/// - Ch. 109-119: Our dead pieces, reported in their starting positions, reported by type (SPY..MARSHAL, BOMB)
/// - Ch. 120-130: Their dead pieces, reported in their starting positions, reported by type (SPY..MARSHAL, BOMB)
/// - Ch. 131-190: Our death reason (one plane per piece type SPY..MARSHAL)
/// - Ch. 191-250: Their death reason (one plane per piece type SPY..MARSHAL)
/// - Ch. 251-263: Our protected (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
/// - Ch. 264-276: Our protected_against (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
/// - Ch. 277-289: Our was_protected_by (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
/// - Ch. 290-302: Our was_protected_against (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
/// - Ch. 303-315: Their protected (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
/// - Ch. 316-328: Their protected_against (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
/// - Ch. 329-341: Their was_protected_by (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
/// - Ch. 342-354: Their was_protected_against (one plane per piece type SPY..MARSHAL, BOMB, EMPTY, UNKNOWN)
const uint32_t NUM_BOARD_STATE_CHANNELS = 355;

/// Size of infostate representation per environment
const uint32_t BOARD_STATE_TENSOR_LEN = NUM_BOARD_STATE_CHANNELS * 100;

class StrategoRolloutBuffer
{
public:
    uint32_t buf_size;
    uint32_t num_envs;
    StrategoConf conf;

    uint32_t NUM_INFOSTATE_CHANNELS;
    std::vector<std::string> INFOSTATE_CHANNEL_DESCRIPTION; // One string per infostate channel

    torch::Tensor board_state_tensor;          // shape: (num_envs, NUM_BOARD_STATE_CHANNELS, 10, 10)  | dtype: MUSTRATEGO_FLOAT_CUDA_DTYPE
    torch::Tensor move_summary_history_tensor; // shape: (move_memory, num_envs, 6)                    | dtype: byte
    torch::Tensor infostate_tensor;            // shape: (num_envs, INFOSTATE_CHANNELS, 10, 10)        | dtype: MUSTRATEGO_FLOAT_CUDA_DTYPE
    torch::Tensor legal_action_mask;           // shape: (num_envs, NUM_ACTIONS)                       | dtype: bool
    torch::Tensor reward_pl0;                  // shape: (num_envs)                                    | dtype: MUSTRATEGO_FLOAT_CUDA_DTYPE
    torch::Tensor is_unknown_piece;            // shape: (num_envs, 10, 10)                            | dtype: bool
    torch::Tensor piece_type_onehot;           // shape: (num_envs, 10, 10, NUM_PIECE_TYPES)           | dtype: bool
    torch::Tensor two_square_rule_applies;     // shape: (num_envs)                                    | dtype: bool
    // The special marker `-1` denotes an empty entry in `illegal_chase_actions`.

    // The following tensors can change shape (due to the `max_k` argument in the methods that compute them).
    // Their storage is dynamically allocated below.
    torch::Tensor unknown_piece_type_onehot;     // shape: (num_envs, K<=40, NUM_PIECE_TYPES)   | dtype: bool
    torch::Tensor unknown_piece_has_moved;       // shape: (num_envs, K<=40)                    | dtype: bool
    torch::Tensor unknown_piece_position_onehot; // shape: (num_envs, K<=40, 100)               | dtype: bool

public:
    StrategoRolloutBuffer(const uint32_t buf_size, const uint32_t num_envs, const StrategoConf stratego_conf = StrategoConf{});
    ~StrategoRolloutBuffer();

    void SaveGames(const std::string &outfile);
    void StopSavingGames();

    void ChangeResetBehavior(const ResetBehavior reset_behavior,
                             const std::optional<std::pair<torch::Tensor, torch::Tensor>> &initial_arrangements_distrib = std::nullopt);
    void ChangeResetBehavior(const EnvState &reset_state);

    /// Instructs the rollout buffer to use the given initial arrangements.
    ///
    /// When `randomize` is set to `true`, the initial arrangements are shuffled
    /// before being assigned to the environments. The distribution of the initial
    /// arrangements is controlled by the optional `initial_arrangements_distrib`
    /// argument. If it is not provided, the initial arrangements are distributed
    /// uniformly at random.
    ///
    /// When `randomize` is set to `false`, the initial arrangements are assigned
    /// to the environments in the order they are provided and `initial_arrangements_distrib`
    /// must be set to nullopt. Environment 0 will receive arrangement 0, num_envs, 2 * num_envs, etc.
    /// modulo the number of available environments.
    void ChangeResetBehavior(const std::pair<StringArrangements, StringArrangements> &initial_arrangements,
                             const std::optional<std::pair<torch::Tensor, torch::Tensor>> &initial_arrangements_distrib = std::nullopt,
                             const bool randomize = true,
                             const bool fullinfo = false);

    /// Resets the current row id to 0, and initializes all environments to the
    /// initial position.
    void Reset();

    uint64_t CurrentStep() const { return current_step_; }
    /// Returns the player (0 = red, 1 = blue) acting at the given step.
    uint8_t ActingPlayer(const uint64_t step) const;
    uint8_t CurrentPlayer() const { return to_play_[CurrentRowId_()] - 1; }

    /// Update the board states given the actions.
    ///
    /// This operation spawns one thread per board.
    ///
    /// ## Auto-reset behavior
    ///
    /// Case 1: Pl0 terminates the game (e.g., wins by stealing Pl1's flag)
    /// ------
    ///
    ///   Pl0 "calls" ApplyActions(), terminates the game
    ///   In `out`: terminated_since = 1, CurrentPlayer() = 1, rewards_pl0 = +1
    ///   The infostate tensor produced by `out` contains data for Pl1
    ///
    ///   Pl1 "calls" ApplyActions()
    ///   In `out`: terminated_since = 2, CurrentPlayer() = 0, rewards_pl0 = +1
    ///   The infostate tensor produced by `out` contains data for Pl0
    ///
    ///   Pl0 "calls" ApplyActions()
    ///   In `out`: terminated_since = 3, CurrentPlayer() = 1, rewards_pl0 = +1
    ///   The infostate tensor produced by `out` contains data for Pl1
    ///
    ///   Pl1 "calls" ApplyActions()
    ///   ENV IS RESET
    ///   In `out`: terminated_since = 0, CurrentPlayer() = 0, rewards_pl0 = 0
    ///
    ///
    /// Case 2: Pl1 terminates the game (e.g., wins by stealing Pl0's flag)
    /// ------
    ///
    ///    Pl1 "calls" ApplyActions(), terminates the game
    ///    In `out`: terminated_since = 1, CurrentPlayer() = 0, rewards_pl0 = -1
    ///    The infostate tensor produced by `out` contains data for Pl0
    ///
    ///    Pl0 "calls" ApplyActions()
    ///    In `out`: terminated_since = 2, CurrentPlayer() = 1, rewards_pl0 = -1
    ///    The infostate tensor produced by `out` contains data for Pl1
    ///
    ///    Pl1 "calls" ApplyActions()
    ///    ENV IS RESET
    ///    In `out`: terminated_since = 0, CurrentPlayer() = 0, rewards_pl0 = 0.
    ///
    ///
    ///
    /// ## Return value
    ///
    /// Returns the row id where the new boards have been written.
    uint64_t ApplyActions(const torch::Tensor actions);

    void SeedActionSampler(const uint64_t seed);
    void SampleRandomLegalAction(torch::Tensor actions_out);
    void SampleFirstLegalAction(torch::Tensor actions_out);

    void ComputeLegalActionMask(const uint64_t step);

    /// The move summary tensor records information about the pieces that
    /// moved. For each time within `conf.move_memory`, and each env, it
    /// stores 6 bytes in the following order:
    ///
    /// - Byte 0: source cell, as a position in [0, 99]. Relative to the pov
    ///   of the acting player at `step`. 100 is used for padding.
    /// - Byte 1: dest. cell, as a position in [0, 99]. Relative to the pov
    ///   of the acting player at `step`. 100 is used for padding.
    /// - Byte 2: source (moving) piece. See below for encoding. 100 is used for padding.
    /// - Byte 3: dest. piece. See below for encoding. 100 is used for padding.
    /// - Byte 4: source piece ID as a position in [0, 40]. Relative to the pov
    ///   of the acting player at `step`. 255 is used for empty cells. 100 is used for padding.
    /// - Byte 5: dest. piece ID as a position in [0, 40]. Relative to the pov
    ///   of the acting player at `step`. 255 is used for empty cells. 100 is used for padding.
    ///
    /// For moves that are NOT attacks, bytes 2 and 3 are set to EMPTY, and bytes 4 and 5 to 255.
    ///
    /// Piece encoding:
    /// - The piece type is encoded in the lowest 4 bits (bits 0..3).
    ///   The piece type encoding reflects the knowledge of the player at the
    ///   end of the move. If 1) the piece was visible prior to the turn, 2) the
    ///   piece participated in a battle on the turn, or 3) the piece played a
    ///   scout move on the turn, the piece type is encoded as (0-13) according to the
    ///   ordering in `stratego_board.h` (SPY = 0, ..., EMPTY = 13).
    ///   Otherwise, the piece type is encoded as 15 (HIDDEN_PIECE).
    /// - Bit 4 is 1 if the piece was visible prior to turn, 0 otherwise.
    /// - Bit 5 is 1 if the piece had moved prior to turn, 0 otherwise.
    ///
    /// In particular, unlike `Piece`, no bits related to the piece color are
    /// stored.
    void ComputeMoveSummaryHistoryTensor(const uint64_t step);

    /// The infostate tensor is the concatenation of the board state tensor
    ///  together with a representation of the past `conf.move_memory` actions of both players.
    ///
    /// The shape is {num_envs, num_channels, 10, 10} floats.
    ///
    /// The representation of the history of past moves is composed of multiple
    /// channels:
    /// - -1 for the cell source and +1 for cell destination of movement
    /// - our hidden (just before the move)
    /// - their hidden (just before the move)
    /// - our piece types, in the 1/K format, just before the move
    ///   (-1.0 if the piece does not belong to us.)
    /// - their piece types for the visible pieces, in the 1/K format (as laid out
    ///   in `stratego_board.h`) just before the move.
    ///   (-1.0 if the piece is not visible or does not belong to them.)
    /// - Deepmind plane:
    ///      -1 at src cell if this was not an attack
    ///      -(2+t/12) if this was an attack, where t is the piece type
    ///      +1 for destination cell
    ///
    /// As usual, the boards are encoded from the point of view of the acting player
    /// at the `step` given as input to `ComputeInfostateTensor`.
    ///
    /// To programmatically extract the meaning of each layer of the infostate tensor, you
    /// can access the `INFOSTATE_CHANNEL_DESCRIPTION` member.
    void ComputeInfostateTensor(const uint64_t step);

    /// Computes the payoff for player 0 (red).
    /// The payoff is 0 unless the game is over.
    void ComputeRewardPl0(const uint64_t step);

    /// Marks unknown pieces from the point of view of the acting player.
    void ComputeIsUnknownPiece(const uint64_t step);

    /// Returns a one-hot encoding of all the pieces on the board, from the
    /// point of view of the acting player.
    void ComputePieceTypeOnehot(const uint64_t step);

    /// Returns whether the two-square rule applies in each env to the next-acting
    /// player.
    void ComputeTwoSquareRuleApplies(const uint64_t step);

    /// Computes a one-hot encoded tensor of shape [num_envs, max_k, NUM_PIECE_TYPES]
    ///
    /// Dimension 1 corresponds to the index of the unknown piece (in the natural row-major
    /// scan of the board from the point of view of the acting player).
    void ComputeUnknownPieceTypeOnehot(const uint64_t step, const uint32_t max_k = 40);

    /// Computes a tensor of shape [num_envs, max_k] encoding whether the kth unknown piece of
    /// each environment has moved.
    ///
    /// Dimension 1 corresponds to the index of the unknown piece (in the natural row-major
    /// scan of the board from the point of view of the acting player).
    void ComputeUnknownPieceHasMoved(const uint64_t step, const uint32_t max_k = 40);

    /// Computes a one-hot encoded tensor of shape [num_envs, max_k, 100]
    ///
    /// Dimension 1 corresponds to the index of the unknown piece (in the natural row-major
    /// scan of the board from the point of view of the acting player).
    void ComputeUnknownPiecePositionOnehot(const uint64_t step, const uint32_t max_k = 40);

    torch::Tensor GetTerminatedSince(const uint64_t step) const;
    torch::Tensor GetHasLegalMovement(const uint64_t step) const;

    /// Information about captured flags. The output tensor is of dtype uint8,
    /// and contains the following information for each environment:
    /// - 0 = no flag has been captured
    /// - 1 = red player has captured blue flag
    /// - 2 = blue player has captured red flag
    torch::Tensor GetFlagCaptured(const uint64_t step) const;

    /// To support custom initial states that might represent boards in which
    /// the players have been already playing for a while, we keep two counters:
    /// - `num_moves` represents the logical counter in the game. So, if the
    ///       reset board already contains a history, num_moves of the reset
    ///       board will start from such a value.
    /// - `num_moves_since_reset` instead only counts actions that have been
    ///       taken in the rollout buffer.
    ///
    /// It always holds that num_moves_since_reset + reset_state.num_moves = num_moves.
    torch::Tensor GetNumMoves(const uint64_t step) const;
    torch::Tensor GetNumMovesSinceLastAttack(const uint64_t step) const;
    torch::Tensor GetNumMovesSinceReset(const uint64_t step) const;
    torch::Tensor GetNumMovesSinceResetTensor() const;

    /// Returns a view into the current action history tensor.
    torch::Tensor GetActionHistoryTensor() const;

    torch::Tensor GetPlayedActions(const uint64_t step) const;

    /// Returns the move summary for all envs at the given step.
    ///
    /// The return tensor has shape `(num_envs, 6)`.
    ///
    /// See description of `ComputeMoveSummaryHistoryTensor` for the
    /// encoding of the tensor.
    torch::Tensor GetMoveSummary(const uint64_t step) const;

    /// Returns a representation of the board as a uint8_t tensor of size
    /// `(num_envs, 1920)`, with the following semantics:
    ///
    /// - Bytes 0..1599: description of the pieces on each cell of the board.
    ///   The piece order is ABSOLUTE, i.e., not relativized per player.
    ///   Each cell is encoded using 16 consecutive bytes:
    ///   * The first byte denotes the piece type, color, visibility, and
    ///     has_moved bits.
    ///   * The second byte is the piece ID (the special value 0xff=255 denotes
    ///     a non-player piece, such as an empty cell or a lake).
    /// - Bytes 1600..1611: piece counts for player 0 (SPY .. BOMB)
    /// - Bytes 1612..1623: piece counts for player 1 (SPY .. BOMB)
    ///
    /// For more information about the encoding, see `stratego_board.h`.
    torch::Tensor GetBoardTensor(const uint64_t step) const;

    /// Same as above, but it dumps the entire internal tensor.
    /// This is extremely low-level and should be used only for debugging or advanced
    /// bookkeeping.
    torch::Tensor GetBoardTensor() const;

    /// Same as `GetBoardTensor(const uint64_t step)`, but for the zero board.
    torch::Tensor GetZeroBoardTensor(const uint64_t step) const;

    /// Returns the state of the two-square state machine for both the red and blue player.
    ///
    /// This is not meant to be user-facing but it can be helpful in debugging.
    std::array<torch::Tensor, 2> GetTwosquareState(const uint64_t step) const;

    /// Returns a tensor of shape `(num_envs, 2)` indicating the illegal chase actions
    /// for each environment.
    /// The special value -1 denotes a missing entry.
    ///
    /// The environment will abort if the chase rule is not enforced,
    /// that is, `conf.continuous_chasing_rule = false`.
    torch::Tensor GetIllegalChaseActions(const uint64_t step) const;

    std::pair<EnvState, EnvState> SnapshotEnvHistory(const uint64_t step, const int64_t env_idx) const;
    EnvState SnapshotState(const uint64_t step) const;

    /// Returns the state of the board at the given `row_id`.
    /// Each string is of size 200, in row-major format starting from row 0.
    ///
    /// Note: unlike the tensors, the representation of the board is not "flipped"
    ///    to account for the point of view of the acting player.
    ///
    /// Each piece is encoded according to the following schema:
    ///
    /// Colored pieces
    /// ==============
    ///
    /// | Piece       |  red  |  blue |
    /// +-------------+-------+-------+
    /// | SPY         |   C   |   O   |
    /// | SCOUT       |   D   |   P   |
    /// | MINER       |   E   |   Q   |
    /// | SERGEANT    |   F   |   R   |
    /// | LIEUTENANT  |   G   |   S   |
    /// | CAPTAIN     |   H   |   T   |
    /// | MAJOR       |   I   |   U   |
    /// | COLONEL     |   J   |   V   |
    /// | GENERAL     |   K   |   W   |
    /// | MARSHAL     |   L   |   X   |
    /// | FLAG        |   M   |   Y   |
    /// | BOMB        |   B   |   N   |
    /// +-------------+-------+-------+
    ///
    /// Inert pieces
    /// ============
    ///
    /// | Piece       |     |
    /// +-------------+-----+
    /// | LAKE        |  _  |
    /// | EMPTY       |  a  |
    /// +-------------+-----+
    ///
    /// A lowercase letter denotes the fact that the piece is visible.
    ///
    /// After each piece character, the symbol `@` appears to mean that the piece
    /// has never moved. If the piece has moved, `.` is used.
    std::vector<std::string> BoardStrs(const uint64_t step) const;

    /// Same as `BoardStrs`, but applied to the zero board, that is, the board that was
    /// installed in each environment at reset time.
    std::vector<std::string> ZeroBoardStrs(const uint64_t step) const;

private:
    inline uint32_t CurrentRowId_() const { return current_step_ % buf_size; }
    void RandomizeBoardInitOffsets_();
    void PopulateInfostateChannelDescription_();
    void ComputeIsUnknownPiece_(const uint64_t step, bool *d_out);
    void ComputeBoardStateTensor_(const uint64_t step, MUSTRATEGO_FLOAT_CUDA_DTYPE *out);
    void UpdateHasLegalMovement_(const uint64_t step);
    void ComputeLegalActionMask_(const uint64_t step, bool *d_out, const bool handle_terminated = true);

    std::optional<GameSaver> game_saver_;

    StrategoBoard *d_boards_;
    StrategoBoard *d_zero_boards_;

    int32_t *d_num_moves_;
    int32_t *d_num_moves_since_last_attack_;
    int32_t *d_num_moves_since_reset_;

    uint8_t *d_terminated_since_;
    uint8_t *d_flag_captured_;      // 0 = no, 1 = red has captured, 2 = blue has captured
    uint8_t *d_has_legal_movement_; // 0 = neither, 1 = only red, 2 = only blue, 3 = both

    // Two-square state machines
    TwosquareState *d_twosquare_state_red_;
    TwosquareState *d_init_twosquare_state_red_;
    TwosquareState *d_twosquare_state_blue_;
    TwosquareState *d_init_twosquare_state_blue_;

    // Continuous chase state machines
    ChaseState d_chase_state_;                     // Owns memory
    std::optional<ChaseState> d_init_chase_state_; // Only a borrow
    int32_t *d_illegal_chase_actions_;             // MAX_CHASE_LENGTH ints per env

    // Scratch space for intermediate computations.
    // Scratchpad of size at least num_envs * (NUM_ACTIONS + 2).
    uint8_t *d_scratch_;

    // These provide the actual memory allocation for the similarly-named tensors.
    // The reason why we allocate these here instead of being tensors is due to the fact that
    // the tensor views require changing size.
    bool *d_unknown_piece_type_onehot_;
    bool *d_unknown_piece_has_moved_;
    bool *d_unknown_piece_position_onehot_;

    int32_t *d_action_history_;
    uint8_t *d_move_summary_history_;

    std::vector<uint32_t> to_play_; // 1 = red, 2 = blue.

    // This is the *absolute* row id, that is, before the modulus.
    // To figure out the row_id in the circular buffer for, e.g., num_moves,
    // it is enough to take the absolute row id modulus `buf_size`.
    // For the row id of the action history, the modulus is
    // `buf_size + conf.move_memory`.
    uint64_t current_step_;

    // Shared distribution object, used for sampling actions or initial states.
    torch::Generator gen_;
    torch::Tensor distrib_;

    // This is used to make sure all selected actions are legal (if compiled in DEBUG mode),
    // and to sample actions via `SampleRandomLegalAction` and `SampleFirstLegalAction`.
    torch::Tensor private_action_mask_;

    // Initialization mechanism
    // ========================
    uint32_t red_init_modulus_;
    uint32_t blue_init_modulus_;
    torch::Tensor init_boards_;      // dtype: uint8_t, shape: (*, sizeof(StrategoBoard))
    torch::Tensor init_zero_boards_; // dtype: uint8_t, shape: (*, sizeof(StrategoBoard))
    torch::Tensor init_offset_red_;  // dtype: int32_t, shape: (buf_size, num_envs)
    torch::Tensor init_offset_blue_; // dtype: int32_t, shape: (buf_size, num_envs)
};

/// This utility function converts from a tensor of actions for the environment,
/// to a list (from, to) of ABSOLUTE movement coordinates. The second argument specifies the
/// players from whose point of view the actions are encoded.
///
/// Each coordinate is a number in the range [0, 100), identifying the cell
/// (for example, 42 is row 4, column 2).
std::vector<std::pair<uint8_t, uint8_t>> ActionsToAbsCoordinates(torch::Tensor actions, const uint8_t player);

/// This utility function converts from a list of (from, to) ABSOLUTE movement coordinates to
/// a host-allocated list of action ids for the given `player`.
///
/// Each coordinate must be a number in the range [0, 100), identifying the cell
/// (for example, 42 is row 4, column 2).
std::vector<int32_t> AbsCoordinatesToActions(const std::vector<std::pair<uint8_t, uint8_t>> movements, const uint8_t player);
