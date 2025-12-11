#pragma once

#include <cstdint>
#include <optional>
#include <torch/torch.h>

#include "src/env/env_state.h"
#include "src/util.h"

// Last bit is board type              (0 = initial, 1 = env_state)
// Second to last bit is randomization (1 = random, 0 = deterministic/step-through)
// Third to last is full information   (0 = no, 1 = yes)
// The other bits identify the board variant (00 = classic, 01 = barrage, 11 = generic)
// clang-format off
enum ResetBehavior : uint8_t
{
    CUSTOM_ENV_STATE                           = 0b00011001,
    RANDOM_CUSTOM_INITIAL_ARRANGEMENT          = 0b00011010,
    STEP_CUSTOM_INITIAL_ARRANGEMENT            = 0b00011000,
    FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT = 0b00011110,
    FULLINFO_STEP_CUSTOM_INITIAL_ARRANGEMENT   = 0b00011100,

    // Special cases of RANDOM_CUSTOM_INITIAL_ARRANGEMENT
    RANDOM_JB_CLASSIC_BOARD                    = 0b00000010,
    RANDOM_JB_BARRAGE_BOARD                    = 0b00001010,
    
    // Special cases of FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT
    FULLINFO_RANDOM_JB_CLASSIC_BOARD           = 0b00000110,
    FULLINFO_RANDOM_JB_BARRAGE_BOARD           = 0b00001110,
};
// clang-format on

const char *ResetBhToString(const ResetBehavior bh);
inline bool ResetBhIsClassicBoard(const ResetBehavior bh) { return (bh & 24) == 0b00000000; }
inline bool ResetBhIsBarrageBoard(const ResetBehavior bh) { return (bh & 24) == 0b00001000; }
inline bool ResetBhIsGenericBoard(const ResetBehavior bh) { return (bh & 24) == 0b00011000; }
inline bool ResetBhIsRandomizedBoard(const ResetBehavior bh) { return bh & 2; }
inline bool ResetBhIsFullinfo(const ResetBehavior bh) { return bh & 4; }
struct StrategoConf
{
    /// Maximum number of moves before the game is considered over.
    /// The move counter in the game is incremented when any of the players makes a move.
    uint32_t max_num_moves = 2000;

    /// Maximum number of consecutive moves since an attack before the game is considered over.
    /// The move counter in the game is incremented when any of the players makes a move.
    uint32_t max_num_moves_between_attacks = 200;

    /// How many actions (for either player) are remembered by each player and
    /// stored in the infostate tensors.
    uint32_t move_memory = 32;

    /// The reset behavior the rollout buffer should employ when a simulation is terminated.
    ResetBehavior reset_behavior = RANDOM_JB_CLASSIC_BOARD;

    /// The state to reset to when `reset_behavior` is set to `CUSTOM_ENV_STATE`.
    std::optional<EnvState> reset_state = std::nullopt;

    /// The list of initial arrangements from which to randomize when `reset_behavior` is set to
    /// `RANDOM_CUSTOM_INITIAL_ARRANGEMENT` or `FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT`.
    ///
    /// The arrangements must be such that the red player can make at least one
    /// move, that is, they cannot be already terminated.
    std::optional<std::pair<StringArrangements, StringArrangements>> initial_arrangements = std::nullopt;

    /// Probability distribution over the initial arrangements.
    ///
    /// If unset, then the uniform distribution is used. Else, the distribution needs to have the same
    /// length as the corresponding vector in `initial_arrangements`.
    ///
    /// The distributions need not be normalized.
    std::optional<std::pair<torch::Tensor, torch::Tensor>> initial_arrangements_distrib = std::nullopt;

    /// If true, DEBUG messages will be output by the environment.
    bool verbose = false;

    /// Enforces the two-square rule. The rule is implemented as follows:
    ///
    /// > No piece can cross the same cell border for more than three times in a row.
    bool two_square_rule = true;

    /// Enforces the continuous chasing rule.
    bool continuous_chasing_rule = true;

    /// Export src/dst planes in infostate tensors.
    bool enable_src_dst_planes = true;

    /// Export hidden pieces and piece types in infostate tensors.
    bool enable_hidden_and_types_planes = false;

    /// Export DeepMind planes in infostate tensors.
    bool enable_dm_planes = false;

    /// CUDA device to use
    int32_t cuda_device = 0;

    /// Hide most INFO / DEBUG messages from the env
    ///
    /// 0: show all messages
    /// 1: show INFO or higher
    /// 2: show WARNING or higher
    uint8_t quiet = 0;

    /// If true, applying actions is disabled.
    bool nonsteppable = false;
};