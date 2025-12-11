#pragma once

#include "src/env/stratego_board.h"
#include "src/util.h"
#include <cstdint>
#include <optional>

// Needs to be int32 and not uint32 because of comparison to the -1 marker
const int32_t MAX_CHASE_LENGTH = 210;

/// State machine for checking whether the continuous chasing rule is
/// triggered.
struct ChaseState
{
    uint8_t *last_dst_pos[2]; // color -> env -> last move destination cell (absolute)
                              // Can be 0xee for missing values.
    uint8_t *last_src_pos[2]; // color -> env -> last move source cell (absolute)
                              // Can be 0xee for missing values.
    int32_t *chase_length[2]; // Length of the chase so far (color -> env -> length).
};

/// Resets the state to the given source.
///
/// When the given mask is nonzero, the update is skipped. If the mask is not given,
/// all environments are reset.
///
/// All tensors are expected to be on the same device as `cuda_device` passed at
/// construction time for the state machine.
void ResetChaseState(ChaseState d_out,
                     const std::optional<ChaseState> &d_src,
                     const uint32_t num_envs,
                     const int32_t cuda_device,
                     const int32_t *d_mask = nullptr);

/// Updates the continuous chase state given new actions.
///
/// All tensors are expected to be on the same device as `cuda_device` passed at
/// construction time for the state machine.
///
/// ## State machine update
///
/// The move summary is used to update `last_dst_pos`, `last_src_pos`, and
/// `chase_length`, according to the following workflow for each env.
/// Let p being the acting player and ~p the opponent.
///
///      i. Set last_src_pos[p] and last_dst_pos[p] to the source and destination
///         cells of the move, respectively. (Do nothing with last_src_pos[~p] and
///         last_dst_pos[~p].)
///     ii. If p is moving from a cell that is adjacent to last_dst_pos[~p],
///         increment chase_length[~p], else set it to 0. (Do nothing with chase_length[p].)
///    iii. If p is attacking, set chase_length[p] = 0 and chase_length[~p] = 0.
///     iv. If p is moving to a destination cell adjacent to an opponent,
///         increment chase_length[p], else set it to 0. (Do nothing with chase_length[~p].)

void UpdateChaseState(
    ChaseState d_state,
    const StrategoBoard *d_board,
    const uint8_t *d_move_summary, // relativized to player
    const uint8_t *d_mask,         // update skipped if mask is nonzero
    const uint8_t player,          // 1 = red, 2 = blue
    const uint32_t num_envs,
    const int32_t cuda_device);

/// This function writes on a tensor of size (MAX_CHASE_LENGTH, num_envs).
///
/// The special marker -1 is used to indicate a missing value.
///
/// To compute the illegal chase moves for the active player p, we need to check the value of
/// `chase_length[p]`.
void ComputeIllegalChaseMoves(
    int32_t *d_out,
    const ChaseState d_state,
    const StrategoBoard *d_board_history,
    const StrategoBoard *d_board_prehistory, // can be nullptr
    const int32_t *d_num_moves_since_reset,
    const uint8_t *d_mask,
    const uint32_t current_step,       // Current index of board
    const uint32_t history_buf_length, // Circular buffer length
    const uint8_t player,              // 1 = red, 2 = blue
    const uint32_t prehistory_size,
    const uint32_t num_envs,
    const int32_t cuda_device);

void RemoveIllegalChaseMoves(
    bool *d_legal_action_mask,
    const int32_t *d_illegal_chase_actions,
    const uint8_t *d_mask,
    const uint32_t num_envs,
    const int32_t cuda_device);
