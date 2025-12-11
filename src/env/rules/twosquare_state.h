#pragma once

#include "src/env/env_state.h"
#include <cstdint>

/// State machine for checking whether the two-square rule applies to
/// a given environment. A `TwosquareState` stores the past 4 positions
/// that the piece that moved occupied.
///
/// When a different piece is moved, or the direction (horizontal/vertical)
/// of movement changes, the state machine is reset so that `C = D = 0xff`,
/// where `0xff` is a special marker to mean that the historical data is
/// not available.
///
/// Cells are encoded in RELATIVE terms.
struct TwosquareState
{
    uint8_t A; // newest cell
    uint8_t B;
    uint8_t C;
    uint8_t D; // oldest cell
};

/// Initializes the TwosquareState to a state that represents lack of
/// history. The special marker byte 0xff is used to represent missing
/// data.
void ClearTwosquareState(
    TwosquareState *d_out,
    const uint32_t num_envs);

/// Initializes the TwosquareState for both players, given an env state.
/// The env state must be allocated on the same CUDA device as the output
/// twosquare state, and it must have a move memory of at least six.
void TwosquareStateFromEnvState(
    TwosquareState *d_out_red,
    TwosquareState *d_out_blue,
    const EnvState &env_state,
    const uint32_t num_envs);

/// Updates the two-square state given new actions.
///
/// When the given mask is nonzero, the update is skipped.
void UpdateTwosquareAction(
    TwosquareState *d_state,
    const int32_t *d_actions,
    const uint8_t *d_mask,
    const uint32_t num_envs);

/// Updates the two-square state given death location.
///
/// This is needed to make sure we don't decrease the counters
/// of the number of available directions by one due to two-square
/// rule when the piece has died and would have 0 available directions
/// anyway.
///
/// When the given mask is nonzero, the update is skipped.
void UpdateTwosquareDeath(
    TwosquareState *d_state,
    const uint8_t *d_death_cell,
    const uint8_t *d_mask,
    const uint32_t num_envs);

/// Returns whether the two-square rule applies to (is activated for) any
/// piece in each of the environments.
///
/// For the scout, the two-square rule might prevent movement only after a
/// certain length. For example, consider the following sequence of positions
/// for a scout:
///
/// time 0:  | . . . D . . . . . . . . |
/// time 1:  | . . . . . . . C . . . . |
/// time 2:  | . B . . . . . . . . . . |
/// time 3:  | . . . . . . . . . . . A |
///
/// At this point, the two-square rule says that A can move left only up to C,
/// but not any further. In this case, the output of `IsTwosquareRuleTriggered`
/// will be `true`, but the output of `IsTwosquareRulePrecludingDirection` will
/// be `false`, since the scout can move in the left direction by at least one.
void IsTwosquareRuleTriggered(
    bool *d_out,
    const TwosquareState *d_state,
    const uint32_t num_envs);

/// Coincides with `IsTwoSquareRuleTriggered` for all pieces other than scouts.
/// For scouts, it returns whether the scout cannot move _by one cell_ in some
/// directions due to the two-square rule.
void IsTwosquareRulePrecludingDirection(
    bool *d_out,
    const TwosquareState *d_state,
    const uint32_t num_envs);

/// Removes actions that violate the two-square rule from the given action mask.
///
/// If d_mask > 1, the removal is skipped.
void RemoveTwosquareActions(
    bool *d_action_mask,
    const TwosquareState *d_state,
    const uint8_t *d_mask,
    const uint32_t num_envs);

torch::Tensor TwosquareStateAsTensor(
    const TwosquareState *d_state,
    const uint32_t num_envs,
    const uint32_t cuda_device);