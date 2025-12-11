#pragma once

#include <iostream>
#include <optional>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "src/env/rules/chase_state.h"
#include "src/env/stratego_board.h"
#include "src/util.h"

/// Represents a "snapshot" in time of the environment.
///
/// The tensors are all owned.
struct EnvState
{
public:
    uint32_t num_envs;

    /// Whose turn is it to play (0 = red player, 1 = blue player)
    uint32_t to_play;

    /// Shape: `(num_envs, sizeof(StrategoBoard)=1920)`
    /// Dtype: `uint8_t`
    ///
    /// This is a byte representation (reinterpret cast to `uint8_t`).
    /// Correspondingly, the pov is ABSOLUTE.
    torch::Tensor boards;

    /// The board that was installed when the environment was reset.
    ///
    /// Shape: `(num_envs, sizeof(StrategoBoard)=1920)`
    /// Dtype: `uint8_t`
    ///
    /// This is a byte representation (reinterpret cast to `uint8_t`)
    /// Correspondingly, the pov is ABSOLUTE.
    torch::Tensor zero_boards;

    /// Shape: `(num_envs,)`
    /// Dtype: `int32_t`
    torch::Tensor num_moves;

    /// Shape: `(num_envs,)`
    /// Dtype: `int32_t`
    torch::Tensor num_moves_since_last_attack;

    /// Shape: `(num_envs,)`
    /// Dtype: `uint8_t`
    torch::Tensor terminated_since;

    /// Shape: `(num_envs,)`
    /// Dtype: `uint8_t`
    torch::Tensor has_legal_movement;

    /// Shape: `(num_envs,)`
    /// Dtype: `uint8_t`
    ///
    /// The meaning of the bytes is as follows:
    /// - `0`: no flag has been captured
    /// - `1`: the first flag that was captured was blue (captured by red)
    /// - `2`: the first flag that was captured was red (captured by blue)
    torch::Tensor flag_captured;

    /// Shape: `(move_memory, num_envs)`
    /// Dtype: `int32_t`
    ///
    /// The first axis is chronological (old -> new).
    /// Data prior to the beginning of the env is set to uninitialized
    /// values.
    ///
    /// The pov is relative to the player acting at each time.
    ///
    /// WARNING: the action history saves only the history of the ACTIVE
    ///   envs. When an env resets, the action history of that env is reset
    ///   to all zeros.
    torch::Tensor action_history;

    /// Shape: `(move_memory, num_envs, sizeof(StrategoBoard)=1920)`
    /// Dtype: `uint8_t`
    ///
    /// The first axis is chronological (old -> new).
    /// Data prior to the beginning of the env is set to uninitialized
    /// values.
    ///
    /// The pov is ABSOLUTE.
    ///
    /// WARNING: the board history saves only the history of the ACTIVE
    ///   envs. When an env resets, the board history of that env is reset
    ///   to all zeros.
    torch::Tensor board_history;

    /// Shape: `(move_memory, num_envs, 6)`
    /// Dtype: `uint8_t`
    ///
    /// This tensor summarizes the past `move_memory`.
    /// The first axis is chronological (old -> new).
    /// Data prior to the beginning of the env is set to uninitialized
    /// values.
    ///
    /// The pov of the cell locations is RELATIVE.
    ///
    /// WARNING: the board history saves only the history of the ACTIVE
    ///   envs. When an env resets, the board history of that env is reset
    ///   to all zeros.
    torch::Tensor move_summary_history;

    /// If applicable and known, the chase state machine. The chase state
    /// machines are assumed to be allocated on the same cuda device as
    /// the rest of the environment state.
    struct TensorChaseState
    {
        std::array<torch::Tensor, 2> last_dst_pos; // color -> env -> last move destination cell (absolute)
        std::array<torch::Tensor, 2> last_src_pos; // color -> env -> last move source cell (absolute)
        std::array<torch::Tensor, 2> chase_length; // Length of the chase so far (color -> env -> length).
    };
    std::optional<TensorChaseState> chase_state;

public:
    /// Makes all environment match the same state as the given environment index `idx`.
    void ReplicateEnv(const uint32_t idx);

    /// Tiles the current state. For example, if `num_tiles = 2`, then the new env state will contain
    /// twice the number of environments, with states tiled.
    void Tile(const uint32_t num_tiles);

    /// See documentation of `StrategoRolloutBuffer::BoardStrs()`.
    std::vector<std::string> BoardStrs() const;

    /// See documentation of `StrategoRolloutBuffer::ZeroBoardStrs()`.
    std::vector<std::string> ZeroBoardStrs() const;

    /// Deep copy the current state.
    EnvState Clone() const;

    /// The CUDA device on which the reset state is allocated.
    int32_t CudaDevice() const;

    /// Concatenate with another env state.
    EnvState Cat(const EnvState &other) const;

    /// Slice the env state.
    EnvState Slice(const uint32_t start, const uint32_t end) const;
};

/// Takes as input an env_state for a single env, and replaces
/// the hidden pieces of the opponent with the given tensor.
///
/// `opponent_hidden` has shape (N, 8|40, 14), denoting the
/// 1-hot encoding of the piece type for each of the hidden pieces.
/// In this context, the hidden pieces are indexed lexicographically
/// in relative row-major order.
///
/// The EnvState contains N environments, one per each assignment
/// of the opponent's pieces encoded by `opponent_pieces`.
EnvState AssignOpponentHiddenPieces(
    const EnvState &env_state,
    const torch::Tensor &opponent_hidden);

/// Legacy implementation for testing purposes.
EnvState LegacyAssignOpponentHiddenPieces(
    const EnvState &env_state,
    const torch::Tensor &opponent_hidden);