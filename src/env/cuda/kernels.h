#pragma once

#include "src/env/rules/twosquare_state.h"
#include "src/env/stratego_board.h"
#include "src/util.h"

//--- action_kernels.cu ---//

__global__ void LegalActionsMaskKernel(
    bool *d_out,
    const uint8_t *d_terminated_since,
    const uint32_t num_envs,
    const uint8_t for_player,
    const StrategoBoard *d_boards,
    const bool handle_terminated);

__global__ void ApplyActionsKernel(
    StrategoBoard *d_boards,
    int32_t *d_num_moves,
    int32_t *d_num_moves_since_last_attack,
    int32_t *d_num_moves_since_reset,
    uint8_t *d_flag_captured,
    uint8_t *d_move_summary_out,
    uint8_t *d_red_death,  // Cell where  red dies or 0xff
    uint8_t *d_blue_death, // Cell where blue dies or 0xff
    const uint8_t *d_terminated_since,
    const uint32_t num_envs,
    const uint8_t player, // the acting player
    const int32_t *d_actions);

//--- board_kernels.cu ---//

__global__ void ComputeIsUnknownPieceKernel(
    bool *d_out,
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint8_t player);

__global__ void ComputePieceTypeOnehotKernel(
    bool *d_out,
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint8_t player);

__global__ void ComputeUnknownPieceTypeOnehotKernel(
    bool *d_out,
    const uint8_t *unknown_ranks, // In RELATIVE coordinates
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint32_t max_k,
    const uint8_t player);

__global__ void ComputeUnknownPieceHasMovedKernel(
    bool *d_out,
    const uint8_t *unknown_ranks, // In RELATIVE coordinates
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint32_t max_k,
    const uint8_t player);

__global__ void ComputeUnknownPiecePositionOnehotKernel(
    bool *d_out,
    const uint8_t *unknown_ranks, // In RELATIVE coordinates
    const uint32_t num_envs,
    const uint32_t max_k,
    const uint8_t player);

__global__ void AssignBoardPiecesKernel(
    uint8_t *d_boards,
    const int32_t *d_id_to_type, // piece id to piece type
    const uint32_t num_boards,
    const uint32_t num_envs,
    const uint8_t opponent);

//--- infostate_kernels.cu ---//
__global__ void InjectInfostateSrcDstKernel(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t *d_action_history,
    const int32_t *d_action_prehistory, // nullptr if no state prehistory is available
    const uint8_t *d_terminated_since,
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs,
    const uint32_t INFOSTATE_STRIDE);

__global__ void InjectInfostateHiddenAndTypesKernel(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const StrategoBoard *d_board_history,
    const StrategoBoard *d_board_prehistory, // nullptr if no state prehistory is available
    const uint8_t *d_terminated_since,
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint8_t player, // The acting player at the given `step`
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs,
    const uint32_t INFOSTATE_STRIDE);

__global__ void InjectInfostateDmKernel(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t *d_action_history,
    const int32_t *d_action_prehistory, // nullptr if no state prehistory is available
    const StrategoBoard *d_board_history,
    const StrategoBoard *d_board_prehistory, // nullptr if no state prehistory is available
    const uint8_t *d_terminated_since,
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint8_t player, // The acting player at the given `step`
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs,
    const uint32_t INFOSTATE_STRIDE);

__global__ void BoardStateKernel__OwnPieceTypes(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player, // The player whose point of view we care about
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE);

__global__ void BoardStateKernel__ProbTypes(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player, // The player whose point of view we care about
    const bool rotate,        // Perform an extra rotation of the board
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE,
    const uint32_t CHANNEL_SHIFT // Typically 1200 or 2400
);

__global__ void BoardStateKernel__InvisiblesEmptyAndMoved(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const int32_t *num_moves,
    const int32_t *num_moves_since_last_attack,
    const uint32_t max_num_moves,
    const uint32_t max_num_moves_between_attacks,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE);

__global__ void BoardStateKernel__ThreatEvadeActiveAdj(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE);

__global__ void BoardStateKernel__Deaths(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const StrategoBoard *d_zero_boards,
    const uint32_t INFOSTATE_STRIDE);

__global__ void BoardStateKernel__DeathReasons(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE);

__global__ void BoardStateKernel__Protections(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE);

//--- snapshot_kernels.cu ---//

__global__ void SnapshotActionHistoryKernel(
    int32_t *d_out,
    const int32_t *d_action_history,
    const int32_t *d_action_prehistory, // nullptr if no state prehistory is available
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs);

__global__ void SnapshotMoveSummaryHistoryKernel(
    uint8_t *d_out,
    const uint8_t *d_move_summary_history,
    const uint8_t *d_move_summary_prehistory, // nullptr if no state prehistory is available
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint8_t player, // The acting player at the given `step`
    const bool relativize,
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs);

__global__ void SnapshotBoardHistoryKernel(
    StrategoBoard *d_out,
    const StrategoBoard *d_boards,
    const StrategoBoard *d_board_prehistory, // nullptr if no state prehistory is available
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs);

//--- termination_kernels.cu ---//
__global__ void InitBoardsKernel(
    StrategoBoard *d_boards,
    const StrategoBoard *init,
    const int32_t *init_offset_red,
    const int32_t *init_offset_blue,
    const uint32_t num_envs,
    const bool make_pieces_visible);

__global__ void ResetTerminatedBoardsKernel(
    StrategoBoard *d_boards,
    StrategoBoard *d_zero_boards,
    int32_t *d_num_moves,
    int32_t *d_num_moves_since_last_attack,
    int32_t *d_num_moves_since_reset,
    uint8_t *d_terminated_since,
    uint8_t *d_flag_captured,
    TwosquareState *d_twosquare_state_red,
    TwosquareState *d_twosquare_state_blue,
    const StrategoBoard *init,
    const StrategoBoard *zero_init,
    const int32_t *init_offset_red,
    const int32_t *init_offset_blue,
    const uint32_t num_envs,
    const bool make_pieces_visible,
    const int32_t *d_reset_state_num_moves,                   // `nullptr` if `!reset_state`
    const int32_t *d_reset_state_num_moves_since_last_attack, // `nullptr` if `!reset_state`
    const uint8_t *d_reset_state_terminated_since,            // `nullptr` if `!reset_state`
    const uint8_t *d_reset_state_flag_captured,               // `nullptr` if `!reset_state`
    const TwosquareState *d_twosquare_state_init_red,
    const TwosquareState *d_twosquare_state_init_blue);

__global__ void ResetTerminationCountersKernel(
    uint8_t *d_terminated_since,
    const uint8_t *d_reset_state_terminated_since, // `nullptr` if `!reset_state`
    const uint32_t num_envs);

__global__ void IncrementTerminationCounterKernel(
    uint8_t *d_terminated_since,
    uint8_t *d_flag_captured,
    int32_t *d_num_moves,
    int32_t *d_num_moves_since_last_attack,
    int32_t *d_num_moves_since_reset,
    const uint8_t *d_has_legal_movement,
    const uint32_t num_envs,
    const uint32_t max_num_moves,
    const uint32_t max_num_moves_between_attacks);

__global__ void ComputeRewardPl0Kernel(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_reward_pl0,
    const uint8_t *d_terminated_since,
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_last_attack,
    const uint8_t *d_flag_captured,
    const uint8_t *d_has_legal_movement,
    const uint32_t num_envs,
    const uint32_t max_num_moves,
    const uint32_t max_num_moves_between_attacks,
    const uint8_t to_play);