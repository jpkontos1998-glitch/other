#include "src/env/cuda/kernels.h"

__global__ void SnapshotActionHistoryKernel(
    int32_t *d_out,
    const int32_t *d_action_history,
    const int32_t *d_action_prehistory, // nullptr if no state prehistory is available
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t env_idx = idx % num_envs;
    const uint32_t delta = 1 + (idx / num_envs);
    const uint64_t row_id = step % buf_size;

    if (delta > move_memory || delta > d_num_moves[row_id * num_envs + env_idx])
        return;

    const uint32_t moves_since_reset = d_num_moves_since_reset[row_id * num_envs + env_idx];

    uint32_t action;
    if (delta <= moves_since_reset)
    {
        assert(step >= moves_since_reset);
        const uint32_t this_thread_act_row_id = (step - delta) % buf_size;
        action = d_action_history[this_thread_act_row_id * num_envs + env_idx];
    }
    else
    {
        assert(d_action_prehistory);
        action = d_action_prehistory[(move_memory + moves_since_reset - delta) * num_envs + env_idx];
    }

    d_out[(move_memory - delta) * num_envs + env_idx] = action;
}

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
    const uint32_t num_envs)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t env_idx = idx % num_envs;
    const uint32_t delta = 1 + (idx / num_envs);
    const uint64_t row_id = step % buf_size;

    if (delta > move_memory || delta > d_num_moves[row_id * num_envs + env_idx])
    {
        return;
    }

    const uint32_t moves_since_reset = d_num_moves_since_reset[row_id * num_envs + env_idx];

    uint8_t src_cell;
    uint8_t dst_cell;
    uint8_t src_piece;
    uint8_t dst_piece;
    uint8_t src_piece_id;
    uint8_t dst_piece_id;
    if (delta <= moves_since_reset)
    {
        assert(step >= moves_since_reset);
        const uint32_t this_thread_act_row_id = (step - delta) % buf_size;
        src_cell = d_move_summary_history[this_thread_act_row_id * num_envs * 6 + env_idx * 6 + 0];
        dst_cell = d_move_summary_history[this_thread_act_row_id * num_envs * 6 + env_idx * 6 + 1];
        src_piece = d_move_summary_history[this_thread_act_row_id * num_envs * 6 + env_idx * 6 + 2];
        dst_piece = d_move_summary_history[this_thread_act_row_id * num_envs * 6 + env_idx * 6 + 3];
        src_piece_id = d_move_summary_history[this_thread_act_row_id * num_envs * 6 + env_idx * 6 + 4];
        dst_piece_id = d_move_summary_history[this_thread_act_row_id * num_envs * 6 + env_idx * 6 + 5];
    }
    else
    {
        assert(d_move_summary_prehistory);
        src_cell = d_move_summary_prehistory[(move_memory + moves_since_reset - delta) * num_envs * 6 + env_idx * 6 + 0];
        dst_cell = d_move_summary_prehistory[(move_memory + moves_since_reset - delta) * num_envs * 6 + env_idx * 6 + 1];
        src_piece = d_move_summary_prehistory[(move_memory + moves_since_reset - delta) * num_envs * 6 + env_idx * 6 + 2];
        dst_piece = d_move_summary_prehistory[(move_memory + moves_since_reset - delta) * num_envs * 6 + env_idx * 6 + 3];
        src_piece_id = d_move_summary_prehistory[(move_memory + moves_since_reset - delta) * num_envs * 6 + env_idx * 6 + 4];
        dst_piece_id = d_move_summary_prehistory[(move_memory + moves_since_reset - delta) * num_envs * 6 + env_idx * 6 + 5];
    }

    if (relativize && delta % 2 == 1)
    {
        src_cell = 99 - src_cell;
        dst_cell = 99 - dst_cell;

        if (dst_piece == 0b011101 && !(src_piece & 0b10000))
        { // If destination is empty and source is not visible...
            // ... then set piece type to 15 while retaining visibility and whether the piece has moved
            src_piece |= 0b1111;
        }

        if (abs((int8_t)src_cell % 10 - (int8_t)dst_cell % 10) > 1 || abs((int8_t)src_cell / 10 - (int8_t)dst_cell / 10) > 1)
        {
            // If the move is not a single square move, then set the piece type to 15
            src_piece &= 0b11110000;
            src_piece |= 1;
        }
    }

    d_out[(move_memory - delta) * num_envs * 6 + env_idx * 6 + 0] = src_cell;
    d_out[(move_memory - delta) * num_envs * 6 + env_idx * 6 + 1] = dst_cell;
    d_out[(move_memory - delta) * num_envs * 6 + env_idx * 6 + 2] = src_piece;
    d_out[(move_memory - delta) * num_envs * 6 + env_idx * 6 + 3] = dst_piece;
    d_out[(move_memory - delta) * num_envs * 6 + env_idx * 6 + 4] = src_piece_id;
    d_out[(move_memory - delta) * num_envs * 6 + env_idx * 6 + 5] = dst_piece_id;
}

__global__ void SnapshotBoardHistoryKernel(
    StrategoBoard *d_out,
    const StrategoBoard *d_boards,
    const StrategoBoard *d_board_prehistory, // nullptr if no state prehistory is available
    const int32_t *d_num_moves,
    const int32_t *d_num_moves_since_reset,
    const uint64_t step,
    const uint32_t move_memory,
    const uint32_t buf_size,
    const uint32_t num_envs)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t env_idx = idx % num_envs;
    const uint32_t delta = 1 + (idx / num_envs);
    const uint64_t row_id = step % buf_size;

    if (idx >= num_envs * move_memory || delta > d_num_moves[row_id * num_envs + env_idx])
        return;

    const uint32_t moves_since_reset = d_num_moves_since_reset[row_id * num_envs + env_idx];

    StrategoBoard board;
    if (delta <= moves_since_reset)
    {
        assert(step >= moves_since_reset);
        const uint32_t this_thread_row_id = (step - delta) % buf_size;
        board = d_boards[this_thread_row_id * num_envs + env_idx];
    }
    else
    {
        assert(d_board_prehistory);
        assert(delta <= move_memory + moves_since_reset);
        board = d_board_prehistory[(move_memory + moves_since_reset - delta) * num_envs + env_idx];
    }

    d_out[(move_memory - delta) * num_envs + env_idx] = board;
}
