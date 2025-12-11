#include "src/env/cuda/kernels.h"
#include <cstdint>

__global__ void ComputeIsUnknownPieceKernel(
    bool *d_out,
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint8_t player)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    const int32_t cell_idx = index % 100;

    if (env_idx >= num_envs)
        return;

    const int32_t row_idx = cell_idx / 10;
    const int32_t col_idx = cell_idx % 10;
    const int32_t pov_cell = (player == 1) ? cell_idx : 99 - cell_idx;

    const StrategoBoard *board = d_boards + env_idx;
    const Piece piece = board->pieces[row_idx][col_idx];

    d_out[100 * env_idx + pov_cell] = (!piece.visible && piece.color != player);
}

__global__ void ComputePieceTypeOnehotKernel(
    bool *d_out,
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint8_t player)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    const int32_t cell_idx = index % 100;

    if (env_idx >= num_envs)
        return;

    const int32_t row_idx = cell_idx / 10;
    const int32_t col_idx = cell_idx % 10;
    const int32_t pov_cell = (player == 1) ? cell_idx : 99 - cell_idx;

    const StrategoBoard *board = d_boards + env_idx;
    const Piece piece = board->pieces[row_idx][col_idx];
    assert(piece.type < NUM_PIECE_TYPES);

    d_out[NUM_PIECE_TYPES * 100 * env_idx + pov_cell * NUM_PIECE_TYPES + piece.type] = 1;
}

__global__ void ComputeUnknownPieceTypeOnehotKernel(
    bool *d_out,
    const uint8_t *unknown_ranks, // In RELATIVE coordinates
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint32_t max_k,
    const uint8_t player)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    const int32_t relative_cell_idx = index % 100;
    const int32_t absolute_cell_idx = (player == 1) ? relative_cell_idx : 99 - relative_cell_idx;
    const uint32_t rank = unknown_ranks[index];

    if (env_idx >= num_envs)
        return;
    if (rank == 0 || rank > max_k)
        return;
    if (relative_cell_idx > 0 && (unknown_ranks[index - 1] == rank))
        return;

    const int32_t row_idx = absolute_cell_idx / 10;
    const int32_t col_idx = absolute_cell_idx % 10;

    const StrategoBoard *board = d_boards + env_idx;
    const Piece piece = board->pieces[row_idx][col_idx];
    assert(piece.type < NUM_PIECE_TYPES);

    d_out[NUM_PIECE_TYPES * max_k * env_idx + (rank - 1) * NUM_PIECE_TYPES + piece.type] = 1;
}

__global__ void ComputeUnknownPieceHasMovedKernel(
    bool *d_out,
    const uint8_t *unknown_ranks, // In RELATIVE coordinates
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint32_t max_k,
    const uint8_t player)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    const int32_t relative_cell_idx = index % 100;
    const int32_t absolute_cell_idx = (player == 1) ? relative_cell_idx : 99 - relative_cell_idx;
    const uint32_t rank = unknown_ranks[index];

    if (env_idx >= num_envs)
        return;
    if (rank == 0 || rank > max_k)
        return;
    if (relative_cell_idx > 0 && (unknown_ranks[index - 1] == rank))
        return;

    const int32_t row_idx = absolute_cell_idx / 10;
    const int32_t col_idx = absolute_cell_idx % 10;

    const StrategoBoard *board = d_boards + env_idx;
    const Piece piece = board->pieces[row_idx][col_idx];

    d_out[max_k * env_idx + (rank - 1)] = piece.has_moved;
}

__global__ void ComputeUnknownPiecePositionOnehotKernel(
    bool *d_out,
    const uint8_t *unknown_ranks, // In RELATIVE coordinates
    const uint32_t num_envs,
    const uint32_t max_k,
    const uint8_t player)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    const int32_t relative_cell_idx = index % 100;
    const uint32_t rank = unknown_ranks[index];

    if (env_idx >= num_envs)
        return;
    if (rank == 0 || rank > max_k)
        return;
    if (relative_cell_idx > 0 && (unknown_ranks[index - 1] == rank))
        return;

    d_out[100 * max_k * env_idx + (rank - 1) * 100 + relative_cell_idx] = 1;
}

__global__ void AssignBoardPiecesKernel(
    uint8_t *d_boards,
    const int32_t *d_id_to_type, // piece id to piece type
    const uint32_t num_boards,
    const uint32_t num_envs,
    const uint8_t opponent)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t cell_idx = idx % 100;
    idx /= 100;
    const uint32_t env_idx = idx % num_envs;
    const uint32_t board_idx = idx / num_envs; // history delta
    if (board_idx >= num_boards)
        return;

    uint8_t *stat = &d_boards[board_idx * num_envs * sizeof(StrategoBoard) + env_idx * sizeof(StrategoBoard) + cell_idx * 16];
    const uint8_t piece_id = d_boards[board_idx * num_envs * sizeof(StrategoBoard) + env_idx * sizeof(StrategoBoard) + cell_idx * 16 + 1];
    const int32_t *assignment = &d_id_to_type[40 * env_idx];

    if (((*stat) & 0b01110000) == (opponent << 4))
    {
        assert(piece_id < 40);
        // When processing past boards, there might be more unknown pieces
        // than at current time. In that case, we do not overwrite the type.
        if (assignment[piece_id] != -1)
        {
            assert(assignment[piece_id] < LAKE);
            *stat = ((*stat) & 0b11110000) | assignment[piece_id];
        }
    }
}