#include "src/env/cuda/kernels.h"
#include "src/env/stratego_board.h"

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
    const uint32_t INFOSTATE_STRIDE)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t env_idx = idx % num_envs;
    const uint32_t delta = move_memory - (idx / num_envs); // in range [1, move_memory]

    const uint64_t row_id = step % buf_size;
    const uint32_t terminated_since = d_terminated_since[row_id * num_envs + env_idx];
    if (
        (idx >= move_memory * num_envs)
        // Do not leak action history across episode boundaries
        || (delta > d_num_moves[row_id * num_envs + env_idx])
        // The env was terminated, so leave the channel empty
        || (delta < terminated_since))
        return;

    const uint32_t moves_since_reset = d_num_moves_since_reset[row_id * num_envs + env_idx];
    MUSTRATEGO_FLOAT_CUDA_DTYPE *out_ptr = d_out + INFOSTATE_STRIDE * env_idx + (idx / num_envs) * 100;

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

    const uint32_t acting_pov_from_cell_idx = action % 100;

    const bool direction = action >= 900; // 0 = vertical, 1 = horizontal
    uint32_t new_coord = (action / 100) % 9;

    // The actions are always encoded from the point of view of the acting player.
    // The acting player is equal to `player` if (num_moves - i) is odd.
    const int32_t acting_pov_from_row_idx = acting_pov_from_cell_idx / 10;
    const int32_t acting_pov_from_col_idx = acting_pov_from_cell_idx % 10;

    int32_t acting_pov_to_row_idx = acting_pov_from_row_idx;
    int32_t acting_pov_to_col_idx = acting_pov_from_col_idx;
    if (!direction)
        acting_pov_to_row_idx = new_coord + (new_coord >= acting_pov_from_row_idx);
    else
        acting_pov_to_col_idx = new_coord + (new_coord >= acting_pov_from_col_idx);

    const bool requires_flip = delta % 2;

    const int32_t pov_from_row_idx = requires_flip ? (9 - acting_pov_from_row_idx) : acting_pov_from_row_idx;
    const int32_t pov_from_col_idx = requires_flip ? (9 - acting_pov_from_col_idx) : acting_pov_from_col_idx;
    const int32_t pov_to_row_idx = requires_flip ? (9 - acting_pov_to_row_idx) : acting_pov_to_row_idx;
    const int32_t pov_to_col_idx = requires_flip ? (9 - acting_pov_to_col_idx) : acting_pov_to_col_idx;

    assert(0 <= pov_to_row_idx && pov_to_row_idx <= 9);
    assert(0 <= pov_to_col_idx && pov_to_col_idx <= 9);

    out_ptr[10 * pov_from_row_idx + pov_from_col_idx] = -1;
    out_ptr[10 * pov_to_row_idx + pov_to_col_idx] = +1;
}

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
    const uint32_t INFOSTATE_STRIDE)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cell_idx = idx % 100;
    const uint32_t env_idx = (idx / 100) % num_envs;
    const uint32_t delta = move_memory - (idx / (100 * num_envs));

    const uint64_t row_id = step % buf_size;
    const uint32_t terminated_since = d_terminated_since[row_id * num_envs + env_idx];
    if (
        (idx >= move_memory * num_envs * 100)
        // Do not leak action history across episode boundaries
        || (delta > d_num_moves[row_id * num_envs + env_idx])
        // The env was terminated, so leave the channel empty
        || (delta < terminated_since))
        return;

    const uint32_t moves_since_reset = d_num_moves_since_reset[row_id * num_envs + env_idx];
    MUSTRATEGO_FLOAT_CUDA_DTYPE *out_ptr = d_out + INFOSTATE_STRIDE * env_idx + (idx / (100 * num_envs)) * 100;

    const Piece *pieces = nullptr;
    if (delta <= moves_since_reset)
    {
        assert(step >= moves_since_reset);
        const uint32_t this_thread_brd_row_id = (step - delta) % buf_size;
        pieces = (const Piece *)d_board_history[this_thread_brd_row_id * num_envs + env_idx].pieces;
    }
    else
    {
        assert(d_board_prehistory);
        pieces = (const Piece *)d_board_prehistory[(move_memory + moves_since_reset - delta) * num_envs + env_idx].pieces;
    }

    const Piece piece = pieces[cell_idx];

    if (player == 2)
        cell_idx = 99 - cell_idx;

    // our_hidden
    out_ptr[cell_idx] = (piece.color == player) && !piece.visible;
    // their_hidden
    out_ptr[cell_idx + 100 * move_memory] = (piece.color == 3 - player) && !piece.visible;
    // our_types
    out_ptr[cell_idx + 200 * move_memory] = (piece.color == player) ? 1.0 / (piece.type + 1) : -1.0;
    // their_visible_types
    out_ptr[cell_idx + 300 * move_memory] = (piece.color == 3 - player && piece.visible) ? 1.0 / (piece.type + 1) : -1.0;
}

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
    const uint32_t INFOSTATE_STRIDE)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t env_idx = idx % num_envs;
    const uint32_t delta = move_memory - (idx / num_envs);

    const uint64_t row_id = step % buf_size;
    const uint32_t terminated_since = d_terminated_since[row_id * num_envs + env_idx];
    if (
        (idx >= move_memory * num_envs)
        // Do not leak action history across episode boundaries
        || (delta > d_num_moves[row_id * num_envs + env_idx])
        // The env was terminated, so leave the channel empty
        || (delta < terminated_since))
        return;

    const uint32_t moves_since_reset = d_num_moves_since_reset[row_id * num_envs + env_idx];
    MUSTRATEGO_FLOAT_CUDA_DTYPE *out_ptr = d_out + INFOSTATE_STRIDE * env_idx + (idx / num_envs) * 100;

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

    const uint32_t acting_pov_from_cell_idx = action % 100;

    const bool direction = action >= 900; // 0 = vertical, 1 = horizontal
    uint32_t new_coord = (action / 100) % 9;

    // The actions are always encoded from the point of view of the acting player.
    // The acting player is equal to `player` if (num_moves - i) is odd.
    const int32_t acting_pov_from_row_idx = acting_pov_from_cell_idx / 10;
    const int32_t acting_pov_from_col_idx = acting_pov_from_cell_idx % 10;

    int32_t acting_pov_to_row_idx = acting_pov_from_row_idx;
    int32_t acting_pov_to_col_idx = acting_pov_from_col_idx;
    if (!direction)
        acting_pov_to_row_idx = new_coord + (new_coord >= acting_pov_from_row_idx);
    else
        acting_pov_to_col_idx = new_coord + (new_coord >= acting_pov_from_col_idx);

    // We now determine if the movement was an attack. We do so by looking at the
    // destination piece and seeing if the color is different from the acting player.
    // First, we need to find the abs coordinatates of the destination cell.
    MUSTRATEGO_FLOAT_CUDA_DTYPE src_encoding = -1.0;
    {
        int32_t abs_to_cell = 10 * acting_pov_to_row_idx + acting_pov_to_col_idx;
        int32_t abs_from_cell = acting_pov_from_cell_idx;
        if ((player + delta) % 2 == 0)
        {
            abs_to_cell = 99 - abs_to_cell;
            abs_from_cell = 99 - abs_from_cell;
        }

        const Piece *pieces = nullptr;
        if (delta <= moves_since_reset)
        {
            assert(step >= moves_since_reset);
            const uint32_t this_thread_brd_row_id = (step - delta) % buf_size;
            pieces = (const Piece *)d_board_history[this_thread_brd_row_id * num_envs + env_idx].pieces;
        }
        else
        {
            assert(d_board_prehistory);
            pieces = (const Piece *)d_board_prehistory[(move_memory + moves_since_reset - delta) * num_envs + env_idx].pieces;
        }
        const Piece src_piece = *(pieces + abs_from_cell);
        const Piece dst_piece = *(pieces + abs_to_cell);
        if (dst_piece.color != 0) // Non-empty destination piece
            src_encoding = -(2.08333333 + MUSTRATEGO_FLOAT_CUDA_DTYPE(src_piece.type) / 12.0);
    }

    const bool requires_flip = delta % 2;

    const int32_t pov_from_row_idx = requires_flip ? (9 - acting_pov_from_row_idx) : acting_pov_from_row_idx;
    const int32_t pov_from_col_idx = requires_flip ? (9 - acting_pov_from_col_idx) : acting_pov_from_col_idx;
    const int32_t pov_to_row_idx = requires_flip ? (9 - acting_pov_to_row_idx) : acting_pov_to_row_idx;
    const int32_t pov_to_col_idx = requires_flip ? (9 - acting_pov_to_col_idx) : acting_pov_to_col_idx;

    assert(0 <= pov_to_row_idx && pov_to_row_idx <= 9);
    assert(0 <= pov_to_col_idx && pov_to_col_idx <= 9);

    out_ptr[10 * pov_from_row_idx + pov_from_col_idx] = src_encoding;
    out_ptr[10 * pov_to_row_idx + pov_to_col_idx] = +1;
}

__global__ void BoardStateKernel__OwnPieceTypes(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player, // The player whose point of view we care about
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    const int32_t cell_idx = index % 100;
    const int32_t row_idx = cell_idx / 10;
    const int32_t col_idx = cell_idx % 10;
    const int32_t pov_cell_idx = (for_player == 2) ? 99 - cell_idx : cell_idx;

    if (env_idx >= num_envs)
        return;

    const Piece piece = d_boards[env_idx].pieces[row_idx][col_idx];

    if (piece.type >= LAKE || piece.color != for_player)
        return;

    MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx;
    out[100 * piece.type + pov_cell_idx] = 1.0;
}

__global__ void BoardStateKernel__ProbTypes(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player, // The player whose point of view we care about
    const bool rotate,        // Perform an extra rotation of the board
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE,
    const uint32_t CHANNEL_SHIFT) // Typically 1200 or 2400
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    const int32_t cell_idx = index % 100;
    const int32_t row_idx = cell_idx / 10;
    const int32_t col_idx = cell_idx % 10;
    const int32_t pov_cell_idx = ((for_player == 2) ^ rotate) ? 99 - cell_idx : cell_idx;

    if (env_idx >= num_envs)
        return;

    const uint8_t *num_hidden = d_boards[env_idx].num_hidden[2 - for_player];
    const uint8_t num_hidden_unmoved = d_boards[env_idx].num_hidden_unmoved[2 - for_player];
    const Piece piece = d_boards[env_idx].pieces[row_idx][col_idx];

    uint8_t total_num_hidden = 0;
    for (int32_t i = 0; i < 12; ++i)
        total_num_hidden += num_hidden[i];

    if (piece.type >= LAKE || piece.color == for_player)
        return;

    // There are only a few possibilities:
    // - If the piece has been moving, it can be any of the types except for BOMB and FLAG.
    // - Else, if the piece has not been moving, it can be any of the types including BOMB and FLAG.
    //
    // There is an exception to the last case. If the number of BOMBs and FLAGs is exactly the number
    // of unmoved pieces, then an unmoved piece cannot be anything else other than BOMB and FLAG.
    if (piece.visible)
    {
        // The piece must belong to the opponent since we did not enter into the previous if statement.
        MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx + CHANNEL_SHIFT;
        out[100 * piece.type + pov_cell_idx] = 1.0;
    }
    else
    {
        // If we are here, the piece is hidden and belongs to the opponent.
        MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx + CHANNEL_SHIFT;

        const float denom = total_num_hidden - num_hidden[FLAG] - num_hidden[BOMB];
        // NOTE: denom is 0 only if the only hidden pieces left are FLAGs and BOMBs.

        if (piece.has_moved)
        {
            // Because the piece has moved, it is not possible that it is a FLAG nor a BOMB.
            // Since the current piece is hidden, then denom > 0.

            out[100 * SPY + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[SPY] / denom);
            out[100 * SCOUT + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[SCOUT] / denom);
            out[100 * MINER + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[MINER] / denom);
            out[100 * SERGEANT + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[SERGEANT] / denom);
            out[100 * LIEUTENANT + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[LIEUTENANT] / denom);
            out[100 * CAPTAIN + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[CAPTAIN] / denom);
            out[100 * MAJOR + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[MAJOR] / denom);
            out[100 * COLONEL + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[COLONEL] / denom);
            out[100 * GENERAL + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[GENERAL] / denom);
            out[100 * MARSHAL + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[MARSHAL] / denom);
        }
        else
        {
            // In this case, we are looking at a hidden piece that has never moved.
            // Hence, num_hidden_unmoved > 0.
            //
            // Unmovable pieces (BOMB and FLAG) must be placed in unmovable cells. So, the
            // probability that the piece we are looking at is a BOMB or a FLAG is higher,
            // especially if the number of hidden unmovable pieces is very close to the number
            // of unmoved cells.
            //
            // Combinatorially, we can imagine that all hidden unmovable pieces are first placed
            // in the unmoved cells, and then the remaining slots are filled with whatever remains.

            // s * (total_num_hidden - num_hidden[FLAG] - num_hidden[BOMB]) + (num_hidden[FLAG] + num_hidden[BOMB]) / num_hidden_unmoved = 1
            //
            // s * denom + (num_hidden[FLAG] + num_hidden[BOMB]) / num_hidden_unmoved = 1
            //
            // => s = (num_hidden_unmoved - num_hidden[FLAG] - num_hidden[BOMB]) / num_hidden_unmoved / denom
            //
            // At this stage, it is still possible that denom == 0. This can happen only when the only remaining
            // hidden pieces are FLAGs and BOMBs.

            if (total_num_hidden != num_hidden[FLAG] + num_hidden[BOMB])
            {
                const float norm_factor = (num_hidden_unmoved - num_hidden[FLAG] - num_hidden[BOMB]) / (1.0f * num_hidden_unmoved * denom);

                out[100 * SPY + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[SPY] * norm_factor);
                out[100 * SCOUT + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[SCOUT] * norm_factor);
                out[100 * MINER + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[MINER] * norm_factor);
                out[100 * SERGEANT + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[SERGEANT] * norm_factor);
                out[100 * LIEUTENANT + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[LIEUTENANT] * norm_factor);
                out[100 * CAPTAIN + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[CAPTAIN] * norm_factor);
                out[100 * MAJOR + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[MAJOR] * norm_factor);
                out[100 * COLONEL + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[COLONEL] * norm_factor);
                out[100 * GENERAL + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[GENERAL] * norm_factor);
                out[100 * MARSHAL + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_hidden[MARSHAL] * norm_factor);
            }

            out[100 * FLAG + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE((float)num_hidden[FLAG] / num_hidden_unmoved);
            out[100 * BOMB + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE((float)num_hidden[BOMB] / num_hidden_unmoved);
        }
    }
}

__global__ void BoardStateKernel__InvisiblesEmptyAndMoved(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const int32_t *num_moves,
    const int32_t *num_moves_since_last_attack,
    const uint32_t max_num_moves,
    const uint32_t max_num_moves_between_attacks,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    if (env_idx >= num_envs)
        return;

    index = index % 100;
    const int32_t row_idx = index / 10;
    const int32_t col_idx = index % 10;
    const Piece piece = d_boards[env_idx].pieces[row_idx][col_idx];
    const int32_t pov_cell_idx = (for_player == 2) ? 99 - index : index;

    MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx;
    out[3600 + pov_cell_idx] = (!piece.visible && piece.color == for_player); // Our own hidden
    out[3700 + pov_cell_idx] = (!piece.visible && piece.color != for_player); // Opponent's hidden
    out[3800 + pov_cell_idx] = (piece.type == EMPTY);
    out[3900 + pov_cell_idx] = (piece.has_moved && piece.color == for_player); // Our own moved
    out[4000 + pov_cell_idx] = (piece.has_moved && piece.color != for_player); // Opponent's moved
    out[4100 + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_moves[env_idx]) / max_num_moves;
    out[4200 + pov_cell_idx] = MUSTRATEGO_FLOAT_CUDA_DTYPE(num_moves_since_last_attack[env_idx]) / max_num_moves_between_attacks;
}

__global__ void BoardStateKernel__ThreatEvadeActiveAdj(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    if (env_idx >= num_envs)
        return;

    index = index % 100;
    const int32_t row_idx = index / 10;
    const int32_t col_idx = index % 10;
    const Piece piece = d_boards[env_idx].pieces[row_idx][col_idx];
    const int32_t pov_cell_idx = (for_player == 2) ? 99 - index : index;

    MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx;
    const int types[] = {SPY, SCOUT, MINER, SERGEANT, LIEUTENANT, CAPTAIN, MAJOR, COLONEL, GENERAL, MARSHAL, HIDDEN_PIECE};
    for (int i = 0; i < 11; ++i)
    {
        const uint8_t t = types[i];
        out[(43 + i) * 100 + pov_cell_idx] = (piece.color != for_player || piece.visible) ? 0 : !!(piece.threatened[t / 8] & (1 << (t % 8)));
        out[(54 + i) * 100 + pov_cell_idx] = (piece.color != for_player || piece.visible) ? 0 : !!(piece.evaded[t / 8] & (1 << (t % 8)));
        out[(65 + i) * 100 + pov_cell_idx] = (piece.color != for_player || piece.visible) ? 0 : !!(piece.actively_adjacent[t / 8] & (1 << (t % 8)));
        out[(76 + i) * 100 + pov_cell_idx] = (piece.color != 3 - for_player || piece.visible) ? 0 : !!(piece.threatened[t / 8] & (1 << (t % 8)));
        out[(87 + i) * 100 + pov_cell_idx] = (piece.color != 3 - for_player || piece.visible) ? 0 : !!(piece.evaded[t / 8] & (1 << (t % 8)));
        out[(98 + i) * 100 + pov_cell_idx] = (piece.color != 3 - for_player || piece.visible) ? 0 : !!(piece.actively_adjacent[t / 8] & (1 << (t % 8)));
    }
}

__global__ void BoardStateKernel__Deaths(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const StrategoBoard *d_zero_boards,
    const uint32_t INFOSTATE_STRIDE)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    if (env_idx >= num_envs || ((index % 100) >= 40 && (index % 100) < 60))
        return;

    index = index % 100;
    const int32_t row_idx = index / 10;
    const int32_t col_idx = index % 10;
    const int32_t rel_cell = index < 40 ? index : 99 - index;
    assert(rel_cell < 40);
    const bool is_dead = !!(d_boards[env_idx].deaths[index >= 60][rel_cell / 8] & (1 << (rel_cell % 8)));
    const Piece piece = d_zero_boards[env_idx].pieces[row_idx][col_idx];

    // Either empty or correct piece id
    assert(piece.piece_id == rel_cell || piece.color == 0);
    assert(piece.color != 0 || (!is_dead && piece.piece_id == 0xff));
    const int32_t pov_cell_idx = (for_player == 2) ? 99 - index : index;

    MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx;
    const PieceType types[] = {SPY, SCOUT, MINER, SERGEANT, LIEUTENANT, CAPTAIN, MAJOR, COLONEL, GENERAL, MARSHAL, BOMB};
    for (int i = 0; i < 11; ++i)
    {
        const PieceType t = types[i];
        out[(109 + i) * 100 + pov_cell_idx] = is_dead && piece.color == for_player && piece.type == t;
        out[(120 + i) * 100 + pov_cell_idx] = is_dead && piece.color == 3 - for_player && piece.type == t;
    }
}

__global__ void BoardStateKernel__DeathReasons(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index % num_envs;
    index /= num_envs;
    const DeathReason reason = (DeathReason)(index % 6);
    assert(reason >= ATTACKED_VISIBLE_STRONGER && reason <= HIDDEN_DEFENDED);
    index /= 6;
    const PieceType piece_type = (PieceType)index;

    if (env_idx >= num_envs || piece_type > MARSHAL)
        return;

    MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx;
    uint8_t plane[100];

    assert(for_player == 1 || for_player == 2);
    assert(piece_type < 10);
    const DeathStatus *ds = d_boards[env_idx].death_status[for_player - 1];
    {
        for (int i = 0; i < 100; ++i)
            plane[i] = 0;
        for (int i = 0; i < 40; ++i)
        {
            if (ds[i].is_dead && ds[i].death_reason == reason && ds[i].piece_type == piece_type)
            {
                const uint8_t loc = ds[i].death_location;
                assert(loc < 100);
                const int32_t pov_cell_idx = (for_player == 2) ? 99 - loc : loc;

                plane[pov_cell_idx] = 1;
            }
        }
        for (int i = 0; i < 100; ++i)
            out[(131 + reason * 10 + piece_type) * 100 + i] = plane[i];
    }

    ds = d_boards[env_idx].death_status[2 - for_player];
    {
        for (int i = 0; i < 100; ++i)
            plane[i] = 0;
        for (int i = 0; i < 40; ++i)
        {
            if (ds[i].is_dead && ds[i].death_reason == reason && ds[i].piece_type == piece_type)
            {
                const uint8_t loc = ds[i].death_location;
                assert(loc < 100);
                const int32_t pov_cell_idx = (for_player == 2) ? 99 - loc : loc;

                plane[pov_cell_idx] = 1;
            }
        }
        for (int i = 0; i < 100; ++i)
            out[(191 + reason * 10 + piece_type) * 100 + i] = plane[i];
    }
}

__global__ void BoardStateKernel__Protections(
    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out,
    const int32_t for_player,
    const uint32_t num_envs,
    const StrategoBoard *d_boards,
    const uint32_t INFOSTATE_STRIDE)
{
    int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;
    if (env_idx >= num_envs)
        return;

    index = index % 100;
    const int32_t row_idx = index / 10;
    const int32_t col_idx = index % 10;
    const Piece piece = d_boards[env_idx].pieces[row_idx][col_idx];
    const int32_t pov_cell_idx = (for_player == 2) ? 99 - index : index;

    MUSTRATEGO_FLOAT_CUDA_DTYPE *out = d_out + INFOSTATE_STRIDE * env_idx;
    const int types[] = {SPY, SCOUT, MINER, SERGEANT, LIEUTENANT, CAPTAIN, MAJOR, COLONEL, GENERAL, MARSHAL, BOMB, EMPTY, HIDDEN_PIECE};
    for (int i = 0; i < 13; ++i)
    {
        const uint8_t t = types[i];
        out[(251 + i) * 100 + pov_cell_idx] = (piece.color != for_player || piece.visible) ? 0 : !!(piece.protected_[t / 8] & (1 << (t % 8)));
        out[(264 + i) * 100 + pov_cell_idx] = (piece.color != for_player || piece.visible) ? 0 : !!(piece.protected_against[t / 8] & (1 << (t % 8)));
        out[(277 + i) * 100 + pov_cell_idx] = (piece.color != for_player || piece.visible) ? 0 : !!(piece.was_protected_by[t / 8] & (1 << (t % 8)));
        out[(290 + i) * 100 + pov_cell_idx] = (piece.color != for_player || piece.visible) ? 0 : !!(piece.was_protected_against[t / 8] & (1 << (t % 8)));
        out[(303 + i) * 100 + pov_cell_idx] = (piece.color != 3 - for_player || piece.visible) ? 0 : !!(piece.protected_[t / 8] & (1 << (t % 8)));
        out[(316 + i) * 100 + pov_cell_idx] = (piece.color != 3 - for_player || piece.visible) ? 0 : !!(piece.protected_against[t / 8] & (1 << (t % 8)));
        out[(329 + i) * 100 + pov_cell_idx] = (piece.color != 3 - for_player || piece.visible) ? 0 : !!(piece.was_protected_by[t / 8] & (1 << (t % 8)));
        out[(342 + i) * 100 + pov_cell_idx] = (piece.color != 3 - for_player || piece.visible) ? 0 : !!(piece.was_protected_against[t / 8] & (1 << (t % 8)));
    }
}