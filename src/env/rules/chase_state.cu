#include "src/env/rules/chase_state.h"

#include "src/env/stratego.h"
#include "src/util.h"
#include <cstdint>

__global__ void RemoveIllegalChaseMovesKernel(
    bool *d_legal_action_mask,
    const int32_t *d_illegal_chase_actions,
    const uint8_t *d_mask,
    const uint32_t num_envs)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t env_idx = idx % num_envs;

    if (idx >= num_envs * MAX_CHASE_LENGTH || d_mask[env_idx])
        return;

    const int32_t action = d_illegal_chase_actions[idx];
    if (action != -1)
    {
        assert(action >= 0 && action < NUM_ACTIONS);
        d_legal_action_mask[env_idx * NUM_ACTIONS + action] = false;
    }
}

/// The update
__global__ void UpdateChaseStateKernel(
    const uint32_t num_envs,
    ChaseState d_state,
    const StrategoBoard *d_board,
    const uint8_t *move_summary,
    const uint8_t *d_mask,
    const uint8_t player // 1 = red, 2 = blue
)
{
    const uint8_t opponent = 3 - player;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs || d_mask[idx])
        return;

    // Step i.
    const uint8_t abs_src = (player == 1) ? move_summary[idx * 6 + 0] : 99 - move_summary[idx * 6 + 0];
    const uint8_t abs_dst = (player == 1) ? move_summary[idx * 6 + 1] : 99 - move_summary[idx * 6 + 1];
    const int32_t is_attack = (move_summary[idx * 6 + 3] != 29);

    d_state.last_dst_pos[player - 1][idx] = abs_dst;
    d_state.last_src_pos[player - 1][idx] = abs_src;

    // Step ii.
    int32_t opp_chase = d_state.chase_length[opponent - 1][idx];
    if (IS_ADJACENT(abs_src, d_state.last_dst_pos[opponent - 1][idx]))
    {
        ++opp_chase;
    }
    else
    {
        opp_chase = 0;
    }

    // Step iii.
    int32_t player_chase = d_state.chase_length[player - 1][idx];
    if (is_attack)
    {
        player_chase = 0;
        opp_chase = 0;
    }
#ifdef DEBUG
    printf("[env_idx: %2d] opponent: %d -> setting chase_length: %d\n", idx, opponent, opp_chase);
#endif
    d_state.chase_length[opponent - 1][idx] = opp_chase;

    // Step iv.
    StrategoBoard board = d_board[idx];
    bool dst_adjacent_opponent = false;
    if (abs_dst >= 10)
        dst_adjacent_opponent |= (board.pieces[abs_dst / 10 - 1][abs_dst % 10].color == opponent);
    if (abs_dst < 90)
        dst_adjacent_opponent |= (board.pieces[abs_dst / 10 + 1][abs_dst % 10].color == opponent);
    if (abs_dst % 10 != 0)
        dst_adjacent_opponent |= (board.pieces[abs_dst / 10][abs_dst % 10 - 1].color == opponent);
    if (abs_dst % 10 != 9)
        dst_adjacent_opponent |= (board.pieces[abs_dst / 10][abs_dst % 10 + 1].color == opponent);
    if (dst_adjacent_opponent)
    {
        ++player_chase;
    }
    else
    {
        player_chase = 0;
    }
#ifdef DEBUG
    printf("[env_idx: %2d] player: %d -> setting chase_length: %d\n", idx, player, player_chase);
#endif
    d_state.chase_length[player - 1][idx] = player_chase;
}

__global__ void ComputeIllegalChaseMovesKernel(
    int32_t *d_out,
    const ChaseState d_state,
    const StrategoBoard *d_board_history,
    const StrategoBoard *d_board_prehistory, // can be nullptr
    const int32_t *d_num_moves_since_reset,
    const uint8_t *d_mask,
    const uint32_t current_step,
    const uint32_t buf_size, // Circular buffer length
    const uint8_t player,    // 1 = red, 2 = blue
    const uint32_t prehistory_size,
    const uint32_t num_envs)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t env_idx = idx % num_envs;
    const uint32_t delta = 1 + (idx / num_envs);
    const uint64_t row_id = current_step % buf_size;

    if (env_idx >= num_envs || d_mask[env_idx])
        return;
    const int32_t chase_length = d_state.chase_length[player - 1][env_idx];
#ifdef DEBUG
    if (delta == 1)
    {
        printf("[env_idx: %2d] player: %d chase_length: %d\n", env_idx, player, chase_length);
    }
#endif
    assert(player == 1 || player == 2);
    const uint8_t last_dst = d_state.last_dst_pos[player - 1][env_idx];
    const uint8_t last_src = d_state.last_src_pos[player - 1][env_idx];
    if (chase_length <= 0 || delta >= chase_length)
        return;
    const uint32_t moves_since_reset = d_num_moves_since_reset[row_id * num_envs + env_idx];

    const StrategoBoard *delta_board;
    if (delta <= moves_since_reset)
    {
        assert(current_step >= moves_since_reset);
        const uint32_t this_thread_row_id = (current_step - delta) % buf_size;
        delta_board = &d_board_history[this_thread_row_id * num_envs + env_idx];
    }
    else
    {
        assert(d_board_prehistory);
        assert(delta <= prehistory_size + moves_since_reset);
        delta_board = &d_board_prehistory[(prehistory_size + moves_since_reset - delta) * num_envs + env_idx];
    }

    const StrategoBoard *current_board = &d_board_history[row_id * num_envs + env_idx];
    uint8_t src_cell = 0xff;
    uint8_t dst_cell = 0xff;
    Piece dst_piece;
    Piece src_piece;

#ifdef DEBUG
    if (delta == 1)
    {
        const char PIECE_ENCODING[4][NUM_PIECE_TYPES] = {
            {'#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', 'a'}, // color: empty (0)
            {'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'B', '#', '#'}, // color: red   (1)
            {'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'N', '#', '#'}, // color: blue  (2)
            {'#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '_', '#'}, // color: lake  (3)
        };
        char s[101];
        for (int r = 0; r < 10; ++r)
            for (int c = 0; c < 10; ++c)
            {
                assert(current_board->pieces[r][c].type < NUM_PIECE_TYPES);
                assert(current_board->pieces[r][c].color < 4);
                s[r * 10 + c] = PIECE_ENCODING[current_board->pieces[r][c].color][current_board->pieces[r][c].type];
            }
        s[100] = '\0';
        printf("[env_idx: %2d, delta: %2d] current_board: %s\n", env_idx, delta, s);
    }
#endif

    for (int r = 0; r < 10; ++r)
    {
        for (int c = 0; c < 10; ++c)
        {
            if (delta_board->pieces[r][c].type != current_board->pieces[r][c].type ||
                delta_board->pieces[r][c].color != current_board->pieces[r][c].color)
            {
                if ((src_cell != 0xff && dst_cell != 0xff) || (delta_board->pieces[r][c].type != EMPTY && current_board->pieces[r][c].type != EMPTY))
                {
                    // Bail if there are more than two changes, or if there is a change between two pieces that are not empty.

#ifdef DEBUG
                    printf("[env_idx: %2d, delta: %2d] Bail because of two changed\n", env_idx, delta);
#endif
                    return;
                }

                if (delta_board->pieces[r][c].type == EMPTY)
                {
                    if (src_cell != 0xff)
                        return;
                    src_cell = r * 10 + c;
                    src_piece = current_board->pieces[r][c];
                }
                else
                {
                    assert(current_board->pieces[r][c].type == EMPTY);
                    if (dst_cell != 0xff)
                        return;
                    dst_cell = r * 10 + c;
                    dst_piece = delta_board->pieces[r][c];
                }
            }
        }
    }
#ifdef DEBUG
    printf("[env_idx: %2d, delta: %2d] CONSIDERING src_cell: %d, src_piece: %d, dt_cell: %d, dst_piece: %d\n", env_idx, delta, src_cell, src_piece.type, dst_cell, dst_piece.type);
#endif

    // Either no changes or exactly one change
    assert(!(src_cell == 0xff ^ dst_cell == 0xff));
    if (src_cell == 0xff)
    {
        // No change.
        return;
    }

    // If exactly one piece has moved, check that it is the same piece.
    assert(src_piece.type == dst_piece.type && src_piece.color == dst_piece.color);

    if ((src_cell % 10 != dst_cell % 10) && (src_cell / 10 != dst_cell / 10))
    {
        // Not reachable.
        return;
    }
    if (src_piece.color != player)
    {
        // Wrong player.

#ifdef DEBUG
        printf("[env_idx: %2d, delta: %2d] Wrong player\n", env_idx, delta);
#endif
        return;
    }
    assert(src_cell != dst_cell);

    // If moving src_cell -> dst_cell is not a threat, the action is not per se illegal.
    const bool is_threat_d = (dst_cell / 10 != 0) && (current_board->pieces[(dst_cell / 10) - 1][dst_cell % 10].color == 3 - player);
    const bool is_threat_u = (dst_cell / 10 != 9) && (current_board->pieces[(dst_cell / 10) + 1][dst_cell % 10].color == 3 - player);
    const bool is_threat_l = (dst_cell % 10 != 0) && (current_board->pieces[dst_cell / 10][(dst_cell % 10) - 1].color == 3 - player);
    const bool is_threat_r = (dst_cell % 10 != 9) && (current_board->pieces[dst_cell / 10][(dst_cell % 10) + 1].color == 3 - player);
    if (!is_threat_d && !is_threat_u && !is_threat_l && !is_threat_r)
    {
        // Not a threat.

#ifdef DEBUG
        printf("[env_idx: %2d, delta: %2d] Not a threat\n", env_idx, delta);
#endif
        return;
    }
    // Allow reverting the last move.
    if (src_cell == last_dst && dst_cell == last_src)
    {
#ifdef DEBUG
        printf("[env_idx: %2d, delta: %2d] Reverting last move\n", env_idx, delta);
#endif
        return;
    }
#ifdef DEBUG
    printf("[env_idx: %2d, delta: %2d] CONFIRMED src_cell: %d, src_piece: %d, dst_cell: %d, dst_piece: %d\n", env_idx, delta, src_cell, src_piece.type, dst_cell, dst_piece.type);
#endif

    int32_t action = src_cell;
    if (src_cell % 10 == dst_cell % 10)
    {
        action += 100 * (dst_cell / 10 - (dst_cell >= src_cell));
        if (player == 2)
            action = 899 - action;
    }
    else
    {
        assert(src_cell / 10 == dst_cell / 10);
        action += 100 * (dst_cell % 10 - (dst_cell >= src_cell));
        if (player == 2)
            action = 899 - action;
        action += 900;
    }
#ifdef DEBUG
    printf("Illegal move: %d, src_cell: %d, dst_cell: %d, chase_length: %d, moves_since_reset: %d\n", action, src_cell, dst_cell, chase_length, moves_since_reset);
#endif
    assert(action >= 0 && action < NUM_ACTIONS);
    d_out[(delta - 1) * num_envs + env_idx] = action;
}

void ResetChaseState(ChaseState d_out,
                     const std::optional<ChaseState> &d_src,
                     const uint32_t num_envs,
                     const int32_t cuda_device,
                     const int32_t *d_mask)
{
    torch::Tensor out_last_dst_pos[2] = {
        MUSTRATEGO_WRAP_CUDA_TENSOR(d_out.last_dst_pos[0], cuda_device, torch::kUInt8, {num_envs}),
        MUSTRATEGO_WRAP_CUDA_TENSOR(d_out.last_dst_pos[1], cuda_device, torch::kUInt8, {num_envs})};
    torch::Tensor out_last_src_pos[2] = {
        MUSTRATEGO_WRAP_CUDA_TENSOR(d_out.last_src_pos[0], cuda_device, torch::kUInt8, {num_envs}),
        MUSTRATEGO_WRAP_CUDA_TENSOR(d_out.last_src_pos[1], cuda_device, torch::kUInt8, {num_envs})};
    torch::Tensor out_chase_length[2] = {
        MUSTRATEGO_WRAP_CUDA_TENSOR(d_out.chase_length[0], cuda_device, torch::kInt32, {num_envs}),
        MUSTRATEGO_WRAP_CUDA_TENSOR(d_out.chase_length[1], cuda_device, torch::kInt32, {num_envs})};

    if (!d_src)
    {
        if (!d_mask)
        {
            out_last_dst_pos[0].fill_(0xee);
            out_last_dst_pos[1].fill_(0xee);

            out_last_src_pos[0].fill_(0xee);
            out_last_src_pos[1].fill_(0xee);

            out_chase_length[0].fill_(0);
            out_chase_length[1].fill_(0);
        }
        else
        {
            // FIXME(gfarina): This creates a temporary tensor.
            const torch::Tensor mask = ~(MUSTRATEGO_WRAP_CUDA_TENSOR((void *)d_mask, cuda_device, torch::kInt32, {num_envs}).to(torch::kBool));

            out_last_dst_pos[0].masked_fill_(mask, 0xee);
            out_last_dst_pos[1].masked_fill_(mask, 0xee);

            out_last_src_pos[0].masked_fill_(mask, 0xee);
            out_last_src_pos[1].masked_fill_(mask, 0xee);

            out_chase_length[0].masked_fill_(mask, 0);
            out_chase_length[1].masked_fill_(mask, 0);
        }
    }
    else
    {
        const torch::Tensor src_last_dst_pos[2] = {
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_src->last_dst_pos[0], cuda_device, torch::kUInt8, {num_envs}),
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_src->last_dst_pos[1], cuda_device, torch::kUInt8, {num_envs})};
        const torch::Tensor src_last_src_pos[2] = {
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_src->last_src_pos[0], cuda_device, torch::kUInt8, {num_envs}),
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_src->last_src_pos[1], cuda_device, torch::kUInt8, {num_envs})};
        const torch::Tensor src_chase_length[2] = {
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_src->chase_length[0], cuda_device, torch::kInt32, {num_envs}),
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_src->chase_length[1], cuda_device, torch::kInt32, {num_envs})};

        if (!d_mask)
        {
            out_last_dst_pos[0].copy_(src_last_dst_pos[0]);
            out_last_dst_pos[1].copy_(src_last_dst_pos[1]);
            out_last_src_pos[0].copy_(src_last_src_pos[0]);
            out_last_src_pos[1].copy_(src_last_src_pos[1]);
            out_chase_length[0].copy_(src_chase_length[0]);
            out_chase_length[1].copy_(src_chase_length[1]);
        }
        else
        {
            // FIXME(gfarina): This creates a temporary tensor.
            const torch::Tensor mask = ~(MUSTRATEGO_WRAP_CUDA_TENSOR((void *)d_mask, cuda_device, torch::kInt32, {num_envs}).to(torch::kBool));
            out_last_dst_pos[0].index_put_({mask}, src_last_dst_pos[0].index({mask}));
            out_last_dst_pos[1].index_put_({mask}, src_last_dst_pos[1].index({mask}));
            out_last_src_pos[0].index_put_({mask}, src_last_src_pos[0].index({mask}));
            out_last_src_pos[1].index_put_({mask}, src_last_src_pos[1].index({mask}));
            out_chase_length[0].index_put_({mask}, src_chase_length[0].index({mask}));
            out_chase_length[1].index_put_({mask}, src_chase_length[1].index({mask}));
        }

        assert(out_chase_length[0].max().item<int32_t>() <= MAX_CHASE_LENGTH - 1);
        assert(out_chase_length[1].max().item<int32_t>() <= MAX_CHASE_LENGTH - 1);
    }
}

void UpdateChaseState(
    ChaseState d_state,
    const StrategoBoard *d_board,
    const uint8_t *d_move_summary,
    const uint8_t *d_mask,
    const uint8_t player,
    const uint32_t num_envs,
    const int32_t cuda_device)
{
    MUSTRATEGO_CHECK(player == 1 || player == 2, "Invalid player (found %d)", player);

    const uint32_t num_threads = 256;
    const uint32_t num_blocks = ceil(num_envs, num_threads);

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(cuda_device));
    UpdateChaseStateKernel<<<num_blocks, num_threads>>>(
        num_envs,
        d_state,
        d_board,
        d_move_summary,
        d_mask,
        player);
}

void ComputeIllegalChaseMoves(
    int32_t *d_out,
    const ChaseState d_state,
    const StrategoBoard *d_board_history,
    const StrategoBoard *d_board_prehistory, // can be nullptr
    const int32_t *d_num_moves_since_reset,
    const uint8_t *d_mask,
    const uint32_t current_step,
    const uint32_t buf_size, // Circular buffer length
    const uint8_t player,    // 1 = red, 2 = blue
    const uint32_t prehistory_size,
    const uint32_t num_envs,
    const int32_t cuda_device)
{
    MUSTRATEGO_CHECK(player == 1 || player == 2, "Invalid player (found %d)", player);

    const torch::Tensor out = MUSTRATEGO_WRAP_CUDA_TENSOR((void *)d_out, cuda_device, torch::kInt32, {MAX_CHASE_LENGTH, num_envs});
    out.fill_(-1);

    const torch::Tensor chase_length_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_state.chase_length[player - 1], cuda_device, torch::kInt32, {num_envs});
    const uint32_t num_threads = 128;
    const int32_t max_chase_length = chase_length_tensor.max().item<int32_t>();
    if (max_chase_length <= 0)
        return;

    MUSTRATEGO_CHECK(max_chase_length <= MAX_CHASE_LENGTH, "MAX_CHASE_LENGTH has been exceeded (current value is %d)", max_chase_length);
    const uint32_t num_blocks = ceil(num_envs * max_chase_length, num_threads);

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(cuda_device));
    ComputeIllegalChaseMovesKernel<<<num_blocks, num_threads>>>(
        d_out,
        d_state,
        d_board_history,
        d_board_prehistory,
        d_num_moves_since_reset,
        d_mask,
        current_step,
        buf_size,
        player,
        prehistory_size,
        num_envs);
}

void RemoveIllegalChaseMoves(
    bool *d_legal_action_mask,
    const int32_t *d_illegal_chase_actions,
    const uint8_t *d_mask,
    const uint32_t num_envs,
    const int32_t cuda_device)
{
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(MAX_CHASE_LENGTH * num_envs, num_threads);

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(cuda_device));
    RemoveIllegalChaseMovesKernel<<<num_blocks, num_threads>>>(
        d_legal_action_mask,
        d_illegal_chase_actions,
        d_mask,
        num_envs);
}
