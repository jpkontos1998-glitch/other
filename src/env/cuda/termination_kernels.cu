#include "src/env/cuda/kernels.h"
#include "src/env/stratego_board.h"

__global__ void InitBoardsKernel(
    StrategoBoard *d_boards,
    const StrategoBoard *init,
    const int32_t *init_offset_red,
    const int32_t *init_offset_blue,
    const uint32_t num_envs,
    const bool make_pieces_visible)
{
    // Each thread is responsible for moving 4 bytes (4 pieces on the board from `init` to `d_boards`).
    // Each board is 1920 bytes, so we need 1920/4=480 tiles per environment.
    const int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int env_idx = tile_idx / 480;

    if (env_idx >= num_envs)
        return;

    // Get the first 8*50 bytes from init_red and the remaining from init_blue.
    const uint32_t offset = (tile_idx % 480 < 200) ? init_offset_red[env_idx] : init_offset_blue[env_idx];
    const uint32_t *from_ptr = (uint32_t *)(init + offset);
    uint32_t *to_ptr = (uint32_t *)d_boards;

    to_ptr[tile_idx] = from_ptr[tile_idx % 480];

    if (make_pieces_visible)
    {
        // Mark pieces as visible (first 800 bytes)
        // and all hidden counters as zeros (remaining 28 bytes)
        if (tile_idx % 480 < 400)
        {
            to_ptr[tile_idx] |= (tile_idx % 4) ? 0 : 0x00000040;
        }
        else if (tile_idx % 480 < 407)
        {
            to_ptr[tile_idx] = 0;
        }
    }
}

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
    const TwosquareState *d_twosquare_state_init_blue)
{
    // Each thread is responsible for moving 4 bytes (4 pieces on the board from `init` to `d_boards`).
    // Each board is 1920 bytes, so we need 1920/4=480 tiles per environment.
    const int32_t tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = tile_idx / 480;

    if (env_idx >= num_envs)
        return;
    if (d_terminated_since[env_idx] < 2)
        return;

    // Get the first 8*50 bytes from init_red and the remaining from init_blue.
    const uint32_t offset = (tile_idx % 480 < 200) ? init_offset_red[env_idx] : init_offset_blue[env_idx];
    const uint32_t *from_ptr = (uint32_t *)(init + offset);
    const uint32_t *zero_from_ptr = (uint32_t *)(zero_init + offset);
    uint32_t *board_ptr = (uint32_t *)d_boards;
    uint32_t *zero_board_ptr = (uint32_t *)d_zero_boards;

    board_ptr[tile_idx] = from_ptr[tile_idx % 480];
    zero_board_ptr[tile_idx] = zero_from_ptr[tile_idx % 480];

    if (make_pieces_visible)
    {
        // Mark pieces as visible (first 800 bytes)
        // and all hidden counters as zeros (remaining 28 bytes)
        if (tile_idx % 480 < 400)
        {
            board_ptr[tile_idx] |= 0x00400040;
            zero_board_ptr[tile_idx] |= 0x00400040;
        }
        else if (tile_idx % 480 < 407)
        {
            board_ptr[tile_idx] = 0;
            zero_board_ptr[tile_idx] = 0;
        }
    }

    if (tile_idx % 480 == 0)
    {
        d_num_moves_since_reset[env_idx] = 0;
        d_twosquare_state_red[env_idx] = d_twosquare_state_init_red[env_idx];
        d_twosquare_state_blue[env_idx] = d_twosquare_state_init_blue[env_idx];

        if (d_reset_state_num_moves)
        {
            assert(d_reset_state_terminated_since && d_reset_state_flag_captured);

            d_num_moves[env_idx] = d_reset_state_num_moves[env_idx];
            d_num_moves_since_last_attack[env_idx] = d_reset_state_num_moves_since_last_attack[env_idx];
            d_flag_captured[env_idx] = d_reset_state_flag_captured[env_idx];
        }
        else
        {
            d_num_moves[env_idx] = 0;
            d_num_moves_since_last_attack[env_idx] = 0;
            d_flag_captured[env_idx] = 0;
        }
    }
}

__global__ void ResetTerminationCountersKernel(
    uint8_t *d_terminated_since,
    const uint8_t *d_reset_state_terminated_since, // `nullptr` if `!reset_state`
    const uint32_t num_envs)
{
    const int32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs)
        return;
    if (d_terminated_since[env_idx] < 2)
        return;

    if (d_reset_state_terminated_since)
        d_terminated_since[env_idx] = d_reset_state_terminated_since[env_idx];
    else
        d_terminated_since[env_idx] = 0;
}

__global__ void IncrementTerminationCounterKernel(
    uint8_t *d_terminated_since,
    uint8_t *d_flag_captured,
    int32_t *d_num_moves,
    int32_t *d_num_moves_since_last_attack,
    int32_t *d_num_moves_since_reset,
    const uint8_t *d_has_legal_movement,
    const uint32_t num_envs,
    const uint32_t max_num_moves,
    const uint32_t max_num_moves_between_attacks)
{
    const int32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs)
        return;

    // The following checks prevents incrementing the termination counter for envs that have
    // just been reset, since for those an action has not been taken yet.
    //
    // This becomes only relevant to handle correctly those rare cases in which the reset state
    // is a state that has terminated.
    if (d_num_moves_since_reset[env_idx] == 0)
        return;

    const bool has_terminated = (d_has_legal_movement[env_idx] < 3) || d_flag_captured[env_idx] || d_num_moves[env_idx] > max_num_moves || d_num_moves_since_last_attack[env_idx] > max_num_moves_between_attacks
                                // The reason why the next `or` is needed is the following scenario:
                                // Pl.0 cannot move (and no flag is captured), so has_terminated is set to `true`.
                                // But then, when the turn passes to Pl.1, Pl.1 could move. In this case, the game
                                // of course needs to remain marked as terminated.
                                || d_terminated_since[env_idx];
    d_terminated_since[env_idx] += has_terminated;
}

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
    const uint8_t to_play)
{
    const int32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs)
        return;
    const uint32_t terminated_since = d_terminated_since[env_idx];
    assert((terminated_since > 1 ? terminated_since - 1 : 0) <= d_num_moves[env_idx]);
    const uint32_t num_moves_before_termination = d_num_moves[env_idx] - (terminated_since > 1 ? terminated_since - 1 : 0);
    const uint32_t num_moves_since_last_attack_before_termination = d_num_moves_since_last_attack[env_idx] - (terminated_since > 1 ? terminated_since - 1 : 0);

    const bool not_timeout = (num_moves_before_termination <= max_num_moves) &&
                             (num_moves_since_last_attack_before_termination <= max_num_moves_between_attacks);

    if (not_timeout && d_flag_captured[env_idx])
    {
        assert(d_terminated_since[env_idx]);

        // flag_captured stores which player captured the flag. If the player is 1,
        // then the reward is +1, if it player 2, then the reward should be -1.
        d_reward_pl0[env_idx] = -2.0 * d_flag_captured[env_idx] + 3.0;
    }
    else if (not_timeout && d_terminated_since[env_idx])
    {
        // At least one player cannot move.
        assert(d_has_legal_movement[env_idx] < 3);

        // The player that cannot move has lost. In case both players cannot move, the game is a tie.
        //
        // Let x be the value of d_has_legal_movement[env_idx]. Then:
        // - x = 0 (no player can move)     -> reward is  0
        // - x = 1 (only red/pl0  can move) -> reward is  1
        // - x = 2 (only blue/pl1 can move) -> reward is -1
        const uint8_t x = d_has_legal_movement[env_idx];

        d_reward_pl0[env_idx] = (x <= 1) ? x : -1;
    }
    else
    {
        d_reward_pl0[env_idx] = 0;
    }
}
