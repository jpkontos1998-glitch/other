#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include "src/env/rules/twosquare_state.h"
#include "src/env/stratego.h"
#include "src/util.h"

#define COL(x) (x % 10)
#define ROW(x) (x / 10)

__global__ void UpdateTwosquareActionKernel(
    TwosquareState *d_state,
    const int32_t *d_actions,
    const uint8_t *d_mask,
    const uint32_t num_envs)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs || (d_mask && d_mask[idx]))
        return;

    TwosquareState state = d_state[idx];
    const int32_t action = d_actions[idx];
    const uint8_t src_col = action % 10;
    const uint8_t src_row = (action / 10) % 10;

    uint8_t dst_col = (action >= 900) ? (action / 100) % 9 : src_col;
    uint8_t dst_row = (action >= 900) ? src_row : (action / 100);
    if (action < 900 && src_col == dst_col)
        dst_row += (dst_row >= src_row);
    if (action >= 900 && src_row == dst_row)
        dst_col += (dst_col >= src_col);

    const bool is_vertical = action < 900;
    const bool last_vertical = (COL(state.A) == COL(state.B));

    // The chain continues if the last position is the source cell of the action, and
    // the previous action was moving in the same direction (horizontal / vertical).
    //
    // Note that if the state is set to the special marker byte 0xff, then the first condition
    // will fail for sure.
    if (state.A == 10 * src_row + src_col && is_vertical == last_vertical)
    {
        state.D = state.C;
        state.C = state.B;
    }
    else
    {
        state.D = 0xff;
        state.C = 0xff;
    }
    state.A = 10 * dst_row + dst_col;
    state.B = 10 * src_row + src_col;

    d_state[idx] = state;
}

__global__ void UpdateTwosquareDeathKernel(
    TwosquareState *d_state,
    const uint8_t *d_death_cell,
    const uint8_t *d_mask,
    const uint32_t num_envs)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs || (d_mask && d_mask[idx]))
        return;

    TwosquareState state = d_state[idx];
    const uint8_t cell = d_death_cell[idx];

    if (state.A == cell)
    {
        state.A = state.B = state.C = state.D = 0xff;
    }

    d_state[idx] = state;
}

__global__ void IsTwosquareRuleTriggeredKernel(
    bool *d_out,
    const TwosquareState *d_state,
    const uint32_t num_envs)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs)
        return;

    TwosquareState state = d_state[idx];
    const bool ans =
        (state.D != 0xff) &&
        (
            // Horizontal
            (COL(state.D) < COL(state.C) && COL(state.C) > COL(state.B) && COL(state.B) < COL(state.A) && COL(state.D) < COL(state.A)) ||
            (COL(state.D) > COL(state.C) && COL(state.C) < COL(state.B) && COL(state.B) > COL(state.A) && COL(state.D) > COL(state.A)) ||
            // Vertical
            (ROW(state.D) < ROW(state.C) && ROW(state.C) > ROW(state.B) && ROW(state.B) < ROW(state.A) && ROW(state.D) < ROW(state.A)) ||
            (ROW(state.D) > ROW(state.C) && ROW(state.C) < ROW(state.B) && ROW(state.B) > ROW(state.A) && ROW(state.D) > ROW(state.A))
            //
        );
    d_out[idx] = ans;
}
__global__ void IsTwosquareRulePrecludingDirectionKernel(
    bool *d_out,
    const TwosquareState *d_state,
    const uint32_t num_envs)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs)
        return;

    TwosquareState state = d_state[idx];
    const bool ans =
        (state.D != 0xff) &&
        (
            // Horizontal
            (COL(state.D) < COL(state.C) && COL(state.C) > COL(state.B) && COL(state.B) < COL(state.A) && COL(state.D) < COL(state.A) && COL(state.A) <= COL(state.C)) ||
            (COL(state.D) > COL(state.C) && COL(state.C) < COL(state.B) && COL(state.B) > COL(state.A) && COL(state.D) > COL(state.A) && COL(state.A) >= COL(state.C)) ||
            // Vertical
            (ROW(state.D) < ROW(state.C) && ROW(state.C) > ROW(state.B) && ROW(state.B) < ROW(state.A) && ROW(state.D) < ROW(state.A) && ROW(state.A) <= ROW(state.C)) ||
            (ROW(state.D) > ROW(state.C) && ROW(state.C) < ROW(state.B) && ROW(state.B) > ROW(state.A) && ROW(state.D) > ROW(state.A) && ROW(state.A) >= ROW(state.C))
            //
        );
    d_out[idx] = ans;
}

// Resets the state if `num_moves` is below cmp.
__global__ void InternalResetIfTerminated_(
    TwosquareState *d_state,
    const int32_t *num_moves,
    const int32_t cmp,
    const uint32_t num_envs)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs)
        return;

    TwosquareState state = d_state[idx];
    if (num_moves[idx] < cmp)
    {
        state.A = state.B = state.C = state.D = 0xff;
    }
    d_state[idx] = state;
}

__global__ void RemoveTwosquareActionsKernel(
    bool *d_action_mask,
    const TwosquareState *d_state,
    const uint8_t *d_mask,
    const uint32_t num_envs)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs || (d_mask && d_mask[idx]))
        return;
    bool *env_action_mask = d_action_mask + idx * NUM_ACTIONS;

    TwosquareState state = d_state[idx];
    assert(state.A == 0xff || (state.A >= 0 && state.A < 100));
    assert(state.B == 0xff || (state.B >= 0 && state.B < 100));
    assert(state.C == 0xff || (state.C >= 0 && state.C < 100));
    assert(state.D == 0xff || (state.D >= 0 && state.D < 100));

    if (state.D != 0xff)
    {
        // Horizontal LTR
        if (COL(state.D) < COL(state.C) && COL(state.C) > COL(state.B) && COL(state.B) < COL(state.A) && COL(state.D) < COL(state.A))
        {
            // Cannot move left of min(COL(A), COL(C)).
            int action_index = 900 + state.A;
            for (int c = 0; c < min(COL(state.C), COL(state.A)); ++c)
            {
                assert(action_index >= 0 && action_index < NUM_ACTIONS);
                env_action_mask[action_index] = false;
                action_index += 100;
            }
        }

        // Horizontal RTL
        if (COL(state.D) > COL(state.C) && COL(state.C) < COL(state.B) && COL(state.B) > COL(state.A) && COL(state.D) > COL(state.A))
        {
            const int start = max(COL(state.C), COL(state.A));

            // Cannot move right of max(COL(A), COL(C)).
            int action_index = 900 + state.A + 100 * start;
            for (int c = start; c < 9; ++c)
            {
                assert(action_index >= 0 && action_index < NUM_ACTIONS);
                env_action_mask[action_index] = false;
                action_index += 100;
            }
        }

        // Vertical TTB
        if (ROW(state.D) < ROW(state.C) && ROW(state.C) > ROW(state.B) && ROW(state.B) < ROW(state.A) && ROW(state.D) < ROW(state.A))
        {
            // Cannot move below min(ROW(A), ROW(C)).
            int action_index = state.A;
            for (int c = 0; c < min(ROW(state.C), ROW(state.A)); ++c)
            {
                assert(action_index >= 0 && action_index < NUM_ACTIONS);
                env_action_mask[action_index] = false;
                action_index += 100;
            }
        }

        // Vertical BTT
        if (ROW(state.D) > ROW(state.C) && ROW(state.C) < ROW(state.B) && ROW(state.B) > ROW(state.A) && ROW(state.D) > ROW(state.A))
        {
            const int start = max(ROW(state.C), ROW(state.A));

            // Cannot move above max(ROW(A), ROW(C)).
            int action_index = state.A + 100 * start;
            for (int c = start; c < 9; ++c)
            {
                assert(action_index >= 0 && action_index < NUM_ACTIONS);
                env_action_mask[action_index] = false;
                action_index += 100;
            }
        }
    }
}

__global__ void EnvStateDeadCellsKernel(
    TwosquareState *d_state,
    const uint8_t player, // 1 = red, 2 = blue
    const StrategoBoard *d_boards,
    const uint8_t *d_terminated_since,
    const uint32_t num_envs)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs || d_terminated_since[idx])
        return;

    TwosquareState state = d_state[idx];
    if (state.A < 100)
    {
        const uint8_t cell = (player == 1) ? state.A : 99 - state.A;
        if ((d_boards[idx].pieces[cell / 10][cell % 10]).color != player)
        {
            state.A = state.B = state.C = state.D = 0xff;
        }
    }
    d_state[idx] = state;
}

void ClearTwosquareState(
    TwosquareState *d_out,
    const uint32_t num_envs)
{
    MUSTRATEGO_CUDA_CHECK(cudaMemsetAsync((void *)d_out, 0xff, num_envs * sizeof(TwosquareState)));
}

void TwosquareStateFromEnvState(
    TwosquareState *d_out_red,
    TwosquareState *d_out_blue,
    const EnvState &env_state,
    const uint32_t num_envs)
{

    const uint32_t move_memory = env_state.action_history.size(0);
    const int32_t *actions = env_state.action_history.data_ptr<int32_t>();
    const int32_t *num_moves = env_state.num_moves.data_ptr<int32_t>();
    MUSTRATEGO_CHECK(move_memory >= 6,
                     "Cannot construct a `TwosquareState` from an `EnvState` with less than six historical moves.");

    ClearTwosquareState(d_out_red, num_envs);
    ClearTwosquareState(d_out_blue, num_envs);

    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs, num_threads);
    for (int i = 0; i < 6; ++i)
    {
        TwosquareState *d_state = ((i + env_state.to_play) % 2 == 0) ? d_out_red : d_out_blue;

        UpdateTwosquareAction(d_state,
                              actions + (move_memory - 6 + i) * num_envs, nullptr, num_envs);
        InternalResetIfTerminated_<<<num_blocks, num_threads>>>(
            d_state,
            num_moves, /* cmp */ 6 - i, num_envs);
    }

    MUSTRATEGO_CHECK((size_t)env_state.boards.data_ptr<uint8_t>() % 128 == 0, "Unexpected alignement of board ptr");
    EnvStateDeadCellsKernel<<<num_blocks, num_threads>>>(d_out_red,
                                                         1, // RED
                                                         (StrategoBoard *)env_state.boards.data_ptr<uint8_t>(),
                                                         env_state.terminated_since.data_ptr<uint8_t>(),
                                                         num_envs);
    EnvStateDeadCellsKernel<<<num_blocks, num_threads>>>(d_out_blue,
                                                         2, // BLUE
                                                         (StrategoBoard *)env_state.boards.data_ptr<uint8_t>(),
                                                         env_state.terminated_since.data_ptr<uint8_t>(),
                                                         num_envs);
}

void UpdateTwosquareAction(
    TwosquareState *d_state,
    const int32_t *d_actions,
    const uint8_t *d_mask,
    const uint32_t num_envs)
{
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs, num_threads);

    UpdateTwosquareActionKernel<<<num_blocks, num_threads>>>(d_state, d_actions, d_mask, num_envs);
}

void UpdateTwosquareDeath(
    TwosquareState *d_state,
    const uint8_t *d_death_cell,
    const uint8_t *d_mask,
    const uint32_t num_envs)
{
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs, num_threads);

    UpdateTwosquareDeathKernel<<<num_blocks, num_threads>>>(d_state, d_death_cell, d_mask, num_envs);
}

void IsTwosquareRuleTriggered(
    bool *d_out,
    const TwosquareState *d_state,
    const uint32_t num_envs)
{
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs, num_threads);

    IsTwosquareRuleTriggeredKernel<<<num_blocks, num_threads>>>(d_out, d_state, num_envs);
}

void IsTwosquareRulePrecludingDirection(
    bool *d_out,
    const TwosquareState *d_state,
    const uint32_t num_envs)
{
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs, num_threads);

    IsTwosquareRulePrecludingDirectionKernel<<<num_blocks, num_threads>>>(d_out, d_state, num_envs);
}

void RemoveTwosquareActions(
    bool *d_action_mask,
    const TwosquareState *d_state,
    const uint8_t *d_mask,
    const uint32_t num_envs)
{
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs, num_threads);

    RemoveTwosquareActionsKernel<<<num_blocks, num_threads>>>(d_action_mask, d_state, d_mask, num_envs);
}

torch::Tensor TwosquareStateAsTensor(
    const TwosquareState *d_state,
    const uint32_t num_envs,
    const uint32_t cuda_device)
{
    return MUSTRATEGO_WRAP_CUDA_TENSOR((uint8_t *)d_state, cuda_device, torch::kUInt8, {num_envs, 4});
}

#undef ROW
#undef COL
