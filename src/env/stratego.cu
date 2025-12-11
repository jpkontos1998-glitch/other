#include "src/env/rules/chase_state.h"
#include "src/env/stratego.h"

#include <ATen/core/ATen_fwd.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/ops/arange.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeTraits.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <random>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/types.h>

#include "src/env/cuda/kernels.h"
#include "src/env/inits/jb_boards.h"
#include "src/env/rules/twosquare_state.h"
#include "src/env/stratego_board.h"
#include "src/env/stratego_conf.h"
#include "src/util.h"

#define GUARD_STEP_ID                                                                                                                                \
    MUSTRATEGO_CHECK(step <= current_step_,                                                                                                          \
                     "%s: Requested step has not been reached yet (requested: %" PRIu64 ", current: %" PRIu64 ")",                                   \
                     __func__, step, current_step_);                                                                                                 \
    MUSTRATEGO_CHECK(step + buf_size > current_step_,                                                                                                \
                     "%s: Requested step is too far in the past (step memory: %" PRIu32 ", requested step: %" PRIu64 ", current step: %" PRIu64 ")", \
                     __func__, buf_size, step, current_step_);
#define GUARD_MEMORY                                                                                                                                                                                       \
    MUSTRATEGO_CHECK(step + buf_size - conf.move_memory > current_step_,                                                                                                                                   \
                     "%s: Requested step is too far in the past considering the MOVE MEMORY (step memory: %" PRIu32 ", move memory: %" PRIu32 ", requested step: %" PRIu64 ", current step: %" PRIu64 ")", \
                     __func__, buf_size, conf.move_memory, step, current_step_);

const uint64_t NO_STEP_YET = std::numeric_limits<uint64_t>::max();
using torch::indexing::None;
using torch::indexing::Slice;

StrategoRolloutBuffer::StrategoRolloutBuffer(
    const uint32_t buf_size,
    const uint32_t num_envs,
    const StrategoConf conf) : buf_size(buf_size),
                               num_envs(num_envs),
                               to_play_(buf_size),
                               conf(conf)
{
    int32_t cuda_device_count = 0;
    MUSTRATEGO_CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    MUSTRATEGO_CHECK(conf.cuda_device < cuda_device_count,
                     "The selected cuda device is unavailable (asked for device %d, but %d devices found)",
                     conf.cuda_device, cuda_device_count);
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    MUSTRATEGO_CHECK(buf_size > 0, "The buffer size cannot be zero");
    MUSTRATEGO_CHECK(buf_size % 2 == 0, "The number of rows in the rollout buffer must be even");
    MUSTRATEGO_CHECK((!conf.two_square_rule || conf.move_memory >= 6),
                     "The move memory must be at least 6 to enforce the two-square rule (found: %d).",
                     conf.move_memory);
    MUSTRATEGO_CHECK(buf_size > conf.move_memory, "The buffer size must be larger than the move memory");
    MUSTRATEGO_CHECK(!conf.continuous_chasing_rule || buf_size >= MAX_CHASE_LENGTH || conf.nonsteppable,
                     "The buffer size is too small to support the continuous chasing rule (found: %d, need: %d)",
                     buf_size, MAX_CHASE_LENGTH);
    if (conf.continuous_chasing_rule && conf.max_num_moves_between_attacks >= MAX_CHASE_LENGTH)
    {
        MUSTRATEGO_FATAL("The maximum number of moves between attacks must be at most %d when the continuous chasing rule is enabled (found: %d)",
                         MAX_CHASE_LENGTH, conf.max_num_moves_between_attacks);
    }

    if (conf.quiet <= 1)
    {
        MUSTRATEGO_LOG("=========================================================");
        MUSTRATEGO_LOG("=                        µSTRATEGO                      =");
        MUSTRATEGO_LOG("=                      Rollout Buffer                   =");
        MUSTRATEGO_LOG("=========================================================");
        MUSTRATEGO_LOG("Env configuration");
        MUSTRATEGO_LOG("  * Buffer size              : %d rows  x  %d envs", buf_size, num_envs);
        MUSTRATEGO_LOG("  * Float dtype              : %s", c10::toString(MUSTRATEGO_FLOAT_TORCH_DTYPE));
        MUSTRATEGO_LOG("  * CUDA device              : %d", conf.cuda_device);
        MUSTRATEGO_LOG("  * Compilation mode         : %s", MUSTRATEGO_COMPILE_MODE);
        MUSTRATEGO_LOG("  * Move memory              : %d", conf.move_memory);
        MUSTRATEGO_LOG("  * Max num moves            : %d", conf.max_num_moves);
        MUSTRATEGO_LOG("  * Reset behavior           : %s", ResetBhToString(conf.reset_behavior));
    }

    NUM_INFOSTATE_CHANNELS = NUM_BOARD_STATE_CHANNELS + conf.move_memory * (4 * (int)conf.enable_hidden_and_types_planes + (int)conf.enable_src_dst_planes + (int)conf.enable_dm_planes);

    // Allocate global tensors
    MUSTRATEGO_CREATE_CUDA_TENSOR(legal_action_mask, conf.cuda_device, torch::kBool, {num_envs, NUM_ACTIONS});
    MUSTRATEGO_CREATE_CUDA_TENSOR(board_state_tensor, conf.cuda_device, MUSTRATEGO_FLOAT_TORCH_DTYPE, {num_envs, NUM_BOARD_STATE_CHANNELS, 10, 10});
    MUSTRATEGO_CREATE_CUDA_TENSOR(move_summary_history_tensor, conf.cuda_device, torch::kUInt8, {conf.move_memory, num_envs, 6});
    MUSTRATEGO_CREATE_CUDA_TENSOR(infostate_tensor, conf.cuda_device, MUSTRATEGO_FLOAT_TORCH_DTYPE, {num_envs, NUM_INFOSTATE_CHANNELS, 10, 10});
    MUSTRATEGO_CREATE_CUDA_TENSOR(reward_pl0, conf.cuda_device, MUSTRATEGO_FLOAT_TORCH_DTYPE, {num_envs});
    MUSTRATEGO_CREATE_CUDA_TENSOR(is_unknown_piece, conf.cuda_device, torch::kBool, {num_envs, 10, 10});
    MUSTRATEGO_CREATE_CUDA_TENSOR(piece_type_onehot, conf.cuda_device, torch::kBool, {num_envs, 10, 10, NUM_PIECE_TYPES});
    MUSTRATEGO_CREATE_CUDA_TENSOR(two_square_rule_applies, conf.cuda_device, torch::kBool, {num_envs});
    MUSTRATEGO_CREATE_CUDA_TENSOR(distrib_, conf.cuda_device, MUSTRATEGO_FLOAT_TORCH_DTYPE, {num_envs, NUM_ACTIONS});
    MUSTRATEGO_CREATE_CUDA_TENSOR(init_offset_red_, conf.cuda_device, torch::kInt32, {buf_size, num_envs});
    MUSTRATEGO_CREATE_CUDA_TENSOR(init_offset_blue_, conf.cuda_device, torch::kInt32, {buf_size, num_envs});
    MUSTRATEGO_CREATE_CUDA_TENSOR(private_action_mask_, conf.cuda_device, torch::kBool, {num_envs, NUM_ACTIONS});

    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_boards_, buf_size * num_envs * sizeof(StrategoBoard)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_zero_boards_, buf_size * num_envs * sizeof(StrategoBoard)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_num_moves_, buf_size * num_envs * sizeof(int32_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_num_moves_since_last_attack_, buf_size * num_envs * sizeof(int32_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_num_moves_since_reset_, buf_size * num_envs * sizeof(int32_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_terminated_since_, buf_size * num_envs * sizeof(uint8_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_flag_captured_, buf_size * num_envs * sizeof(uint8_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_has_legal_movement_, buf_size * num_envs * sizeof(uint8_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_twosquare_state_red_, buf_size * num_envs * sizeof(TwosquareState)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_init_twosquare_state_red_, num_envs * sizeof(TwosquareState)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_twosquare_state_blue_, buf_size * num_envs * sizeof(TwosquareState)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_init_twosquare_state_blue_, num_envs * sizeof(TwosquareState)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_action_history_, buf_size * num_envs * sizeof(int32_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_move_summary_history_, buf_size * num_envs * sizeof(uint8_t) * 6));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_illegal_chase_actions_, buf_size * num_envs * sizeof(int32_t) * MAX_CHASE_LENGTH));
    static_assert(sizeof(bool) == sizeof(uint8_t)); // The scratch tensor can be used as any of those two types
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_scratch_, num_envs * (2 + NUM_ACTIONS) * sizeof(uint8_t)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_unknown_piece_type_onehot_, num_envs * 40 * NUM_PIECE_TYPES * sizeof(bool)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_unknown_piece_has_moved_, num_envs * 40 * sizeof(bool)));
    MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&d_unknown_piece_position_onehot_, num_envs * 40 * 100 * sizeof(bool)));

    if (conf.continuous_chasing_rule)
    {
        MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&(d_chase_state_.last_dst_pos[0]), buf_size * num_envs * sizeof(uint8_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&(d_chase_state_.last_dst_pos[1]), buf_size * num_envs * sizeof(uint8_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&(d_chase_state_.last_src_pos[0]), buf_size * num_envs * sizeof(uint8_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&(d_chase_state_.last_src_pos[1]), buf_size * num_envs * sizeof(uint8_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&(d_chase_state_.chase_length[0]), buf_size * num_envs * sizeof(int32_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMalloc((void **)&(d_chase_state_.chase_length[1]), buf_size * num_envs * sizeof(int32_t)));
    }

    uint64_t TOTAL_BYTES = 0;
#define LOG_ALLOCATION(NAME, BYTES)                                          \
    {                                                                        \
        MUSTRATEGO_LOG("  * " NAME " : %9.4f MBytes", 1ll * (BYTES) * 1e-6); \
        TOTAL_BYTES += 1ll * (BYTES);                                        \
    }

    if (conf.quiet <= 1)
    {
        MUSTRATEGO_LOG("---------------------------------------------------------");
        MUSTRATEGO_LOG("CUDA allocations");
        LOG_ALLOCATION("legal_action_mask               ", legal_action_mask.numel() * sizeof(bool));
        LOG_ALLOCATION("board_state_tensor              ", board_state_tensor.numel() * sizeof(MUSTRATEGO_FLOAT_CUDA_DTYPE));
        LOG_ALLOCATION("move_summary_history_tensor     ", move_summary_history_tensor.numel() * sizeof(uint8_t));
        LOG_ALLOCATION("infostate_tensor                ", infostate_tensor.numel() * sizeof(MUSTRATEGO_FLOAT_CUDA_DTYPE));
        LOG_ALLOCATION("reward_pl0                      ", reward_pl0.numel() * sizeof(MUSTRATEGO_FLOAT_CUDA_DTYPE));
        LOG_ALLOCATION("is_unknown_piece                ", is_unknown_piece.numel() * sizeof(bool));
        LOG_ALLOCATION("piece_type_onehot               ", piece_type_onehot.numel() * sizeof(bool));
        LOG_ALLOCATION("two_square_rule_applies         ", two_square_rule_applies.numel() * sizeof(bool));
        LOG_ALLOCATION("distrib_                        ", distrib_.numel() * sizeof(MUSTRATEGO_FLOAT_CUDA_DTYPE));
        LOG_ALLOCATION("init_offset_red_                ", init_offset_red_.numel() * sizeof(int32_t));
        LOG_ALLOCATION("init_offset_blue_               ", init_offset_blue_.numel() * sizeof(int32_t));
        LOG_ALLOCATION("private_action_mask_            ", private_action_mask_.numel() * sizeof(bool));
        LOG_ALLOCATION("d_boards_                       ", buf_size * num_envs * sizeof(StrategoBoard));
        LOG_ALLOCATION("d_zero_boards_                  ", buf_size * num_envs * sizeof(StrategoBoard));
        LOG_ALLOCATION("d_num_moves_                    ", buf_size * num_envs * sizeof(int32_t));
        LOG_ALLOCATION("d_num_moves_since_last_attack_  ", buf_size * num_envs * sizeof(int32_t));
        LOG_ALLOCATION("d_num_moves_since_reset_        ", buf_size * num_envs * sizeof(int32_t));
        LOG_ALLOCATION("d_terminated_since_             ", buf_size * num_envs * sizeof(uint8_t));
        LOG_ALLOCATION("d_flag_captured_                ", buf_size * num_envs * sizeof(uint8_t));
        LOG_ALLOCATION("d_has_legal_movement_           ", buf_size * num_envs * sizeof(uint8_t));
        LOG_ALLOCATION("d_twosquare_state_red_          ", buf_size * num_envs * sizeof(TwosquareState));
        LOG_ALLOCATION("d_init_twosquare_state_red_     ", num_envs * sizeof(TwosquareState));
        LOG_ALLOCATION("d_twosquare_state_blue_         ", buf_size * num_envs * sizeof(TwosquareState));
        LOG_ALLOCATION("d_init_twosquare_state_blue_    ", num_envs * sizeof(TwosquareState));
        LOG_ALLOCATION("d_action_history_               ", buf_size * num_envs * sizeof(int32_t));
        LOG_ALLOCATION("d_move_summary_history_         ", buf_size * num_envs * 6 * sizeof(uint8_t));
        LOG_ALLOCATION("d_illegal_chase_actions_        ", buf_size * num_envs * MAX_CHASE_LENGTH * sizeof(int32_t));
        LOG_ALLOCATION("d_scratch_                      ", num_envs * (2 + NUM_ACTIONS) * sizeof(uint8_t));
        LOG_ALLOCATION("d_unknown_piece_type_onehot_    ", num_envs * 40 * NUM_PIECE_TYPES * sizeof(bool));
        LOG_ALLOCATION("d_unknown_piece_has_moved_      ", num_envs * 40 * sizeof(bool));
        LOG_ALLOCATION("d_unknown_piece_position_onehot_", num_envs * 40 * 100 * sizeof(bool));
        LOG_ALLOCATION("d_chase_state_.last_dst_pos     ", conf.continuous_chasing_rule ? 2 * buf_size * num_envs * sizeof(uint8_t) : 0);
        LOG_ALLOCATION("d_chase_state_.last_src_pos     ", conf.continuous_chasing_rule ? 2 * buf_size * num_envs * sizeof(uint8_t) : 0);
        LOG_ALLOCATION("d_chase_state_.chase_length     ", conf.continuous_chasing_rule ? 2 * buf_size * num_envs * sizeof(int32_t) : 0);

        MUSTRATEGO_LOG("  > CUDA TOTAL                       : %9.4f MBytes", TOTAL_BYTES * 1e-6);
        MUSTRATEGO_LOG("=========================================================");
    }
#undef LOG_ALLOCATION

    gen_ = at::cuda::detail::createCUDAGenerator();
    if (conf.quiet <= 1)
    {
        MUSTRATEGO_LOG("Constructed CUDA random number generator (seed: %zu)", gen_.seed());
    }

    current_step_ = NO_STEP_YET;

    if (conf.reset_state)
    {
        MUSTRATEGO_CHECK(conf.reset_behavior == CUSTOM_ENV_STATE,
                         "The reset behavior must be CUSTOM_ENV_STATE when a reset state is specified (found: %s)",
                         ResetBhToString(conf.reset_behavior));
        ChangeResetBehavior(*conf.reset_state);
    }
    else if (conf.initial_arrangements)
    {
        MUSTRATEGO_CHECK(conf.reset_behavior == RANDOM_CUSTOM_INITIAL_ARRANGEMENT || conf.reset_behavior == FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT,
                         "The reset behavior must be RANDOM_CUSTOM_INITIAL_ARRANGEMENT or FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT "
                         "when `initial_arrangements` is specified (found: %s)",
                         ResetBhToString(conf.reset_behavior));
        ChangeResetBehavior(*conf.initial_arrangements, conf.initial_arrangements_distrib, ResetBhIsRandomizedBoard(conf.reset_behavior), ResetBhIsFullinfo(conf.reset_behavior));
    }
    else
    {
        MUSTRATEGO_CHECK(!ResetBhIsGenericBoard(conf.reset_behavior),
                         "Missing reset state or initial board tensor for the supplied reset behavior (%s)",
                         ResetBhToString(conf.reset_behavior));
        ChangeResetBehavior(conf.reset_behavior, conf.initial_arrangements_distrib);
    }

    PopulateInfostateChannelDescription_();
    Reset();
}

void StrategoRolloutBuffer::SaveGames(const std::string &outfile)
{
    MUSTRATEGO_CHECK(!game_saver_, "Game saving had already been started. You must call StopSavingGames before starting a new one.");
    MUSTRATEGO_CHECK(current_step_ == 0, "SaveGames called on a non-empty buffer. You must reset the buffer.");
    game_saver_.emplace(num_envs, outfile);
}

void StrategoRolloutBuffer::StopSavingGames()
{
    MUSTRATEGO_CHECK(game_saver_, "StopSavingGames called without a prior call to SaveGames");
    game_saver_->Push(*this, /* force */ true);
    game_saver_.reset();
}

void StrategoRolloutBuffer::ChangeResetBehavior(const ResetBehavior reset_behavior,
                                                const std::optional<std::pair<torch::Tensor, torch::Tensor>> &initial_arrangements_distrib)
{
    conf.reset_behavior = reset_behavior;

    MUSTRATEGO_CHECK(!ResetBhIsGenericBoard(reset_behavior),
                     "This method cannot be called with the given reset behavior (found: %s). Use one of the "
                     "overloaded methods instead",
                     ResetBhToString(reset_behavior));
    MUSTRATEGO_CHECK(ResetBhIsRandomizedBoard(reset_behavior) || !initial_arrangements_distrib,
                     "An initial arrangement distribution cannot be specified with a deterministic board reset behavior");

    // JB's boards are implemented as a special case of the more general `RANDOM_CUSTOM_INITIAL_ARRANGEMENT`.
    // The only difference is that, in order to support deterministic boards, we set the `reset_behavior` below
    // to reflect the choice that was passed here.
    StringArrangements arrangements;
    if (ResetBhIsClassicBoard(conf.reset_behavior))
    {
        arrangements.reserve(JB_INIT_BOARDS_CLASSIC.size());
        for (const auto &arr : JB_INIT_BOARDS_CLASSIC)
            arrangements.emplace_back(arr);
    }
    else if (ResetBhIsBarrageBoard(conf.reset_behavior))
    {
        arrangements.reserve(JB_INIT_BOARDS_BARRAGE.size());
        for (const auto &arr : JB_INIT_BOARDS_BARRAGE)
            arrangements.emplace_back(arr);
    }
    ChangeResetBehavior({arrangements, arrangements}, initial_arrangements_distrib,
                        ResetBhIsRandomizedBoard(conf.reset_behavior),
                        ResetBhIsFullinfo(conf.reset_behavior));
}

void StrategoRolloutBuffer::ChangeResetBehavior(const EnvState &reset_state)
{
    conf.reset_behavior = CUSTOM_ENV_STATE;

    MUSTRATEGO_CHECK(reset_state.num_envs > 0, "The reset state must contain at least one environment");
    MUSTRATEGO_CHECK(reset_state.CudaDevice() == conf.cuda_device,
                     "The reset state device does not match the env deivce (found: %d, expected: %d)",
                     reset_state.CudaDevice(), conf.cuda_device);
    MUSTRATEGO_CHECK(reset_state.num_envs == num_envs,
                     "Mismatching number of envs in reset state (got %d, but rollout buffer has %d)",
                     reset_state.num_envs, num_envs);
    MUSTRATEGO_CHECK(reset_state.action_history.size(0) >= conf.move_memory,
                     "Reset state's move history size (%d) is smaller than "
                     "rollout buffer's configured move memory (%d)",
                     reset_state.action_history.size(0), conf.move_memory);
    MUSTRATEGO_CHECK(!conf.continuous_chasing_rule || reset_state.chase_state.has_value(),
                     "The continuous chasing rule is enabled but the reset state does not contain the chase state. "
                     "This usually happens when the reset state was created by snapshotting a step that was not current.");
    MUSTRATEGO_CHECK(reset_state.board_history.size(0) >= conf.move_memory,
                     "Reset state's board history size (%d) does not match the rollout buffer's move memory (%d)",
                     reset_state.board_history.size(0), conf.move_memory);

    MUSTRATEGO_CHECK(reset_state.move_summary_history.size(0) == conf.move_memory, "Inconsistent memory sizes");
    MUSTRATEGO_CHECK(reset_state.action_history.size(0) == conf.move_memory, "Inconsistent memory sizes");

    const uint8_t max_reset_since = reset_state.terminated_since.max().item<uint8_t>();

    if (max_reset_since > 250)
    {
        MUSTRATEGO_FATAL("The specified reset state has at least one env whose `terminated_since` is above 250. "
                         "terminated_since > 3 can happen when taking one action, snapshotting, setting the reset state "
                         "to that, taking an action, etc. However, a very large value might be a symptom that something "
                         "unusual is going on. Since terminated_since is defined as an 8-bit unsigned integer, we need "
                         "to abort.");
    }
    else if (max_reset_since > 0)
    {
        // Aug 1st, 2024: This used to be a warning but it is now a fatal error.
        // This is because we realized that it is unlikely this would be hit in one of
        // our applications, and it is more likely that it will cause suble issues downstream.
        // ---------------------------------------------
        // Apr 22th, 2024: We are changing this back to a warning to support search during training.
        if (!conf.quiet)
        {
            MUSTRATEGO_WARN("The specified reset state contains at least one environment that has terminated");
        }
    }

    // We clone the reset state so that we are sure that the reset state cannot be accidentally modified externally.
    conf.reset_state.emplace(reset_state.Clone());
    conf.initial_arrangements = std::nullopt;
    conf.initial_arrangements_distrib = std::nullopt;

    init_boards_ = conf.reset_state->boards;
    init_zero_boards_ = conf.reset_state->zero_boards;
    MUSTRATEGO_CHECK(size_t(init_boards_.data_ptr<uint8_t>()) % 128 == 0,
                     "INTERNAL BUG: Unexpected alignment for `init_boards` data pointer");
    MUSTRATEGO_CHECK(size_t(init_zero_boards_.data_ptr<uint8_t>()) % 128 == 0,
                     "INTERNAL BUG: Unexpected alignment for `init_zero_boards` data pointer");

    red_init_modulus_ = 1;
    blue_init_modulus_ = 1;

    TwosquareStateFromEnvState(
        d_init_twosquare_state_red_,
        d_init_twosquare_state_blue_,
        *conf.reset_state,
        num_envs);

    RandomizeBoardInitOffsets_();

    if (!conf.quiet)
    {
        MUSTRATEGO_DEBUG("Installing a reset state triggers a hard reset");
    }
    Reset();
}

void StrategoRolloutBuffer::ChangeResetBehavior(const std::pair<StringArrangements, StringArrangements> &initial_arrangements,
                                                const std::optional<std::pair<torch::Tensor, torch::Tensor>> &initial_arrangements_distrib,
                                                const bool randomize,
                                                const bool fullinfo)
{
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    MUSTRATEGO_CHECK(!initial_arrangements.first.empty(), "The list of red initial arrangements is empty");
    MUSTRATEGO_CHECK(!initial_arrangements.second.empty(), "The list of blue initial arrangements is empty");
    if (initial_arrangements_distrib)
    {
        MUSTRATEGO_CHECK(randomize, "An initial arrangement distribution can only be specified when randomize is set to True");
        MUSTRATEGO_CHECK_CUDA_DTYPE(initial_arrangements_distrib->first, conf.cuda_device, torch::kFloat32, "Invalid type or device for red distribution");
        MUSTRATEGO_CHECK_CUDA_DTYPE(initial_arrangements_distrib->second, conf.cuda_device, torch::kFloat32, "Invalid type or device for blue distribution");
        MUSTRATEGO_CHECK(initial_arrangements_distrib->first.sizes().equals({int64_t(initial_arrangements.first.size())}),
                         "Red distribution shape does not match the number of initial arrangements (expected: {%zu})",
                         initial_arrangements.first.size());
        MUSTRATEGO_CHECK(initial_arrangements_distrib->second.sizes().equals({int64_t(initial_arrangements.second.size())}),
                         "Blue distribution shape does not match the number of initial arrangements (expected: {%zu})",
                         initial_arrangements.second.size());
        MUSTRATEGO_CHECK_DISTRIBUTION(initial_arrangements_distrib->first);
        MUSTRATEGO_CHECK_DISTRIBUTION(initial_arrangements_distrib->second);
    }

    if (!randomize)
    {
        MUSTRATEGO_CHECK(initial_arrangements.first.size() == initial_arrangements.second.size(),
                         "The number of red and blue initial arrangements must be the same when randomize is set to False");
        MUSTRATEGO_CHECK(initial_arrangements.first.size() % num_envs == 0,
                         "The number of initial arrangements must be a multiple of the number of environments when randomize is set to False");
    }

    // We start by making a copy of the arrangements, since we might need to pad them so that
    // they have the same length.
    StringArrangements red_arrangements = initial_arrangements.first;
    StringArrangements blue_arrangements = initial_arrangements.second;
    while (red_arrangements.size() < blue_arrangements.size())
        red_arrangements.push_back(red_arrangements.back());
    while (blue_arrangements.size() < red_arrangements.size())
        blue_arrangements.push_back(blue_arrangements.back());

    const ResetBehavior new_reset_behavior = randomize ? (fullinfo ? FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT : RANDOM_CUSTOM_INITIAL_ARRANGEMENT)
                                                       : (fullinfo ? FULLINFO_STEP_CUSTOM_INITIAL_ARRANGEMENT : STEP_CUSTOM_INITIAL_ARRANGEMENT);
    const bool needs_reset = !!conf.reset_state;

    if (!randomize && red_arrangements.size() != blue_arrangements.size())
    {
        MUSTRATEGO_WARN("The number of red and blue initial arrangements is different but randomize is set to False. "
                        "Was this intended?");
    }

    conf.reset_behavior = new_reset_behavior;
    conf.reset_state = std::nullopt;
    conf.initial_arrangements = initial_arrangements;
    conf.initial_arrangements_distrib = initial_arrangements_distrib;
    if (conf.initial_arrangements_distrib)
    {
        // We clone the tensors to prevent outside modification.
        conf.initial_arrangements_distrib->first = conf.initial_arrangements_distrib->first.clone();
        conf.initial_arrangements_distrib->second = conf.initial_arrangements_distrib->second.clone();
    }

    init_boards_ = GenerateInitializationBoards(ArrangementTensorFromStrings(red_arrangements),
                                                ArrangementTensorFromStrings(blue_arrangements),
                                                conf.cuda_device);
    init_zero_boards_ = init_boards_;
    MUSTRATEGO_CHECK(size_t(init_boards_.data_ptr<uint8_t>()) % 128 == 0,
                     "INTERNAL BUG: Unexpected alignment for `init_boards` data pointer");

    ClearTwosquareState(d_init_twosquare_state_red_, num_envs);
    ClearTwosquareState(d_init_twosquare_state_blue_, num_envs);

    red_init_modulus_ = initial_arrangements.first.size();   // IMPORTANT: we are using the non-padded version
    blue_init_modulus_ = initial_arrangements.second.size(); // IMPORTANT: we are using the non-padded version

    RandomizeBoardInitOffsets_();

    if (needs_reset)
    {
        if (!conf.quiet)
        {
            MUSTRATEGO_DEBUG("Uninstalling a reset state triggers a hard reset");
        }
        Reset();
    }
}

void StrategoRolloutBuffer::Reset()
{

    MUSTRATEGO_CHECK(size_t(init_boards_.data_ptr<uint8_t>()) % 128 == 0,
                     "INTERNAL BUG: Unexpected alignment for `init_boards` data pointer");
    MUSTRATEGO_CHECK(size_t(init_zero_boards_.data_ptr<uint8_t>()) % 128 == 0,
                     "INTERNAL BUG: Unexpected alignment for `init_zero_boards` data pointer");
    if (game_saver_)
    {
        game_saver_->Push(*this, /* force */ true);
        game_saver_->Reset();
    }

    if (current_step_ != NO_STEP_YET && conf.verbose && !conf.quiet)
    {
        MUSTRATEGO_DEBUG("Resetting rollout buffer (erasing %" PRIu64 " steps)", current_step_ + 1);
    }
    current_step_ = 0;

    RandomizeBoardInitOffsets_();

    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs * sizeof(StrategoBoard), 4 * num_threads);
    InitBoardsKernel<<<num_blocks, num_threads>>>(
        d_boards_,
        (const StrategoBoard *)init_boards_.data_ptr<uint8_t>(),
        init_offset_red_.data_ptr<int32_t>(),
        init_offset_blue_.data_ptr<int32_t>(),
        num_envs,
        /* make_pieces_visible */ ResetBhIsFullinfo(conf.reset_behavior));

    if (conf.reset_behavior != CUSTOM_ENV_STATE)
    {
        to_play_[0] = 1;
        d_init_chase_state_.reset();

        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_zero_boards_, d_boards_, num_envs * sizeof(StrategoBoard), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemsetAsync(d_num_moves_, 0, num_envs * sizeof(int32_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMemsetAsync(d_num_moves_since_reset_, 0, num_envs * sizeof(int32_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMemsetAsync(d_num_moves_since_last_attack_, 0, num_envs * sizeof(int32_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMemsetAsync(d_terminated_since_, 0, num_envs * sizeof(uint8_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMemsetAsync(d_flag_captured_, 0, num_envs * sizeof(uint8_t)));

        if (conf.continuous_chasing_rule)
        {
            ResetChaseState(d_chase_state_, d_init_chase_state_, num_envs, conf.cuda_device);
            // At this stage, there cannot be any illegal chase actions.
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_illegal_chase_actions_, conf.cuda_device, torch::kInt32, {MAX_CHASE_LENGTH, num_envs})
                .fill_(-1);
        }
    }
    else
    {
        assert(conf.reset_state);
        to_play_[0] = conf.reset_state->to_play + 1;

        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_zero_boards_, conf.reset_state->zero_boards.data_ptr<uint8_t>(), num_envs * sizeof(StrategoBoard), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_num_moves_, conf.reset_state->num_moves.data_ptr<int32_t>(), num_envs * sizeof(int32_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemsetAsync(d_num_moves_since_reset_, 0, num_envs * sizeof(int32_t)));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_num_moves_since_last_attack_, conf.reset_state->num_moves_since_last_attack.data_ptr<int32_t>(), num_envs * sizeof(int32_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_terminated_since_, conf.reset_state->terminated_since.data_ptr<uint8_t>(), num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_flag_captured_, conf.reset_state->flag_captured.data_ptr<uint8_t>(), num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_has_legal_movement_, conf.reset_state->has_legal_movement.data_ptr<uint8_t>(), num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));

        if (conf.continuous_chasing_rule)
        {
            MUSTRATEGO_CHECK(conf.reset_state->chase_state.has_value(), "INTERNAL BUG: missing chase state in reset state");
            d_init_chase_state_.emplace();
            d_init_chase_state_->last_dst_pos[0] = conf.reset_state->chase_state->last_dst_pos[0].data_ptr<uint8_t>();
            d_init_chase_state_->last_dst_pos[1] = conf.reset_state->chase_state->last_dst_pos[1].data_ptr<uint8_t>();
            d_init_chase_state_->last_src_pos[0] = conf.reset_state->chase_state->last_src_pos[0].data_ptr<uint8_t>();
            d_init_chase_state_->last_src_pos[1] = conf.reset_state->chase_state->last_src_pos[1].data_ptr<uint8_t>();
            d_init_chase_state_->chase_length[0] = conf.reset_state->chase_state->chase_length[0].data_ptr<int32_t>();
            d_init_chase_state_->chase_length[1] = conf.reset_state->chase_state->chase_length[1].data_ptr<int32_t>();

            MUSTRATEGO_CHECK((size_t)conf.reset_state->board_history.data_ptr<uint8_t>() % 128 == 0,
                             "INTERNAL BUG: Unexpected alignment for `board_history` data pointer");

            ResetChaseState(d_chase_state_, d_init_chase_state_, num_envs, conf.cuda_device);
            // We need to recompute the illegal chase moves.
            ComputeIllegalChaseMoves(
                /* d_out */ d_illegal_chase_actions_,
                /* d_state */ d_chase_state_,
                /* d_board_history */ d_boards_,
                /* d_board_prehistory */ (StrategoBoard *)conf.reset_state->board_history.data_ptr<uint8_t>(),
                /* d_num_moves_since_reset */ d_num_moves_since_reset_,
                /* d_mask */ d_terminated_since_,
                /* board_cur_index */ current_step_, // = 0
                /* history_buf_length */ buf_size,
                /* player */ to_play_[0],
                /* prehistory_size */ conf.reset_state->board_history.size(0),
                /* num_envs */ num_envs,
                /* cuda_device */ conf.cuda_device);
        }
    }

    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_twosquare_state_red_, d_init_twosquare_state_red_, num_envs * sizeof(TwosquareState), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(d_twosquare_state_blue_, d_init_twosquare_state_blue_, num_envs * sizeof(TwosquareState), cudaMemcpyDeviceToDevice));

    // Finally, we compute the legal movements for the initial
    // board (this is used by the reward computation kernel).
    UpdateHasLegalMovement_(0);
    if (conf.reset_behavior != CUSTOM_ENV_STATE)
    {
        MUSTRATEGO_CHECK(
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_has_legal_movement_, conf.cuda_device, torch::kUInt8, {num_envs}).eq(3).all().item<bool>(),
            "INTERNAL BUG: `d_has_legal_movement_` should be 3 for all environments after reset");
    }
}

void StrategoRolloutBuffer::RandomizeBoardInitOffsets_()
{
    if (ResetBhIsRandomizedBoard(conf.reset_behavior))
    {
        assert(red_init_modulus_ <= init_boards_.size(0));
        assert(blue_init_modulus_ <= init_boards_.size(0));

        if (!conf.initial_arrangements_distrib)
        {
            torch::randint_out(init_offset_red_, red_init_modulus_, {buf_size, num_envs});
            torch::randint_out(init_offset_blue_, blue_init_modulus_, {buf_size, num_envs});
        }
        else
        {
            assert(conf.initial_arrangements_distrib->first.sizes() == red_init_modulus_);
            assert(conf.initial_arrangements_distrib->second.sizes() == blue_init_modulus_);

            SampleOut(conf.initial_arrangements_distrib->first, init_offset_red_, conf.cuda_device);
            SampleOut(conf.initial_arrangements_distrib->second, init_offset_blue_, conf.cuda_device);

            assert((init_offset_red_ <= int32_t(red_init_modulus_)).all().item<bool>());
            assert((init_offset_blue_ <= int32_t(blue_init_modulus_)).all().item<bool>());
        }
    }
    else
    {
        // Set each column to 0, 1, ..., num_envs - 1.
        torch::arange_out(init_offset_red_, int32_t(buf_size * num_envs));
        torch::arange_out(init_offset_blue_, int32_t(buf_size * num_envs));

        if (conf.initial_arrangements)
        {
            init_offset_red_.remainder_((int32_t)conf.initial_arrangements->first.size());
            init_offset_blue_.remainder_((int32_t)conf.initial_arrangements->second.size());
        }
        else
        {
            init_offset_red_.remainder_((int32_t)num_envs);
            init_offset_blue_.remainder_((int32_t)num_envs);
        }
    }

    assert(init_offset_red_.sizes().equals({buf_size, num_envs}));
    assert(init_offset_blue_.sizes().equals({buf_size, num_envs}));
}

StrategoRolloutBuffer::~StrategoRolloutBuffer()
{
    if (game_saver_)
    {
        game_saver_->Push(*this, /* force */ true);
    }

    if (conf.quiet <= 1)
    {
        MUSTRATEGO_LOG("Tearing down rollout buffer");
    }
    if (conf.continuous_chasing_rule)
    {
        MUSTRATEGO_CUDA_CHECK(cudaFree(d_chase_state_.chase_length[1]));
        MUSTRATEGO_CUDA_CHECK(cudaFree(d_chase_state_.chase_length[0]));
        MUSTRATEGO_CUDA_CHECK(cudaFree(d_chase_state_.last_src_pos[1]));
        MUSTRATEGO_CUDA_CHECK(cudaFree(d_chase_state_.last_src_pos[0]));
        MUSTRATEGO_CUDA_CHECK(cudaFree(d_chase_state_.last_dst_pos[1]));
        MUSTRATEGO_CUDA_CHECK(cudaFree(d_chase_state_.last_dst_pos[0]));
    }
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_unknown_piece_position_onehot_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_unknown_piece_has_moved_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_unknown_piece_type_onehot_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_scratch_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_illegal_chase_actions_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_move_summary_history_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_action_history_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_init_twosquare_state_blue_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_twosquare_state_blue_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_init_twosquare_state_red_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_twosquare_state_red_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_has_legal_movement_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_flag_captured_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_terminated_since_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_num_moves_since_reset_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_num_moves_since_last_attack_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_num_moves_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_zero_boards_));
    MUSTRATEGO_CUDA_CHECK(cudaFree(d_boards_));
}

uint8_t StrategoRolloutBuffer::ActingPlayer(const uint64_t step) const
{
    GUARD_STEP_ID

    return to_play_[step % buf_size] - 1;
}

uint64_t StrategoRolloutBuffer::ApplyActions(const torch::Tensor actions)
{
    MUSTRATEGO_CHECK(!conf.nonsteppable, "Called ApplyActions on nonsteppable buffer");
    MUSTRATEGO_CHECK_CUDA_DTYPE(actions, conf.cuda_device, torch::kInt32,
                                "Action tensor must be CUDA-allocated on device %d and of dtype `torch.int32`",
                                conf.cuda_device);
    if (actions.sizes() != torch::IntArrayRef({num_envs}))
    {
        MUSTRATEGO_FATAL("`action` tensor should have shape (%d,)", num_envs);
    }

    { // Make sure all selected actions are legal.
        private_action_mask_.zero_();
        ComputeLegalActionMask_(current_step_, private_action_mask_.data_ptr<bool>());

        torch::Tensor scratch =
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kBool, {num_envs, 1});
        at::gather_out(scratch, private_action_mask_, /* dim */ 1, actions.to(torch::kInt64).view({num_envs, 1}));

        if (!scratch.all().item<bool>())
        {
            MUSTRATEGO_WARN("Illegal action check is failing. Debug information:");
            MUSTRATEGO_WARN("*** Acting player: %s", (to_play_[CurrentRowId_()] == 1) ? "RED" : "BLUE");
            MUSTRATEGO_WARN("*** Actions selected:");
            std::cout << actions.reshape({1, -1}) << std::endl;
            MUSTRATEGO_WARN("*** Envs with illegal actions selected:");
            std::cout << (~scratch).reshape({1, -1}) << std::endl;
            MUSTRATEGO_WARN("*** Termination flags (`terminated_since`):");
            uint8_t *terminated_since = d_terminated_since_ + (current_step_ % buf_size) * num_envs;
            std::cout << MUSTRATEGO_WRAP_CUDA_TENSOR(terminated_since, conf.cuda_device, torch::kUInt8, {1, num_envs}) << std::endl;
            MUSTRATEGO_WARN("*** Red / blue twosquare state:");
            const TwosquareState *twosquare_state_red = d_twosquare_state_red_ + (current_step_ % buf_size) * num_envs;
            const TwosquareState *twosquare_state_blue = d_twosquare_state_blue_ + (current_step_ % buf_size) * num_envs;
            std::cout << "RED  : " << TwosquareStateAsTensor(twosquare_state_red, num_envs, conf.cuda_device) << std::endl;
            std::cout << "BLUE : " << TwosquareStateAsTensor(twosquare_state_blue, num_envs, conf.cuda_device) << std::endl;

            MUSTRATEGO_FATAL("One or more actions are illegal");
        }
    }

    const int32_t *actions_ptr = actions.data_ptr<int32_t>();
    const uint32_t next_row_id = (current_step_ + 1) % buf_size;
    const int32_t *action_history_ptr = d_action_history_ + (current_step_ % buf_size) * num_envs;
    uint8_t *move_summary_ptr = d_move_summary_history_ + (current_step_ % buf_size) * num_envs * 6;

    if (conf.reset_behavior == STEP_CUSTOM_INITIAL_ARRANGEMENT ||
        conf.reset_behavior == FULLINFO_STEP_CUSTOM_INITIAL_ARRANGEMENT)
    {
        torch::Tensor scratch = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kInt32, {num_envs});
        scratch.copy_(GetTerminatedSince(current_step_));
        scratch.eq_(1);

        torch::Tensor red = init_offset_red_.index({(int)next_row_id});
        red.copy_(init_offset_red_.index({(int)(current_step_ % buf_size)}));
        red.add_(scratch, /* alpha */ (int32_t)num_envs);
        red.remainder_((int32_t)conf.initial_arrangements->first.size());

        torch::Tensor blue = init_offset_blue_.index({(int)next_row_id});
        blue.copy_(init_offset_blue_.index({(int)(current_step_ % buf_size)}));
        blue.add_(scratch, /* alpha */ (int32_t)num_envs);
        blue.remainder_((int32_t)conf.initial_arrangements->second.size());
    }

    MUSTRATEGO_CUDA_CHECK(cudaMemcpy(
        (void *)action_history_ptr,
        actions_ptr,
        num_envs * sizeof(int32_t),
        cudaMemcpyDeviceToDevice));

    const uint32_t to_play = to_play_[current_step_ % buf_size];
    const uint32_t out_to_play_ = 3 - to_play;
    to_play_[(current_step_ + 1) % buf_size] = out_to_play_;

    const StrategoBoard *boards = d_boards_ + (current_step_ % buf_size) * num_envs;
    const StrategoBoard *zero_boards = d_zero_boards_ + (current_step_ % buf_size) * num_envs;
    const TwosquareState *twosquare_state_red = d_twosquare_state_red_ + (current_step_ % buf_size) * num_envs;
    const TwosquareState *twosquare_state_blue = d_twosquare_state_blue_ + (current_step_ % buf_size) * num_envs;
    const uint8_t *has_legal_movement = d_has_legal_movement_ + (current_step_ % buf_size) * num_envs;
    const uint8_t *flag_captured = d_flag_captured_ + (current_step_ % buf_size) * num_envs;
    const uint8_t *terminated_since = d_terminated_since_ + (current_step_ % buf_size) * num_envs;
    const int32_t *num_moves = d_num_moves_ + (current_step_ % buf_size) * num_envs;
    const int32_t *num_moves_since_last_attack = d_num_moves_since_last_attack_ + (current_step_ % buf_size) * num_envs;
    const int32_t *num_moves_since_reset = d_num_moves_since_reset_ + (current_step_ % buf_size) * num_envs;
    ChaseState chase_state;
    if (conf.continuous_chasing_rule)
    {
        chase_state.last_dst_pos[0] = d_chase_state_.last_dst_pos[0] + (current_step_ % buf_size) * num_envs;
        chase_state.last_dst_pos[1] = d_chase_state_.last_dst_pos[1] + (current_step_ % buf_size) * num_envs;
        chase_state.last_src_pos[0] = d_chase_state_.last_src_pos[0] + (current_step_ % buf_size) * num_envs;
        chase_state.last_src_pos[1] = d_chase_state_.last_src_pos[1] + (current_step_ % buf_size) * num_envs;
        chase_state.chase_length[0] = d_chase_state_.chase_length[0] + (current_step_ % buf_size) * num_envs;
        chase_state.chase_length[1] = d_chase_state_.chase_length[1] + (current_step_ % buf_size) * num_envs;
    }

    StrategoBoard *out_boards = d_boards_ + ((current_step_ + 1) % buf_size) * num_envs;
    StrategoBoard *out_zero_boards = d_zero_boards_ + ((current_step_ + 1) % buf_size) * num_envs;
    uint8_t *out_has_legal_movement = d_has_legal_movement_ + ((current_step_ + 1) % buf_size) * num_envs;
    uint8_t *out_flag_captured = d_flag_captured_ + ((current_step_ + 1) % buf_size) * num_envs;
    uint8_t *out_terminated_since = d_terminated_since_ + ((current_step_ + 1) % buf_size) * num_envs;
    int32_t *out_num_moves = d_num_moves_ + ((current_step_ + 1) % buf_size) * num_envs;
    int32_t *out_num_moves_since_last_attack = d_num_moves_since_last_attack_ + ((current_step_ + 1) % buf_size) * num_envs;
    int32_t *out_num_moves_since_reset = d_num_moves_since_reset_ + ((current_step_ + 1) % buf_size) * num_envs;
    int32_t *out_illegal_chase_actions = d_illegal_chase_actions_ + ((current_step_ + 1) % buf_size) * num_envs * MAX_CHASE_LENGTH;
    ChaseState out_chase_state;
    if (conf.continuous_chasing_rule)
    {
        out_chase_state.last_dst_pos[0] = d_chase_state_.last_dst_pos[0] + ((current_step_ + 1) % buf_size) * num_envs;
        out_chase_state.last_dst_pos[1] = d_chase_state_.last_dst_pos[1] + ((current_step_ + 1) % buf_size) * num_envs;
        out_chase_state.last_src_pos[0] = d_chase_state_.last_src_pos[0] + ((current_step_ + 1) % buf_size) * num_envs;
        out_chase_state.last_src_pos[1] = d_chase_state_.last_src_pos[1] + ((current_step_ + 1) % buf_size) * num_envs;
        out_chase_state.chase_length[0] = d_chase_state_.chase_length[0] + ((current_step_ + 1) % buf_size) * num_envs;
        out_chase_state.chase_length[1] = d_chase_state_.chase_length[1] + ((current_step_ + 1) % buf_size) * num_envs;
    }

    TwosquareState *out_twosquare_state_red = d_twosquare_state_red_ + ((current_step_ + 1) % buf_size) * num_envs;
    TwosquareState *out_twosquare_state_blue = d_twosquare_state_blue_ + ((current_step_ + 1) % buf_size) * num_envs;

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_boards, boards, num_envs * sizeof(StrategoBoard), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_zero_boards, zero_boards, num_envs * sizeof(StrategoBoard), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_num_moves, num_moves, num_envs * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_num_moves_since_last_attack, num_moves_since_last_attack, num_envs * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_num_moves_since_reset, num_moves_since_reset, num_envs * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_terminated_since, terminated_since, num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_flag_captured, flag_captured, num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_has_legal_movement, has_legal_movement, num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_twosquare_state_red, twosquare_state_red, num_envs * sizeof(TwosquareState), cudaMemcpyDeviceToDevice));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_twosquare_state_blue, twosquare_state_blue, num_envs * sizeof(TwosquareState), cudaMemcpyDeviceToDevice));
    if (conf.continuous_chasing_rule)
    {
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_chase_state.last_dst_pos[0], chase_state.last_dst_pos[0], num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_chase_state.last_dst_pos[1], chase_state.last_dst_pos[1], num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_chase_state.last_src_pos[0], chase_state.last_src_pos[0], num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_chase_state.last_src_pos[1], chase_state.last_src_pos[1], num_envs * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_chase_state.chase_length[0], chase_state.chase_length[0], num_envs * sizeof(int32_t), cudaMemcpyDeviceToDevice));
        MUSTRATEGO_CUDA_CHECK(cudaMemcpyAsync(out_chase_state.chase_length[1], chase_state.chase_length[1], num_envs * sizeof(int32_t), cudaMemcpyDeviceToDevice));
    }

    {
        const uint32_t num_threads = 256;
        const uint32_t num_blocks = ceil(num_envs, num_threads);

        UpdateTwosquareAction(to_play == 1 ? out_twosquare_state_red : out_twosquare_state_blue,
                              actions_ptr,
                              out_terminated_since,
                              num_envs);
        uint8_t *red_death = d_scratch_;
        uint8_t *blue_death = d_scratch_ + 64 * num_envs;

        ApplyActionsKernel<<<num_blocks, num_threads>>>(
            out_boards,
            out_num_moves,
            out_num_moves_since_last_attack,
            out_num_moves_since_reset,
            out_flag_captured,
            move_summary_ptr,
            red_death,
            blue_death,
            out_terminated_since,
            num_envs,
            to_play, // the ACTING player
            actions_ptr);

        UpdateTwosquareDeath(out_twosquare_state_red, red_death, out_terminated_since, num_envs);
        UpdateTwosquareDeath(out_twosquare_state_blue, blue_death, out_terminated_since, num_envs);

        if (conf.continuous_chasing_rule)
        {
            UpdateChaseState(
                /* state */ out_chase_state,
                /* board */ out_boards,
                /* move_summary */ move_summary_ptr,
                /* mask */ out_terminated_since,
                /* player */ to_play,
                num_envs,
                conf.cuda_device);
        }
    }

    // We only consider resetting the environment if the player matches the player of the reset state.
    const uint8_t reset_player = conf.reset_state ? (conf.reset_state->to_play + 1) : 1;
    if (out_to_play_ == reset_player)
    {
        const StrategoBoard *out_init_boards = (const StrategoBoard *)init_boards_.data_ptr<uint8_t>();
        const StrategoBoard *out_zero_init_boards = (const StrategoBoard *)init_zero_boards_.data_ptr<uint8_t>();
        const int32_t *out_init_offset_red = init_offset_red_.data_ptr<int32_t>() + ((current_step_ + 1) % buf_size) * num_envs;
        const int32_t *out_init_offset_blue = init_offset_blue_.data_ptr<int32_t>() + ((current_step_ + 1) % buf_size) * num_envs;
        {
            const uint32_t num_threads = 1024;
            const uint32_t num_blocks = ceil(num_envs * sizeof(StrategoBoard), 4 * num_threads);

            ResetTerminatedBoardsKernel<<<num_blocks, num_threads>>>(
                out_boards,
                out_zero_boards,
                out_num_moves,
                out_num_moves_since_last_attack,
                out_num_moves_since_reset,
                out_terminated_since,
                out_flag_captured,
                out_twosquare_state_red,
                out_twosquare_state_blue,
                out_init_boards,
                out_zero_init_boards,
                out_init_offset_red,
                out_init_offset_blue,
                num_envs,
                /* make_pieces_visible */ ResetBhIsFullinfo(conf.reset_behavior),
                conf.reset_state ? conf.reset_state->num_moves.data_ptr<int32_t>() : nullptr,
                conf.reset_state ? conf.reset_state->num_moves_since_last_attack.data_ptr<int32_t>() : nullptr,
                conf.reset_state ? conf.reset_state->terminated_since.data_ptr<uint8_t>() : nullptr,
                conf.reset_state ? conf.reset_state->flag_captured.data_ptr<uint8_t>() : nullptr,
                d_init_twosquare_state_red_,
                d_init_twosquare_state_blue_);
        }
        {
            const uint32_t num_threads = 512;
            const uint32_t num_blocks = ceil(num_envs, num_threads);

            ResetTerminationCountersKernel<<<num_blocks, num_threads>>>(
                out_terminated_since,
                conf.reset_state ? conf.reset_state->terminated_since.data_ptr<uint8_t>() : nullptr,
                num_envs);
        }

        if (conf.continuous_chasing_rule)
        {
            ResetChaseState(
                /* d_out */ out_chase_state,
                /* d_src */ d_init_chase_state_,
                num_envs,
                conf.cuda_device,
                /* mask */ out_num_moves_since_reset);
        }
    }

    // Compute the illegal chase actions
    if (conf.continuous_chasing_rule)
    {
        ComputeIllegalChaseMoves(
            /* d_out */ out_illegal_chase_actions,
            /* d_state */ out_chase_state,
            /* d_board_history */ d_boards_,
            /* d_board_prehistory */ conf.reset_state ? (StrategoBoard *)conf.reset_state->board_history.data_ptr<uint8_t>() : nullptr,
            /* d_num_moves_since_reset */ d_num_moves_since_reset_,
            /* d_mask */ out_terminated_since,
            /* board_cur_index */ (current_step_ + 1),
            /* history_buf_length */ buf_size,
            /* player */ out_to_play_,
            /* prehistory_size */ conf.reset_state ? conf.reset_state->board_history.size(0) : 0,
            num_envs,
            conf.cuda_device);
    }

    ++current_step_;
    // Determine whether the player has legal moves
    UpdateHasLegalMovement_(current_step_);

    // Determine whether the environment has terminated and possibly increment the counter
    {
        const uint32_t num_threads = 1024;
        const uint32_t num_blocks = ceil(num_envs, num_threads);

        IncrementTerminationCounterKernel<<<num_blocks, num_threads>>>(
            out_terminated_since,
            out_flag_captured,
            out_num_moves,
            out_num_moves_since_last_attack,
            out_num_moves_since_reset,
            out_has_legal_movement,
            num_envs,
            conf.max_num_moves,
            conf.max_num_moves_between_attacks);
    }

    if (!next_row_id && ResetBhIsRandomizedBoard(conf.reset_behavior))
    {
        RandomizeBoardInitOffsets_();
    }

    if (game_saver_ && (current_step_ + 1) % buf_size == 0)
    {
        game_saver_->Push(*this, /* force */ false);
    }

    return current_step_;
}

void StrategoRolloutBuffer::SeedActionSampler(const uint64_t seed)
{
    if (conf.quiet <= 1)
    {
        MUSTRATEGO_LOG("Re-seeding action sampler (seed: %zu). This triggers a reset of the rollout buffer", seed);
    }
    gen_.set_current_seed(seed);
    Reset();
}

void StrategoRolloutBuffer::SampleRandomLegalAction(torch::Tensor actions_out)
{
    MUSTRATEGO_CHECK_CUDA_DTYPE(actions_out, conf.cuda_device, torch::kInt32,
                                "Action tensor must be CUDA-allocated on device %d and of dtype `torch.int32`",
                                conf.cuda_device);
    if (actions_out.sizes() != torch::IntArrayRef({num_envs}))
        MUSTRATEGO_FATAL("Output action tensor has wrong shape (must be: (%d,))", num_envs);

    private_action_mask_.zero_();
    ComputeLegalActionMask_(current_step_, private_action_mask_.data_ptr<bool>());

    distrib_.uniform_(0.0, 0.9, gen_);
    distrib_ += private_action_mask_;
    auto iter =
        at::meta::make_reduction(distrib_, actions_out, /*dims=*/{1}, /*keepdim=*/false, MUSTRATEGO_FLOAT_TORCH_DTYPE);
    at::native::gpu_reduce_kernel<MUSTRATEGO_FLOAT_CUDA_DTYPE, int32_t>(
        iter,
        at::native::ArgMaxOps<MUSTRATEGO_FLOAT_CUDA_DTYPE>{},
        thrust::pair<MUSTRATEGO_FLOAT_CUDA_DTYPE, int32_t>(0, 0));

#ifdef DEBUG
    {
        torch::Tensor scratch =
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kBool, {num_envs, 1});
        at::gather_out(scratch, private_action_mask_, /* dim */ 1, actions_out.to(torch::kInt64).view({num_envs, 1}));

        MUSTRATEGO_CHECK(scratch.all().item<bool>(), "INTERNAL BUG: Random action sampler sampled an illegal action");
    }
#endif
}

void StrategoRolloutBuffer::SampleFirstLegalAction(torch::Tensor actions_out)
{
    MUSTRATEGO_CHECK_CUDA_DTYPE(actions_out, conf.cuda_device, torch::kInt32,
                                "Action tensor must be CUDA-allocated on device %d and of dtype `torch.int32`",
                                conf.cuda_device);
    if (actions_out.sizes() != torch::IntArrayRef({num_envs}))
        MUSTRATEGO_FATAL("Output action tensor has wrong shape (must be: (%d,))", num_envs);

    private_action_mask_.zero_();
    ComputeLegalActionMask_(current_step_, private_action_mask_.data_ptr<bool>());

    auto iter =
        at::meta::make_reduction(private_action_mask_, actions_out, /*dims=*/{1}, /*keepdim=*/false, torch::kBool);
    at::native::gpu_reduce_kernel<bool, int32_t>(
        iter,
        at::native::ArgMaxOps<bool>{},
        thrust::pair<bool, int32_t>(false, 0));

#ifdef DEBUG
    {
        torch::Tensor scratch =
            MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kBool, {num_envs, 1});
        at::gather_out(scratch, private_action_mask_, /* dim */ 1, actions_out.to(torch::kInt64).view({num_envs, 1}));

        MUSTRATEGO_CHECK(scratch.all().item<bool>(), "INTERNAL BUG: First legal action sampler sampled an illegal action");
    }
#endif
}

void StrategoRolloutBuffer::ComputeLegalActionMask(const uint64_t step)
{
    GUARD_STEP_ID

    legal_action_mask.zero_();
    bool *d_out = legal_action_mask.data_ptr<bool>();
    ComputeLegalActionMask_(step, d_out);
}

void StrategoRolloutBuffer::ComputeLegalActionMask_(const uint64_t step, bool *d_out, const bool handle_terminated)
{
    GUARD_STEP_ID

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));

    // Each thread handles a cell in the board. There are 100 * num_env cells.
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);

    const uint32_t to_play = to_play_[step % buf_size];
    const uint8_t *terminated_since = d_terminated_since_ + (step % buf_size) * num_envs;
    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;
    const TwosquareState *twosquare_state_red = d_twosquare_state_red_ + (step % buf_size) * num_envs;
    const TwosquareState *twosquare_state_blue = d_twosquare_state_blue_ + (step % buf_size) * num_envs;
    const int32_t *illegal_chase_actions = d_illegal_chase_actions_ + (step % buf_size) * num_envs * MAX_CHASE_LENGTH;

    LegalActionsMaskKernel<<<num_blocks, num_threads>>>(
        d_out,
        terminated_since,
        num_envs,
        to_play,
        boards,
        handle_terminated);

    if (conf.two_square_rule)
    {
        RemoveTwosquareActions(d_out,
                               to_play == 1 ? twosquare_state_red : twosquare_state_blue,
                               terminated_since,
                               num_envs);
    }
    if (conf.continuous_chasing_rule)
    {
        RemoveIllegalChaseMoves(d_out, illegal_chase_actions, terminated_since, num_envs, conf.cuda_device);
    }

#ifdef DEBUG
    if (handle_terminated)
    {
        const torch::Tensor action_mask = MUSTRATEGO_WRAP_CUDA_TENSOR(
            d_out,
            conf.cuda_device,
            torch::kBool,
            {num_envs, NUM_ACTIONS});
        MUSTRATEGO_CHECK(action_mask.any(/* dim */ {1}).all().item<bool>(),
                         "INTERNAL BUG: The legal action mask is empty");
    }
#endif
}
void StrategoRolloutBuffer::UpdateHasLegalMovement_(const uint64_t step)
{
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));

    const uint32_t to_play = to_play_[step % buf_size];
    uint8_t *has_legal_movement = d_has_legal_movement_ + (step % buf_size) * num_envs;
    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;
    const TwosquareState *twosquare_state_red = d_twosquare_state_red_ + (step % buf_size) * num_envs;
    const TwosquareState *twosquare_state_blue = d_twosquare_state_blue_ + (step % buf_size) * num_envs;

    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);

    torch::Tensor has_legal_movement_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(has_legal_movement, conf.cuda_device, torch::kUInt8, {num_envs});
    torch::Tensor not_terminated = MUSTRATEGO_WRAP_CUDA_TENSOR(d_terminated_since_ + (step % buf_size) * num_envs, conf.cuda_device, torch::kUInt8, {num_envs}).eq(0);
    torch::Tensor red_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_ + NUM_ACTIONS * num_envs, conf.cuda_device, torch::kUInt8, {num_envs});
    torch::Tensor blue_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_ + (NUM_ACTIONS + 1) * num_envs, conf.cuda_device, torch::kUInt8, {num_envs});
    torch::Tensor twosquare_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs});

    // Compute whether the red player has any legal move
    if (conf.continuous_chasing_rule && conf.two_square_rule && to_play == 1)
    {
        // The interaction between two-square rule and continuous chasing rule is a bit tricky.
        //
        // In particular, it is possible that they interact to preclude all moves. But more fundamentally, it is possible that
        // a move of length one is precluded by the continuous chasing rule, but not one of length two. So, looking at saturated
        // precluded directions is not enough.
        torch::Tensor scratch_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs, NUM_ACTIONS});
        scratch_tensor.zero_();
        ComputeLegalActionMask_(step, (bool *)d_scratch_, false);
        torch::any_out(
            red_tensor,
            scratch_tensor,
            /* dim */ 1,
            /* keepdim */ false);
    }
    else
    {
        torch::Tensor scratch_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs, 100});
        SaturatedNumMovementDirectionsKernel<<<num_blocks, num_threads>>>(
            (uint8_t *)d_scratch_,
            boards,
            num_envs,
            1 /* red */);
        torch::sum_out(
            red_tensor,
            scratch_tensor,
            /* dim */ 1,
            /* keepdim */ false);
        if (conf.two_square_rule && to_play == 1)
        {
            // Off-turn player cannot lack legal movement on account of the two-square rule.
            IsTwosquareRulePrecludingDirection(
                (bool *)twosquare_tensor.data_ptr<uint8_t>(),
                twosquare_state_red,
                num_envs);

            assert((twosquare_tensor <= red_tensor).all().item<bool>());
            red_tensor.sub_(twosquare_tensor);
        }
        red_tensor.clamp_max_(1);
    }
    if (conf.continuous_chasing_rule && conf.two_square_rule && to_play == 2)
    {
        torch::Tensor scratch_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs, NUM_ACTIONS});
        scratch_tensor.zero_();
        ComputeLegalActionMask_(step, (bool *)d_scratch_, false);
        torch::any_out(
            blue_tensor,
            scratch_tensor,
            /* dim */ 1,
            /* keepdim */ false);
    }
    else
    {
        torch::Tensor scratch_tensor = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs, 100});
        SaturatedNumMovementDirectionsKernel<<<num_blocks, num_threads>>>(
            (uint8_t *)d_scratch_,
            boards,
            num_envs,
            2 /* blue */);
        torch::sum_out(
            blue_tensor,
            scratch_tensor,
            /* dim */ 1,
            /* keepdim */ false);
        if (conf.two_square_rule && to_play == 2)
        {
            IsTwosquareRulePrecludingDirection(
                (bool *)twosquare_tensor.data_ptr<uint8_t>(),
                twosquare_state_blue,
                num_envs);
            assert((twosquare_tensor <= blue_tensor).all().item<bool>());
            blue_tensor.sub_(twosquare_tensor);
        }
        blue_tensor.clamp_max_(1);
    }

    red_tensor.add_(blue_tensor, /* alpha */ 2);
    has_legal_movement_tensor.index_put_({not_terminated}, red_tensor.index({not_terminated}));
}

void StrategoRolloutBuffer::ComputeMoveSummaryHistoryTensor(const uint64_t step)
{
    MUSTRATEGO_FATAL("ComputeMoveSummaryHistoryTensor is deprecated.");

    GUARD_STEP_ID
    GUARD_MEMORY
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));

    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(conf.move_memory * num_envs, num_threads);

    move_summary_history_tensor.fill_(100);
    SnapshotMoveSummaryHistoryKernel<<<num_threads, num_blocks>>>(
        move_summary_history_tensor.data_ptr<uint8_t>(),
        d_move_summary_history_,
        /* prehistory */ conf.reset_state ? conf.reset_state->move_summary_history.data_ptr<uint8_t>() : nullptr,
        d_num_moves_,
        d_num_moves_since_reset_,
        step,
        to_play_[step % buf_size],
        /* relativize */ true,
        conf.move_memory,
        buf_size,
        num_envs);
}

void StrategoRolloutBuffer::ComputeBoardStateTensor_(const uint64_t step, MUSTRATEGO_FLOAT_CUDA_DTYPE *out)
{
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    const uint32_t INFOSTATE_STRIDE = NUM_INFOSTATE_CHANNELS * 100;

    const uint32_t to_play = to_play_[step % buf_size];
    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;
    const StrategoBoard *zero_boards = d_zero_boards_ + (step % buf_size) * num_envs;
    const int32_t *num_moves = d_num_moves_ + (step % buf_size) * num_envs;
    const int32_t *num_moves_since_last_attack = d_num_moves_since_last_attack_ + (step % buf_size) * num_envs;

    {
        // Each thread handles a cell in the board. There are 100 * num_env cells.
        const uint32_t num_threads = 1024;
        const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);

        // Channels 0 .. 11
        BoardStateKernel__OwnPieceTypes<<<num_blocks, num_threads>>>(out, to_play, num_envs, boards, INFOSTATE_STRIDE);

        // Channels 12 .. 23
        BoardStateKernel__ProbTypes<<<num_blocks, num_threads>>>(out, to_play, /* rotate */ false, num_envs, boards, INFOSTATE_STRIDE, /* CHANNEL_SHIFT */ 1200);

        // Channels 24 .. 35
        BoardStateKernel__ProbTypes<<<num_blocks, num_threads>>>(out, 3 - to_play, /* rotate */ true, num_envs, boards, INFOSTATE_STRIDE, /* CHANNEL_SHIFT */ 2400);

        // Fill channels 36 (own hidden), 37 (opponent hidden), 38 (empty pieces), 39 (own moved), 40 (opponent moved), 41 (proportion complete)
        BoardStateKernel__InvisiblesEmptyAndMoved<<<num_blocks, num_threads>>>(out,
                                                                               to_play,
                                                                               num_envs,
                                                                               num_moves,
                                                                               num_moves_since_last_attack,
                                                                               conf.max_num_moves,
                                                                               conf.max_num_moves_between_attacks,
                                                                               boards,
                                                                               INFOSTATE_STRIDE);

        BoardStateKernel__ThreatEvadeActiveAdj<<<num_blocks, num_threads>>>(out,
                                                                            to_play,
                                                                            num_envs,
                                                                            boards,
                                                                            INFOSTATE_STRIDE);

        BoardStateKernel__Deaths<<<num_blocks, num_threads>>>(out,
                                                              to_play,
                                                              num_envs,
                                                              boards,
                                                              zero_boards,
                                                              INFOSTATE_STRIDE);

        BoardStateKernel__Protections<<<num_blocks, num_threads>>>(out,
                                                                   to_play,
                                                                   num_envs,
                                                                   boards,
                                                                   INFOSTATE_STRIDE);
    }
    {
        const uint32_t num_threads = 128;
        const uint32_t num_blocks = ceil(60ll * num_envs, num_threads);

        BoardStateKernel__DeathReasons<<<num_blocks, num_threads>>>(out,
                                                                    to_play,
                                                                    num_envs,
                                                                    boards,
                                                                    INFOSTATE_STRIDE);
    }
}

void StrategoRolloutBuffer::ComputeInfostateTensor(const uint64_t step)
{
    GUARD_STEP_ID
    GUARD_MEMORY
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));

    infostate_tensor.zero_();
    MUSTRATEGO_FLOAT_CUDA_DTYPE *infostate_ptr = infostate_tensor.data_ptr<MUSTRATEGO_FLOAT_CUDA_DTYPE>();
    ComputeBoardStateTensor_(step, infostate_ptr);

    infostate_ptr += NUM_BOARD_STATE_CHANNELS * 100;

    // At this point, the only remaining thing to do is to encode the history for the past
    // min{conf.move_memory, row_id} rows.
    if (conf.move_memory)
    {
        StrategoBoard *board_prehistory = nullptr;
        if (conf.reset_state)
        {
            board_prehistory = ((StrategoBoard *)(conf.reset_state->board_history.data_ptr<uint8_t>()));
            assert(conf.move_memory <= conf.reset_state->board_history.size(0));
            board_prehistory += (conf.reset_state->board_history.size(0) - conf.move_memory) * num_envs;
        }
        if (conf.enable_src_dst_planes)
        {
            const uint32_t num_threads = 1024;
            const uint32_t num_blocks = ceil(conf.move_memory * num_envs, num_threads);

            InjectInfostateSrcDstKernel<<<num_blocks, num_threads>>>(
                infostate_ptr,
                d_action_history_,
                /* prehistory */ conf.reset_state ? conf.reset_state->action_history.data_ptr<int32_t>() : nullptr,
                d_terminated_since_,
                d_num_moves_,
                d_num_moves_since_reset_,
                step,
                conf.move_memory,
                buf_size,
                num_envs,
                /* INFOSTATE_STRIDE */ NUM_INFOSTATE_CHANNELS * 100);

            infostate_ptr += conf.move_memory * 100;
        }
        if (conf.enable_hidden_and_types_planes)
        {
            const uint32_t num_threads = 1024;
            const uint32_t num_blocks = ceil(conf.move_memory * num_envs * 100, num_threads);

            // WE USE -1 TO MEAN INAPPLICABLE:
            // our_types is -1 if the piece does not belong to us
            // their_visible_types is -1 if the piece is not visible or does not belong to them.
            InjectInfostateHiddenAndTypesKernel<<<num_blocks, num_threads>>>(
                infostate_ptr,
                /* board history */ d_boards_,
                /* board prehistory */ board_prehistory,
                d_terminated_since_,
                d_num_moves_,
                d_num_moves_since_reset_,
                step,
                to_play_[step % buf_size],
                conf.move_memory,
                buf_size,
                num_envs,
                /* INFOSTATE_STRIDE */ NUM_INFOSTATE_CHANNELS * 100);

            infostate_ptr += conf.move_memory * 400;
        }
        if (conf.enable_dm_planes)
        {
            const uint32_t num_threads = 1024;
            const uint32_t num_blocks = ceil(conf.move_memory * num_envs, num_threads);

            InjectInfostateDmKernel<<<num_blocks, num_threads>>>(
                infostate_ptr,
                /* action history */ d_action_history_,
                /* action prehistory */ conf.reset_state ? conf.reset_state->action_history.data_ptr<int32_t>() : nullptr,
                /* board history */ d_boards_,
                /* board prehistory */ board_prehistory,
                d_terminated_since_,
                d_num_moves_,
                d_num_moves_since_reset_,
                step,
                to_play_[step % buf_size],
                conf.move_memory,
                buf_size,
                num_envs,
                /* INFOSTATE_STRIDE */ NUM_INFOSTATE_CHANNELS * 100);

            infostate_ptr += conf.move_memory * 100;
        }
    }
}

void StrategoRolloutBuffer::ComputeRewardPl0(const uint64_t step)
{
    GUARD_STEP_ID

    MUSTRATEGO_FLOAT_CUDA_DTYPE *d_out = reward_pl0.data_ptr<MUSTRATEGO_FLOAT_CUDA_DTYPE>();
    const uint8_t *terminated_since = d_terminated_since_ + (step % buf_size) * num_envs;
    const int32_t *num_moves = d_num_moves_ + (step % buf_size) * num_envs;
    const int32_t *num_moves_since_last_attack = d_num_moves_since_last_attack_ + (step % buf_size) * num_envs;
    const uint8_t *flag_captured = d_flag_captured_ + (step % buf_size) * num_envs;
    const uint8_t *has_legal_movement = d_has_legal_movement_ + (step % buf_size) * num_envs;
    const uint32_t to_play = to_play_[step % buf_size];

    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(num_envs, num_threads);

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    ComputeRewardPl0Kernel<<<num_envs, num_threads>>>(
        d_out,
        terminated_since,
        num_moves,
        num_moves_since_last_attack,
        flag_captured,
        has_legal_movement,
        num_envs,
        conf.max_num_moves,
        conf.max_num_moves_between_attacks,
        to_play);
}

void StrategoRolloutBuffer::ComputeIsUnknownPiece_(const uint64_t step, bool *d_out)
{
    GUARD_STEP_ID

    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;
    const uint32_t to_play = to_play_[step % buf_size];

    // Each thread handles a cell in the board. There are 100 * num_env cells.
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    ComputeIsUnknownPieceKernel<<<num_blocks, num_threads>>>(
        d_out,
        boards,
        num_envs,
        to_play);
}

void StrategoRolloutBuffer::ComputeIsUnknownPiece(const uint64_t step)
{
    GUARD_STEP_ID

    bool *d_out = is_unknown_piece.data_ptr<bool>();
    ComputeIsUnknownPiece_(step, d_out);
}

void StrategoRolloutBuffer::ComputePieceTypeOnehot(const uint64_t step)
{
    GUARD_STEP_ID

    piece_type_onehot.zero_();
    bool *d_out = piece_type_onehot.data_ptr<bool>();
    const uint32_t to_play = to_play_[step % buf_size];
    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;

    // Each thread handles a cell in the board. There are 100 * num_env cells.
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    ComputePieceTypeOnehotKernel<<<num_blocks, num_threads>>>(
        d_out,
        boards,
        num_envs,
        to_play);
}

void StrategoRolloutBuffer::ComputeTwoSquareRuleApplies(const uint64_t step)
{
    GUARD_STEP_ID

    two_square_rule_applies.zero_();
    bool *d_out = two_square_rule_applies.data_ptr<bool>();

    const uint32_t to_play = to_play_[step % buf_size];
    const TwosquareState *twosquare_state_red = d_twosquare_state_red_ + (step % buf_size) * num_envs;
    const TwosquareState *twosquare_state_blue = d_twosquare_state_blue_ + (step % buf_size) * num_envs;

    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    IsTwosquareRuleTriggered(
        d_out,
        (to_play == 1) ? twosquare_state_red : twosquare_state_blue,
        num_envs);
}

void StrategoRolloutBuffer::ComputeUnknownPieceTypeOnehot(const uint64_t step, const uint32_t max_k)
{
    GUARD_STEP_ID
    MUSTRATEGO_CHECK(max_k >= 1 && max_k <= 40, "max_k parameter out of bound (expected in range [1, 40], found: %d)", max_k);

    unknown_piece_type_onehot = MUSTRATEGO_WRAP_CUDA_TENSOR(d_unknown_piece_type_onehot_,
                                                            conf.cuda_device, torch::kBool,
                                                            {num_envs, max_k, NUM_PIECE_TYPES});
    unknown_piece_type_onehot.zero_();
    bool *d_out = unknown_piece_type_onehot.data_ptr<bool>();
    const uint32_t to_play = to_play_[step % buf_size];
    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;

    // First, we call `ComputeIsUnknownPiece`
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    ComputeIsUnknownPiece_(step, (bool *)d_scratch_);

    torch::Tensor scratch = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs, 100});
    scratch.cumsum_(/* dim */ 1);

    // Each thread handles a cell in the board. There are 100 * num_env cells.
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);
    ComputeUnknownPieceTypeOnehotKernel<<<num_blocks, num_threads>>>(
        d_out,
        d_scratch_,
        boards,
        num_envs,
        max_k,
        to_play);
}

void StrategoRolloutBuffer::ComputeUnknownPieceHasMoved(const uint64_t step, const uint32_t max_k)
{
    GUARD_STEP_ID
    MUSTRATEGO_CHECK(max_k >= 1 && max_k <= 40, "max_k parameter out of bound (expected in range [1, 40], found: %d)", max_k);

    unknown_piece_has_moved = MUSTRATEGO_WRAP_CUDA_TENSOR(d_unknown_piece_has_moved_,
                                                          conf.cuda_device, torch::kBool, {num_envs, max_k});
    unknown_piece_has_moved.zero_();
    bool *d_out = unknown_piece_has_moved.data_ptr<bool>();
    const uint32_t to_play = to_play_[step % buf_size];
    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;

    // First, we call `ComputeIsUnknownPiece`
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    ComputeIsUnknownPiece_(step, (bool *)d_scratch_);

    torch::Tensor scratch = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs, 100});
    scratch.cumsum_(/* dim */ 1);

    // Each thread handles a cell in the board. There are 100 * num_env cells.
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);
    ComputeUnknownPieceHasMovedKernel<<<num_blocks, num_threads>>>(
        d_out,
        d_scratch_,
        boards,
        num_envs,
        max_k,
        to_play);
}

void StrategoRolloutBuffer::ComputeUnknownPiecePositionOnehot(const uint64_t step, const uint32_t max_k)
{
    GUARD_STEP_ID
    MUSTRATEGO_CHECK(max_k >= 1 && max_k <= 40, "max_k parameter out of bound (expected in range [1, 40], found: %d)", max_k);

    unknown_piece_position_onehot = MUSTRATEGO_WRAP_CUDA_TENSOR(d_unknown_piece_position_onehot_,
                                                                conf.cuda_device, torch::kBool, {num_envs, max_k, 100});
    unknown_piece_position_onehot.zero_();
    bool *d_out = unknown_piece_position_onehot.data_ptr<bool>();
    const uint32_t to_play = to_play_[step % buf_size];
    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;

    // First, we call `ComputeIsUnknownPiece`
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));
    ComputeIsUnknownPiece_(step, (bool *)d_scratch_);

    torch::Tensor scratch = MUSTRATEGO_WRAP_CUDA_TENSOR(d_scratch_, conf.cuda_device, torch::kUInt8, {num_envs, 100});
    scratch.cumsum_(/* dim */ 1);

    // Each thread handles a cell in the board. There are 100 * num_env cells.
    const uint32_t num_threads = 1024;
    const uint32_t num_blocks = ceil(100ll * num_envs, num_threads);
    ComputeUnknownPiecePositionOnehotKernel<<<num_blocks, num_threads>>>(
        d_out,
        d_scratch_,
        num_envs,
        max_k,
        to_play);
}

torch::Tensor StrategoRolloutBuffer::GetTerminatedSince(const uint64_t step) const
{
    GUARD_STEP_ID

    uint8_t *terminated_since = d_terminated_since_ + (step % buf_size) * num_envs;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(terminated_since, conf.cuda_device, torch::kUInt8, {num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetHasLegalMovement(const uint64_t step) const
{
    GUARD_STEP_ID

    uint8_t *has_legal_movement = d_has_legal_movement_ + (step % buf_size) * num_envs;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(has_legal_movement, conf.cuda_device, torch::kUInt8, {num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetFlagCaptured(const uint64_t step) const
{
    GUARD_STEP_ID

    uint8_t *flag_captured = d_flag_captured_ + (step % buf_size) * num_envs;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(flag_captured, conf.cuda_device, torch::kUInt8, {num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetNumMoves(const uint64_t step) const
{
    GUARD_STEP_ID

    int32_t *num_moves = d_num_moves_ + (step % buf_size) * num_envs;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(num_moves, conf.cuda_device, torch::kInt32, {num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetNumMovesSinceLastAttack(const uint64_t step) const
{
    GUARD_STEP_ID

    int32_t *num_moves_since_last_attack = d_num_moves_since_last_attack_ + (step % buf_size) * num_envs;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(num_moves_since_last_attack,
                                       conf.cuda_device, torch::kInt32, {num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetNumMovesSinceReset(const uint64_t step) const
{
    GUARD_STEP_ID

    int32_t *num_moves_since_reset = d_num_moves_since_reset_ + (step % buf_size) * num_envs;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(num_moves_since_reset,
                                       conf.cuda_device, torch::kInt32, {num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetNumMovesSinceResetTensor() const
{
    return MUSTRATEGO_WRAP_CUDA_TENSOR(d_num_moves_since_reset_, conf.cuda_device, torch::kInt32, {buf_size, num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetActionHistoryTensor() const
{
    return MUSTRATEGO_WRAP_CUDA_TENSOR(d_action_history_, conf.cuda_device, torch::kInt32, {buf_size, num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetPlayedActions(const uint64_t step) const
{
    GUARD_STEP_ID
    MUSTRATEGO_CHECK(step < current_step_,
                     "Cannot get played actions for current step %zd (actions have not been selected yet)", step);

    const uint32_t row_id = step % buf_size;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(d_action_history_ + row_id * num_envs,
                                       conf.cuda_device, torch::kInt32, {num_envs});
}

torch::Tensor StrategoRolloutBuffer::GetMoveSummary(const uint64_t step) const
{
    GUARD_STEP_ID
    MUSTRATEGO_CHECK(step < current_step_,
                     "Cannot get played actions for current step %zd (actions have not been selected yet)", step);

    const uint32_t row_id = step % buf_size;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(d_move_summary_history_ + row_id * num_envs * 6,
                                       conf.cuda_device, torch::kUInt8, {num_envs, 6});
}

torch::Tensor StrategoRolloutBuffer::GetBoardTensor(const uint64_t step) const
{
    GUARD_STEP_ID

    const uint32_t row_id = step % buf_size;
    const StrategoBoard *boards = d_boards_ + row_id * num_envs;
    MUSTRATEGO_CHECK((size_t)boards % 128 == 0,
                     "INTERNAL BUG: Unexpected alignment for `board` data pointer");
    return MUSTRATEGO_WRAP_CUDA_TENSOR((uint8_t *)boards,
                                       conf.cuda_device, torch::kUInt8, {num_envs, sizeof(StrategoBoard)});
}

torch::Tensor StrategoRolloutBuffer::GetBoardTensor() const
{
    return MUSTRATEGO_WRAP_CUDA_TENSOR((uint8_t *)d_boards_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs, sizeof(StrategoBoard)});
}

torch::Tensor StrategoRolloutBuffer::GetZeroBoardTensor(const uint64_t step) const
{
    GUARD_STEP_ID

    const uint32_t row_id = step % buf_size;
    const StrategoBoard *zero_boards = d_zero_boards_ + row_id * num_envs;
    MUSTRATEGO_CHECK((size_t)zero_boards % 128 == 0,
                     "INTERNAL BUG: Unexpected alignment for `zero_board` data pointer");
    return MUSTRATEGO_WRAP_CUDA_TENSOR((uint8_t *)zero_boards,
                                       conf.cuda_device, torch::kUInt8, {num_envs, sizeof(StrategoBoard)});
}

std::array<torch::Tensor, 2> StrategoRolloutBuffer::GetTwosquareState(const uint64_t step) const
{
    GUARD_STEP_ID

    const TwosquareState *twosquare_state_red = d_twosquare_state_red_ + (step % buf_size) * num_envs;
    const TwosquareState *twosquare_state_blue = d_twosquare_state_blue_ + (step % buf_size) * num_envs;

    return {TwosquareStateAsTensor(twosquare_state_red, num_envs, conf.cuda_device),
            TwosquareStateAsTensor(twosquare_state_blue, num_envs, conf.cuda_device)};
}

std::pair<EnvState, EnvState> StrategoRolloutBuffer::SnapshotEnvHistory(const uint64_t step, const int64_t env_idx) const
{
    MUSTRATEGO_CHECK(conf.reset_behavior != CUSTOM_ENV_STATE, "Snapshotting env history is not allowed with CUSTOM_ENV_STATE");
    GUARD_STEP_ID
    GUARD_MEMORY

    MUSTRATEGO_CHECK(env_idx < num_envs, "env_idx out of bounds (expected in range [0, %d), found: %zd)", num_envs, env_idx);
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));

    int32_t num_moves;
    cudaMemcpy(&num_moves, d_num_moves_ + (step % buf_size) * num_envs + env_idx, sizeof(int32_t), cudaMemcpyDeviceToHost);

    // FIXME: Can this be relaxed into a >= ?
    MUSTRATEGO_CHECK(step + buf_size > num_moves + current_step_,
                     "The env's num_moves (%d) is too high given the current buf_size (%d) starting at the request step (%zd). The current step is %zd. You need to increase the buf size",
                     num_moves, buf_size, step, current_step_);
    assert(num_moves < buf_size); // This is implied by the check above.
    const int64_t to_row_id = step % buf_size;
    assert(step >= num_moves);
    const int64_t from_row_id = (step - num_moves) % buf_size;

    if (!conf.quiet)
    {
        MUSTRATEGO_DEBUG("Snapshotting env history for env_idx %d with num_moves %d...", env_idx, num_moves);
    }
    EnvState even_states;
    EnvState odd_states;

    even_states.num_envs = (num_moves / 2) + 1;
    // Since snapshotting from reset states is not allowed, this is safe. Let's double
    // check however, to catch possible future regressions
    assert(1 + (step % 2) == to_play_.at(to_row_id));
    even_states.to_play = 0;
    odd_states.num_envs = (num_moves + 1) / 2;
    odd_states.to_play = 1;

    const uint32_t buf_size = this->buf_size;
    const auto snip_tensor = [from_row_id, to_row_id, buf_size, env_idx, num_moves, &even_states, &odd_states](torch::Tensor in, torch::Tensor &even_out, torch::Tensor &odd_out)
    {
        assert(in.dim() >= 2);
        assert(in.size(0) == buf_size);
        // Since the buffer is circular, we need to adjust the tensor
        if (from_row_id > to_row_id)
        {
            in = torch::cat({in.index({Slice(from_row_id, None), env_idx}),
                             in.index({Slice(None, to_row_id + 1), env_idx})},
                            /* dim */ 0);
        }
        else
        {
            in = in.index({Slice(from_row_id, to_row_id + 1), env_idx});
        }
        // The newest board is now last.
        even_out = in.index({Slice(0, None, 2)}).contiguous();
        odd_out = in.index({Slice(1, None, 2)}).contiguous();

        assert(even_out.size(0) == even_states.num_envs);
        assert(odd_out.size(0) == odd_states.num_envs);
    };
    const auto snip_tensor_tri = [from_row_id, to_row_id, buf_size, env_idx, num_moves, &even_states, &odd_states](torch::Tensor in, torch::Tensor &even_out, torch::Tensor &odd_out, const uint32_t move_memory)
    {
        assert(in.dim() >= 2);
        assert(in.size(0) == buf_size);
        std::vector<int64_t> even_shape(in.sizes().begin(), in.sizes().end());
        assert(even_shape.size() >= 2);
        even_shape[0] = move_memory;
        even_shape[1] = even_states.num_envs;
        std::vector<int64_t> odd_shape(in.sizes().begin(), in.sizes().end());
        assert(odd_shape.size() >= 2);
        odd_shape[0] = move_memory;
        odd_shape[1] = odd_states.num_envs;
        const auto options = in.options();

        if (num_moves > 0 && move_memory > 0)
        {
            // Since the buffer is circular, we need to adjust the tensor
            if (from_row_id > to_row_id)
            {
                in = torch::cat({in.index({Slice(from_row_id, None), env_idx}),
                                 in.index({Slice(None, to_row_id), env_idx})},
                                /* dim */ 0);
            }
            else
            {
                in = in.index({Slice(from_row_id, to_row_id), env_idx});
            }

            // At this stage, we have information about times
            // 0 1 ... (n-1)   -> n times

            // The history tensor need to contain
            // . 0 1 ... (n-1) -> n + 1 times
            // . . 0 ... (n-2)
            // . . . ... (n-3)
            // ...
            // . . . ... n - move memory
            in = torch::cat({in.index({Slice(0, 1)}), in, in.index({Slice(0, 1)})}, /* dim */ 0);
            in.unsqueeze_(0);
            assert(in.size(0) == 1 && in.size(1) == num_moves + 2);
            if (in.dim() == 2)
            {
                // After the expand, we have a 2D tensor of shape (move_memory, num_moves + 2)
                // We want to reshape it into something of the form (move_memory, num_moves + 1)
                in = in.tile({move_memory, 1}).flatten();
                in.resize_({in.numel() - move_memory});
                in = in.view({-1, num_moves + 1});
            }
            else
            {
                assert(in.dim() == 3 && in.size(0) == 1);

                const auto payload = in.size(2);
                in = in.tile({move_memory, 1, 1}).flatten();
                in.resize_({in.numel() - move_memory * payload});
                in = in.view({-1, num_moves + 1, payload});
            }

            if (in.dim() == 2)
            {
                in.triu_(/* diagonal */ 1);
            }
            else
            {
                in = in.permute({2, 0, 1}).triu_(/* diagonal */ 1).permute({1, 2, 0});
            }
            in = in.flip(0);
        }
        else
        {
            if (in.dim() == 2)
            {
                in = torch::zeros({move_memory, num_moves + 1}, options);
            }
            else
            {
                assert(in.dim() == 3);
                in = torch::zeros({move_memory, num_moves + 1, in.size(2)}, options);
            }
        }
        assert(in.size(0) == move_memory);

        even_out = in.index({Slice(), Slice(0, None, 2)}).contiguous();
        odd_out = in.index({Slice(), Slice(1, None, 2)}).contiguous();

        assert(even_out.size(0) == move_memory);
        assert(odd_out.size(0) == move_memory);
        assert(even_out.size(1) == even_states.num_envs);
        assert(odd_out.size(1) == odd_states.num_envs);
    };

    snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_boards_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs, sizeof(StrategoBoard)}),
                /* */ even_states.boards,
                /* */ odd_states.boards);
    snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_zero_boards_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs, sizeof(StrategoBoard)}),
                /* */ even_states.zero_boards,
                /* */ odd_states.zero_boards);
    snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_num_moves_, conf.cuda_device, torch::kInt32, {buf_size, num_envs}),
                /* */ even_states.num_moves,
                /* */ odd_states.num_moves);
    snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_num_moves_since_last_attack_, conf.cuda_device, torch::kInt32, {buf_size, num_envs}),
                /* */ even_states.num_moves_since_last_attack,
                /* */ odd_states.num_moves_since_last_attack);
    snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_terminated_since_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs}),
                /* */ even_states.terminated_since,
                /* */ odd_states.terminated_since);
    snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_has_legal_movement_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs}),
                /* */ even_states.has_legal_movement,
                /* */ odd_states.has_legal_movement);
    snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_flag_captured_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs}),
                /* */ even_states.flag_captured,
                /* */ odd_states.flag_captured);

    assert(conf.move_memory > 0);
    snip_tensor_tri(MUSTRATEGO_WRAP_CUDA_TENSOR(d_boards_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs, sizeof(StrategoBoard)}),
                    /* */ even_states.board_history,
                    /* */ odd_states.board_history,
                    /* move_memory */ conf.continuous_chasing_rule ? std::max(conf.move_memory, (uint32_t)MAX_CHASE_LENGTH) : conf.move_memory);
    snip_tensor_tri(MUSTRATEGO_WRAP_CUDA_TENSOR(d_action_history_, conf.cuda_device, torch::kInt32, {buf_size, num_envs}),
                    /* */ even_states.action_history,
                    /* */ odd_states.action_history,
                    /* move_memory */ conf.move_memory);
    snip_tensor_tri(MUSTRATEGO_WRAP_CUDA_TENSOR(d_move_summary_history_, conf.cuda_device, torch::kUInt8, {buf_size, num_envs, 6}),
                    /* */ even_states.move_summary_history,
                    /* */ odd_states.move_summary_history,
                    /* move_memory */ conf.move_memory);

    if (conf.continuous_chasing_rule)
    {
        even_states.chase_state.emplace();
        odd_states.chase_state.emplace();

        snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_dst_pos[0], conf.cuda_device, torch::kUInt8, {buf_size, num_envs}),
                    /* */ even_states.chase_state->last_dst_pos[0],
                    /* */ odd_states.chase_state->last_dst_pos[0]);
        snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_dst_pos[1], conf.cuda_device, torch::kUInt8, {buf_size, num_envs}),
                    /* */ even_states.chase_state->last_dst_pos[1],
                    /* */ odd_states.chase_state->last_dst_pos[1]);
        snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_src_pos[0], conf.cuda_device, torch::kUInt8, {buf_size, num_envs}),
                    /* */ even_states.chase_state->last_src_pos[0],
                    /* */ odd_states.chase_state->last_src_pos[0]);
        snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_src_pos[1], conf.cuda_device, torch::kUInt8, {buf_size, num_envs}),
                    /* */ even_states.chase_state->last_src_pos[1],
                    /* */ odd_states.chase_state->last_src_pos[1]);
        snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.chase_length[0], conf.cuda_device, torch::kInt32, {buf_size, num_envs}),
                    /* */ even_states.chase_state->chase_length[0],
                    /* */ odd_states.chase_state->chase_length[0]);
        snip_tensor(MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.chase_length[1], conf.cuda_device, torch::kInt32, {buf_size, num_envs}),
                    /* */ even_states.chase_state->chase_length[1],
                    /* */ odd_states.chase_state->chase_length[1]);
    }

    return {even_states, odd_states};
}

EnvState
StrategoRolloutBuffer::SnapshotState(const uint64_t step) const
{
    GUARD_STEP_ID
    GUARD_MEMORY
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(conf.cuda_device));

    const uint32_t adjusted_memory = conf.continuous_chasing_rule ? std::max(conf.move_memory, (uint32_t)MAX_CHASE_LENGTH) : conf.move_memory;

    torch::Tensor action_history;
    torch::Tensor board_history;
    torch::Tensor move_summary_history;
    MUSTRATEGO_CREATE_CUDA_TENSOR(action_history, conf.cuda_device, torch::kInt32, {conf.move_memory, num_envs});
    MUSTRATEGO_CREATE_CUDA_TENSOR(board_history, conf.cuda_device, torch::kUInt8, {adjusted_memory, num_envs, sizeof(StrategoBoard)});
    MUSTRATEGO_CREATE_CUDA_TENSOR(move_summary_history, conf.cuda_device, torch::kUInt8, {conf.move_memory, num_envs, 6});

    action_history.zero_();
    board_history.zero_();
    move_summary_history.zero_();

    if (conf.move_memory)
    {

        const uint32_t num_threads = 2056;
        const uint32_t num_blocks = ceil(conf.move_memory * num_envs, num_threads);

        SnapshotActionHistoryKernel<<<num_threads, num_blocks>>>(
            action_history.data_ptr<int32_t>(),
            d_action_history_,
            /* prehistory */ conf.reset_state ? conf.reset_state->action_history.data_ptr<int32_t>() : nullptr,
            d_num_moves_,
            d_num_moves_since_reset_,
            step,
            conf.move_memory,
            buf_size,
            num_envs);
    }

    if (adjusted_memory)
    {
        MUSTRATEGO_CHECK(!conf.reset_state || (((size_t)conf.reset_state->board_history.data_ptr<uint8_t>()) % 128 == 0),
                         "Unexpected alignment of board prehistory pointer");
        MUSTRATEGO_CHECK((size_t)(board_history.data_ptr<uint8_t>()) % 128 == 0, "Tensor is not 128-byte aligned");
        MUSTRATEGO_CHECK(!conf.reset_state || (size_t)(conf.reset_state->board_history.data_ptr<uint8_t>()) % 128 == 0, "Tensor is not 128-byte aligned");
        MUSTRATEGO_CHECK(!conf.continuous_chasing_rule || buf_size >= MAX_CHASE_LENGTH, "INTERNAL BUG: Buffer size is too small for continuous chasing rule");

        const uint32_t num_threads = 2056;
        const uint32_t num_blocks = ceil(adjusted_memory * num_envs, num_threads);

        SnapshotBoardHistoryKernel<<<num_threads, num_blocks>>>(
            (StrategoBoard *)board_history.data_ptr<uint8_t>(),
            d_boards_,
            /* prehistory */ conf.reset_state ? (StrategoBoard *)conf.reset_state->board_history.data_ptr<uint8_t>() : nullptr,
            d_num_moves_,
            d_num_moves_since_reset_,
            step,
            adjusted_memory,
            buf_size,
            num_envs);
    }

    if (conf.move_memory)
    {

        const uint32_t num_threads = 2048;
        const uint32_t num_blocks = ceil(conf.move_memory * num_envs, num_threads);

        SnapshotMoveSummaryHistoryKernel<<<num_threads, num_blocks>>>(
            move_summary_history.data_ptr<uint8_t>(),
            d_move_summary_history_,
            /* prehistory */ conf.reset_state ? conf.reset_state->move_summary_history.data_ptr<uint8_t>() : nullptr,
            d_num_moves_,
            d_num_moves_since_reset_,
            step,
            to_play_[step % buf_size],
            /* relativize */ false,
            conf.move_memory,
            buf_size,
            num_envs);
    }

    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;
    const StrategoBoard *zero_boards = d_zero_boards_ + (step % buf_size) * num_envs;
    int32_t *num_moves = d_num_moves_ + (step % buf_size) * num_envs;
    int32_t *num_moves_since_last_attack = d_num_moves_since_last_attack_ + (step % buf_size) * num_envs;
    uint8_t *terminated_since = d_terminated_since_ + (step % buf_size) * num_envs;
    uint8_t *has_legal_movement = d_has_legal_movement_ + (step % buf_size) * num_envs;
    uint8_t *flag_captured = d_flag_captured_ + (step % buf_size) * num_envs;

    std::optional<EnvState::TensorChaseState> chase_state;
    if (conf.continuous_chasing_rule)
    {
        chase_state.emplace();
        chase_state->last_dst_pos[0] = MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_dst_pos[0] + (step % buf_size) * num_envs, conf.cuda_device, torch::kUInt8, {num_envs});
        chase_state->last_dst_pos[1] = MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_dst_pos[1] + (step % buf_size) * num_envs, conf.cuda_device, torch::kUInt8, {num_envs});
        chase_state->last_src_pos[0] = MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_src_pos[0] + (step % buf_size) * num_envs, conf.cuda_device, torch::kUInt8, {num_envs});
        chase_state->last_src_pos[1] = MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.last_src_pos[1] + (step % buf_size) * num_envs, conf.cuda_device, torch::kUInt8, {num_envs});
        chase_state->chase_length[0] = MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.chase_length[0] + (step % buf_size) * num_envs, conf.cuda_device, torch::kInt32, {num_envs});
        chase_state->chase_length[1] = MUSTRATEGO_WRAP_CUDA_TENSOR(d_chase_state_.chase_length[1] + (step % buf_size) * num_envs, conf.cuda_device, torch::kInt32, {num_envs});
    }

    return EnvState{
        .num_envs = num_envs,
        .to_play = to_play_[step % buf_size] - 1,
        .boards = MUSTRATEGO_WRAP_CUDA_TENSOR((uint8_t *)boards, conf.cuda_device, torch::kUInt8, {num_envs, sizeof(StrategoBoard)}).clone(),
        .zero_boards = MUSTRATEGO_WRAP_CUDA_TENSOR((uint8_t *)zero_boards, conf.cuda_device, torch::kUInt8, {num_envs, sizeof(StrategoBoard)}).clone(),
        .num_moves = MUSTRATEGO_WRAP_CUDA_TENSOR(num_moves, conf.cuda_device, torch::kInt32, {num_envs}).clone(),
        .num_moves_since_last_attack = MUSTRATEGO_WRAP_CUDA_TENSOR(num_moves_since_last_attack, conf.cuda_device, torch::kInt32, {num_envs}).clone(),
        .terminated_since = MUSTRATEGO_WRAP_CUDA_TENSOR(terminated_since, conf.cuda_device, torch::kUInt8, {num_envs}).clone(),
        .has_legal_movement = MUSTRATEGO_WRAP_CUDA_TENSOR(has_legal_movement, conf.cuda_device, torch::kUInt8, {num_envs}).clone(),
        .flag_captured = MUSTRATEGO_WRAP_CUDA_TENSOR(flag_captured, conf.cuda_device, torch::kUInt8, {num_envs}).clone(),
        .action_history = action_history,
        .board_history = board_history,
        .move_summary_history = move_summary_history,
        .chase_state = chase_state,
    }
        .Clone();
}

std::vector<std::string> StrategoRolloutBuffer::BoardStrs(const uint64_t step) const
{
    GUARD_STEP_ID

    const StrategoBoard *boards = d_boards_ + (step % buf_size) * num_envs;
    return ::BoardStrs(boards, num_envs, conf.cuda_device);
}

std::vector<std::string> StrategoRolloutBuffer::ZeroBoardStrs(const uint64_t step) const
{
    GUARD_STEP_ID

    const StrategoBoard *boards = d_zero_boards_ + (step % buf_size) * num_envs;
    return ::BoardStrs(boards, num_envs, conf.cuda_device);
}

torch::Tensor StrategoRolloutBuffer::GetIllegalChaseActions(const uint64_t step) const
{
    GUARD_STEP_ID
    MUSTRATEGO_CHECK(conf.continuous_chasing_rule, "GetIllegalChaseActions() is only available when the continuous chasing rule is enabled");

    int32_t *illegal_chase_actions = d_illegal_chase_actions_ + (step % buf_size) * num_envs * MAX_CHASE_LENGTH;
    return MUSTRATEGO_WRAP_CUDA_TENSOR(
        illegal_chase_actions,
        conf.cuda_device,
        torch::kInt32,
        {MAX_CHASE_LENGTH, num_envs});
}

void StrategoRolloutBuffer::PopulateInfostateChannelDescription_()
{
    assert(INFOSTATE_CHANNEL_DESCRIPTION.empty());

    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "flag", "bomb"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "flag", "bomb"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_" + piece + "_prob");
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "flag", "bomb"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_" + piece + "_prob");
    INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_hidden_bool");
    INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_hidden_bool");
    INFOSTATE_CHANNEL_DESCRIPTION.push_back("empty_bool");
    INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_moved_bool");
    INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_moved_bool");
    INFOSTATE_CHANNEL_DESCRIPTION.push_back("max_num_moves_frac");
    INFOSTATE_CHANNEL_DESCRIPTION.push_back("max_num_moves_between_attacks_frac");
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("we_threatened_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("we_evaded_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("we_actively_adj_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("they_threatened_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("they_evaded_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("they_actively_adj_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_dead_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_dead_" + piece);
    for (const std::string &reason : {"attacked_visible_stronger", "attacked_visible_tie", "attacked_hidden", "visible_defended_weaker", "visible_defended_tie", "hidden_defended"})
        for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal"})
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_deathstatus_" + reason + "_" + piece);
    for (const std::string &reason : {"attacked_visible_stronger", "attacked_visible_tie", "attacked_hidden", "visible_defended_weaker", "visible_defended_tie", "hidden_defended"})
        for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal"})
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_deathstatus_" + reason + "_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_protected_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_protected_against_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_was_protected_by_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_was_protected_against_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_protected_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_protected_against_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_was_protected_by_" + piece);
    for (const std::string &piece : {"spy", "scout", "miner", "sergeant", "lieutenant", "captain", "major", "colonel", "general", "marshal", "bomb", "empty", "unknown"})
        INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_was_protected_against_" + piece);

    assert(INFOSTATE_CHANNEL_DESCRIPTION.size() == NUM_BOARD_STATE_CHANNELS);

    if (conf.enable_src_dst_planes)
    {
        for (int i = 0; i < conf.move_memory; ++i)
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("src_dst_cell[-" + std::to_string(conf.move_memory - i) + "]");
    }
    if (conf.enable_hidden_and_types_planes)
    {
        for (int i = 0; i < conf.move_memory; ++i)
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_hidden[-" + std::to_string(conf.move_memory - i) + "]");
        for (int i = 0; i < conf.move_memory; ++i)
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_hidden[-" + std::to_string(conf.move_memory - i) + "]");
        for (int i = 0; i < conf.move_memory; ++i)
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("our_types[-" + std::to_string(conf.move_memory - i) + "]");
        for (int i = 0; i < conf.move_memory; ++i)
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("their_visible_types[-" + std::to_string(conf.move_memory - i) + "]");
    }
    if (conf.enable_dm_planes)
    {
        for (int i = 0; i < conf.move_memory; ++i)
            INFOSTATE_CHANNEL_DESCRIPTION.push_back("dm[-" + std::to_string(conf.move_memory - i) + "]");
    }
    MUSTRATEGO_CHECK(INFOSTATE_CHANNEL_DESCRIPTION.size() == NUM_INFOSTATE_CHANNELS,
                     "*INTERNAL BUG*: Infostate channel number mismatch with description");
}

std::vector<std::pair<uint8_t, uint8_t>> ActionsToAbsCoordinates(torch::Tensor actions, const uint8_t player)
{
    MUSTRATEGO_CHECK_IS_DTYPE(actions, torch::kInt32, "Action tensor must be of dtype `torch.int32`");
    MUSTRATEGO_CHECK(actions.dim() == 1, "Action tensor must be 1-dimensional (found %zd dimensions)", actions.dim());
    MUSTRATEGO_CHECK((player == 0 || player == 1), "Unexpected player argument: expected 0 or 1, found %d", player);

    if (actions.is_cuda())
    {
        actions = actions.cpu();
    }
    std::vector<std::pair<uint8_t, uint8_t>> out(actions.numel());
    const int32_t *ptr = actions.data_ptr<int32_t>();

#pragma omp parallel for
    for (int64_t i = 0; i < actions.numel(); ++i)
    {
        const int32_t action = *(ptr + i);
        MUSTRATEGO_CHECK(0 <= action && action < int32_t(NUM_ACTIONS), "Invalid action: out-of-range (found: %d, valid range: [0, %d])", action, NUM_ACTIONS - 1);

        uint8_t from_cell = action % 100;
        uint8_t to_cell = from_cell;

        uint8_t new_coord = action / 100;
        if (new_coord < 9)
        {
            to_cell = 10 * (new_coord + (new_coord >= from_cell / 10)) + from_cell % 10;
        }
        else
        {
            new_coord -= 9;
            to_cell = 10 * (from_cell / 10) + (new_coord + (new_coord >= from_cell % 10));
        }

        if (player == 1)
        {
            from_cell = 99 - from_cell;
            to_cell = 99 - to_cell;
        }

        out[i].first = from_cell;
        out[i].second = to_cell;
    }

    return out;
}

std::vector<int32_t> AbsCoordinatesToActions(const std::vector<std::pair<uint8_t, uint8_t>> movements, const uint8_t player)
{
    MUSTRATEGO_CHECK((player == 0 || player == 1), "Unexpected player argument: expected 0 or 1, found %d", player);

    const std::pair<uint8_t, uint8_t> *ptr = movements.data();
    std::vector<int32_t> out(movements.size(), -1);

#pragma omp parallel for
    for (size_t i = 0; i < movements.size(); ++i)
    {
        MUSTRATEGO_CHECK(movements[i].first < 100, "Source cell at index %zu not in range [0, 99] (found: %d).", i, movements[i].first);
        MUSTRATEGO_CHECK(movements[i].second < 100, "Destination cell at index %zu not in range [0, 99] (found: %d).", i, movements[i].second);

        const uint8_t src_cell = (player == 0) ? movements[i].first : 99 - movements[i].first;
        const uint8_t src_row = src_cell / 10;
        const uint8_t src_col = src_cell % 10;

        const uint8_t dst_cell = (player == 0) ? movements[i].second : 99 - movements[i].second;
        const uint8_t dst_row = dst_cell / 10;
        const uint8_t dst_col = dst_cell % 10;

        MUSTRATEGO_CHECK((src_row != dst_row) ^ (src_col != dst_col) == 1,
                         "Invalid movement at index %zu (src_cell: %d, dst_cell: %d). "
                         "Exactly one between the row or colum index needs to change.",
                         i, movements[i].first, movements[i].second);

        int32_t action = src_cell;
        if (src_col == dst_col)
        {
            // Vertical movement
            action += (dst_row - (dst_row > src_row)) * 100;
        }
        else
        {
            action += (dst_col - (dst_col > src_col)) * 100 + 900;
        }

        MUSTRATEGO_CHECK(action < int32_t(NUM_ACTIONS), "*Internal bug*: action out of range");
        out[i] = action;
    }

    return out;
}
