#include "src/env/game_saver.h"

#include <filesystem>

#include "src/env/stratego.h"
#include "src/env/stratego_board.h"
#include "src/json.hpp"
#include "src/util.h"

using json = nlohmann::json;

GameSaver::GameSaver(const uint32_t num_envs, const std::string &outfile) : outfile_(outfile), games_(num_envs), last_step_(0)
{
    MUSTRATEGO_LOG("Constructing GameSaver (num_envs: %d, outfile: \"%s\")", num_envs, outfile.c_str());
    MUSTRATEGO_CHECK(!std::filesystem::exists(outfile), "The output file already exists. Please delete it first.");

    file_.open(outfile.c_str(), std::ios::binary | std::ios::out);
    if (!file_)
    {
        MUSTRATEGO_FATAL("Failed to open file \"%s\" for writing: %s", outfile.c_str(), std::strerror(errno));
    }
}

GameSaver::~GameSaver()
{
    MUSTRATEGO_LOG("GameSaver is about to be destructed. Flushing all games to disk.");
    Reset();
    Flush_();
}

void GameSaver::Push(const StrategoRolloutBuffer &buffer, const bool force)
{
    MUSTRATEGO_CHECK(buffer.CurrentStep() <= last_step_ + buffer.buf_size - 1,
                     "The data in the buffer is too new. Did you forget to call `Push` on the previous buffer?");
    MUSTRATEGO_CHECK(force || buffer.CurrentStep() == last_step_ + buffer.buf_size - 1, "The buffer is not full yet.");
    MUSTRATEGO_CHECK(buffer.num_envs == games_.size(), "The number of environments in the buffer does not match the number of environments in the GameSaver.");
    MUSTRATEGO_CHECK(last_step_ % buffer.buf_size == 0, "The last step is not a multiple of the buffer size.");

    MUSTRATEGO_DEBUG("Pushing buffer to GameSaver. Current step: %d, last step: %d (force: %d)...", buffer.CurrentStep(), last_step_, force);

    // env -> steps
    torch::Tensor actions = buffer.GetActionHistoryTensor().transpose(0, 1).cpu().contiguous();
    torch::Tensor nmsr = buffer.GetNumMovesSinceResetTensor().transpose(0, 1).cpu().contiguous();

    // env -> step -> 1920
    torch::Tensor boards_tensor = buffer.GetBoardTensor().transpose(0, 1).cpu().contiguous();
    assert(boards_tensor.dtype() == torch::kByte);
    assert(boards_tensor.numel() % sizeof(StrategoBoard) == 0);
    assert(boards_tensor.size(0) == buffer.num_envs);
    assert(boards_tensor.size(1) == buffer.buf_size);
    assert(boards_tensor.size(2) == sizeof(StrategoBoard));
    StrategoBoard *boards = new StrategoBoard[boards_tensor.numel() / sizeof(StrategoBoard)];
    memcpy(boards, boards_tensor.data_ptr<uint8_t>(), boards_tensor.numel());

    const uint64_t cur_step = buffer.CurrentStep();

#pragma omp parallel for
    for (int env_idx = 0; env_idx < buffer.num_envs; env_idx++)
    {
        const int32_t *actions_ptr = actions.data_ptr<int32_t>() + env_idx * buffer.buf_size;
        const int32_t *nmsr_ptr = nmsr.data_ptr<int32_t>() + env_idx * buffer.buf_size;
        const StrategoBoard *boards_ptr = boards + env_idx * buffer.buf_size;
        GameStruct *game = games_[env_idx].size() ? &games_[env_idx].back() : nullptr;

        for (uint64_t t = last_step_; t <= cur_step; ++t)
        {
            const uint64_t prev_t = (t + buffer.buf_size - 1) % buffer.buf_size;
            assert(t || !game); // game should be null if t == 0
            if (game)
                game->second.push_back(actions_ptr[prev_t]);

            if (nmsr_ptr[t % buffer.buf_size] == 0)
            {
                // A new game has started at this time on this environment.
                games_[env_idx].emplace_back();
                game = &games_[env_idx].back();

                const std::string board_str = boards_ptr[t % buffer.buf_size].BoardString();
                game->first = board_str;
            }
        }
    }
    delete[] boards;

    last_step_ = cur_step + 1;
    MUSTRATEGO_DEBUG("... all done");
}

void GameSaver::Flush_()
{
    MUSTRATEGO_CHECK(file_, "INTERNAL BUG: The file has already been closed.");

    uint32_t num_games = 0;
    std::vector<GameStruct> env_games;
    for (int env_idx = 0; env_idx < games_.size(); ++env_idx)
    {
        if (!games_[env_idx].empty())
        {
            num_games += games_[env_idx].size();
            env_games.insert(env_games.end(), std::move_iterator(games_[env_idx].begin()), std::move_iterator(games_[env_idx].end()));
            games_[env_idx].clear();
        }
    }
    const std::vector<uint8_t> msgpack = json::to_msgpack(env_games);
    file_.write(reinterpret_cast<const char *>(msgpack.data()), msgpack.size());
    file_.flush();
    file_.close();
    MUSTRATEGO_LOG("GameSaver wrote %u games to \"%s\"", num_games, outfile_.c_str());
}

void GameSaver::Reset()
{
    uint32_t num_discarded = 0;
    for (int env_idx = 0; env_idx < games_.size(); env_idx++)
    {
        if (!games_[env_idx].empty())
        {
            ++num_discarded;
            games_[env_idx].pop_back();
        }
    }
    MUSTRATEGO_LOG("GameSaver discarded %u partially completed games", num_discarded);
    last_step_ = 0;
}