#pragma once

#include <deque>
#include <fstream>
#include <optional>
#include <vector>

#include "src/env/stratego_board.h"

class StrategoRolloutBuffer;

/// This class is responsible for splitting games across multiple
/// environments and restarts, with data obtained in chunks, and
/// exporting them to one HDF5 file.
class GameSaver
{
public:
    /// Each game is saved according to the following structure:
    using GameStruct = std::pair<
        std::string /*  starting_board */,
        std::vector<int32_t> /* actions */
        >;

    GameSaver(const uint32_t num_envs, const std::string &outfile);
    ~GameSaver();

    /// Pushes the data from the buffer to the internal tracker.
    ///
    /// Normally, this is ignored if the step of the buffer is not the next
    /// multiple of the buffer size, unless `force` is set to true.
    ///
    /// For example, if the buffer is going to be deallocated, you will want to
    /// set force to true to avoid losing the data.
    void Push(const StrategoRolloutBuffer &buffer, const bool force = false);

    /// Discards partially completed games, and resets the internal state.
    void Reset();

private:
    std::string outfile_;
    std::ofstream file_;
    uint64_t last_step_;

    void Flush_();

    std::vector<std::deque<GameStruct>> games_;
};