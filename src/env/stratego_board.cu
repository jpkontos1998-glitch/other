#include "src/env/stratego_board.h"
#include "src/util.h"
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <c10/core/Layout.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <numeric>
#include <torch/types.h>

namespace
{
    std::array<uint8_t, NUM_PIECE_TYPES> PieceArrangementToTypeCounts(const uint8_t *data)
    {
        std::array<uint8_t, NUM_PIECE_TYPES> counts = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t j = 0; j < 40; ++j)
        {
            const uint8_t piece = data[j];
            MUSTRATEGO_CHECK(piece < NUM_PIECE_TYPES, "Piece type out of range (found: %d)", piece);
            ++counts[piece];
        }

        return counts;
    }

    constexpr const std::array<uint8_t, 13> CHAR_TO_PIECE_TYPE = {
        EMPTY,      // A
        BOMB,       // B
        SPY,        // C
        SCOUT,      // D
        MINER,      // E
        SERGEANT,   // F
        LIEUTENANT, // G
        CAPTAIN,    // H
        MAJOR,      // I
        COLONEL,    // J
        GENERAL,    // K
        MARSHAL,    // L
        FLAG        // M
    };

    void CheckStringArrangementsValidity(const StringArrangements &arrangements)
    {
#pragma omp parallel for
        for (uint32_t i = 0; i < arrangements.size(); ++i)
        {
            MUSTRATEGO_CHECK(arrangements[i].length() == 40,
                             "Invalid length of arrangement at index %d (found: %zu, expected: 40)",
                             i, arrangements[i].length());
            for (uint32_t j = 0; j < 40; ++j)
            {
                MUSTRATEGO_CHECK(arrangements[i][j] != '_', "Lake character (`_`) is invalid in arrangement");
                MUSTRATEGO_CHECK(arrangements[i][j] >= 'A' && arrangements[i][j] <= 'M',
                                 "Arrangement character `%c` out of range (expected >= `A` <= `M`)",
                                 arrangements[i][j]);
            }
        }
    }
} // namespace

std::string StrategoBoard::BoardString() const
{
    std::string out;
    out.reserve(201);
    for (uint32_t r = 0; r < BOARD_SIZE; ++r)
    {
        for (uint32_t c = 0; c < BOARD_SIZE; ++c)
        {
            const Piece p = pieces[r][c];
            char ch = PIECE_ENCODING[p.color][p.type];
            MUSTRATEGO_CHECK(ch != '#', "Invalid piece type # (row: %d, col: %d; found color: %d, type: %d)", r, c, p.color, p.type);

            if (p.visible)
                ch = tolower(ch);

            out += ch;
            out += p.has_moved ? '.' : '@';
        }
    }
    return out;
}

void CheckIsValidInitBoard(const StrategoBoard &board)
{
    std::array<uint8_t, NUM_PIECE_TYPES> count_red = {0};
    std::array<uint8_t, NUM_PIECE_TYPES> count_blue = {0};

    for (int i = 0; i < 10; ++i)
    {
        if (i < 4 || i > 5)
        { // Player rows
            for (int j = 0; j < 10; ++j)
            {
                uint8_t pov_cell_id = i * 10 + j;
                if (i > 5)
                    pov_cell_id = 99 - pov_cell_id;

                MUSTRATEGO_CHECK(board.pieces[i][j].type < NUM_PIECE_TYPES, "Piece at (%d, %d) is out of range", i, j);
                MUSTRATEGO_CHECK(board.pieces[i][j].type != LAKE, "Lake piece in init player rows at (%d, %d)", i, j);
                if (board.pieces[i][j].type == EMPTY)
                {
                    MUSTRATEGO_CHECK(board.pieces[i][j].color == 0, "Piece at (%d, %d) have the wrong color (expected: 0)", i, j);
                }
                else
                {
                    MUSTRATEGO_CHECK(board.pieces[i][j].color == 1 + (i > 5), "Piece at (%d, %d) have the wrong color (expected: %d)", i, j, 1 + (i > 5));
                }
                MUSTRATEGO_CHECK(board.pieces[i][j].type == EMPTY || !board.pieces[i][j].visible, "Non-empty player piece at (%d, %d) is visible", i, j);
                MUSTRATEGO_CHECK(!board.pieces[i][j].has_moved, "Piece at (%d, %d) has moved", i, j);
                MUSTRATEGO_CHECK(board.pieces[i][j].piece_id == pov_cell_id, "Piece at (%d, %d) has wrong piece ID (found: %d, expected: %d)", i, j, board.pieces[i][j].piece_id, pov_cell_id);

                if (i < 4)
                    ++count_red[board.pieces[i][j].type];
                else
                    ++count_blue[board.pieces[i][j].type];
            }
        }
        else
        {
            for (int j = 0; j < 10; ++j)
            {
                if (j % 4 < 2)
                {
                    MUSTRATEGO_CHECK(board.pieces[i][j].color == 0, "Piece at (%d, %d) have the wrong color (expected: 0)", i, j);
                    MUSTRATEGO_CHECK(board.pieces[i][j].type == EMPTY, "Piece at (%d, %d) should be empty (found: %d)", i, j, board.pieces[i][j].type);
                }
                else
                {
                    MUSTRATEGO_CHECK(board.pieces[i][j].color == 3, "Piece at (%d, %d) have the wrong color (expected: 3)", i, j);
                    MUSTRATEGO_CHECK(board.pieces[i][j].type == LAKE, "Piece at (%d, %d) should be lake (found: %d)", i, j, board.pieces[i][j].type);
                }
                MUSTRATEGO_CHECK(board.pieces[i][j].visible, "Piece at (%d, %d) cannot be hidden", i, j);
                MUSTRATEGO_CHECK(!board.pieces[i][j].has_moved, "Piece at (%d, %d) cannot move", i, j);
            }
        }
    }

    MUSTRATEGO_CHECK(count_blue == count_red, "Blue/red counts mismatch");
    uint8_t total_hidden = 0;
    for (int t = 0; t < 12; ++t)
    {
        MUSTRATEGO_CHECK(count_red[t] == board.num_hidden[0][t], "Red counter mismatch on board (type: %d, found: %d, counted: %d)", t, board.num_hidden[0][t], count_red[t]);
        MUSTRATEGO_CHECK(count_blue[t] == board.num_hidden[1][t], "Blue counter mismatch on board (type: %d, found: %d, counted: %d)", t, board.num_hidden[1][t], count_blue[t]);
        total_hidden += count_red[t];
    }
    MUSTRATEGO_CHECK(total_hidden == board.num_hidden_unmoved[0], "Num hidden unmoved for player 0 mismatches with total piece count (found: %d, expected: %d)", board.num_hidden_unmoved[0], total_hidden);
    MUSTRATEGO_CHECK(total_hidden == board.num_hidden_unmoved[1], "Num hidden unmoved for player 1 mismatches with total piece count (found: %d, expected: %d)", board.num_hidden_unmoved[1], total_hidden);
}

std::vector<std::string> BoardStrs(const StrategoBoard *d_boards, const uint32_t num_boards, const uint32_t cuda_device)
{
    MUSTRATEGO_CHECK((size_t)d_boards % 128 == 0, "Unexpected alignment of `d_boards` is not a multiple of 128!");

    std::vector<std::string> strings(num_boards);
    StrategoBoard *h_boards = new StrategoBoard[num_boards];
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(cuda_device));
    MUSTRATEGO_CUDA_CHECK(cudaMemcpy(h_boards, d_boards, num_boards * sizeof(StrategoBoard), cudaMemcpyDeviceToHost));

#pragma omp parallel for
    for (uint32_t env_idx = 0; env_idx < num_boards; ++env_idx)
    {
        strings[env_idx] = h_boards[env_idx].BoardString();
    }

    delete[] h_boards;
    return strings;
}

PieceArrangementGenerator::PieceArrangementGenerator(const std::array<uint8_t, NUM_PIECE_TYPES> &type_counts) : type_counts_(type_counts)
{
    MUSTRATEGO_CHECK(type_counts_[LAKE] == 0, "The piece count for lakes must be 0 (found: %d)", type_counts_[LAKE]);
    num_pieces_ = std::accumulate(type_counts_.begin(), type_counts_.end(), 0);
    MUSTRATEGO_CHECK(num_pieces_ == 40, "The total piece count must be == 40 (found: %d)", num_pieces_);

    InitializeCache_();

    // Compute number of possible arrangements.
    num_arrangements_ = 1;
    uint32_t tally = 0;
    std::stringstream ss;
    ss << '{';
    for (uint8_t i = 0; i < NUM_PIECE_TYPES; ++i)
    {
        const uint32_t c = type_counts_[i];
        if (i)
            ss << ", ";
        ss << c;

        tally += c;
        num_arrangements_ *= cache_[tally][c];
    }
    ss << '}';
    MUSTRATEGO_LOG("Constructed PieceArrangementGenerator (type_counts: %s, num_arrangements: %s)", ss.str().c_str(), UInt128ToString(num_arrangements_).c_str());
}

PieceArrangementGenerator::PieceArrangementGenerator(const BoardVariant variant)
    : PieceArrangementGenerator((variant == CLASSIC) ? CLASSIC_INITIAL_COUNTS : BARRAGE_INITIAL_COUNTS)
{
    MUSTRATEGO_CHECK(variant == CLASSIC || variant == BARRAGE, "Invalid board variant (found %d)", variant);
}

torch::Tensor PieceArrangementGenerator::GenerateArrangements(const std::vector<uint128_t> &board_ids) const
{
    return ArrangementTensorFromStrings(GenerateStringArrangements(board_ids));
}

StringArrangements PieceArrangementGenerator::GenerateStringArrangements(const std::vector<uint128_t> &board_ids) const
{
    MUSTRATEGO_CHECK(!board_ids.empty(), "Empty list of board ids");
    for (uint32_t i = 0; i < board_ids.size(); ++i)
    {
        MUSTRATEGO_CHECK(board_ids[i] < num_arrangements_, "Requested board ID at index %d exceeds the number of valid arrangements", i);
    }

    StringArrangements out(board_ids.size(), std::string(40, '?'));

#pragma omp parallel for
    for (uint32_t i = 0; i < board_ids.size(); ++i)
    {
        uint128_t board_id = board_ids[i];
        uint128_t arrangements_in_block = num_arrangements_;
        std::array<uint8_t, NUM_PIECE_TYPES> counts = type_counts_;

        for (uint32_t j = 0; j < num_pieces_; ++j)
        {
            assert(board_id < arrangements_in_block);

            const uint32_t num_remaining_pieces = num_pieces_ - j;
            bool found = false;
            for (uint8_t type = SPY; type < NUM_PIECE_TYPES; ++type)
            {
                // Skip piece types that have run out
                if (!counts[type])
                    continue;

                uint128_t sub = arrangements_in_block;
                sub *= counts[type];
                assert(sub % num_remaining_pieces == 0);
                sub /= num_remaining_pieces;
                if (board_id >= sub)
                {
                    board_id -= sub;
                }
                else
                {
                    assert(counts[type] > 0);
                    counts[type] -= 1;
                    out[i][j] = "CDEFGHIJKLMB_A"[type];
                    arrangements_in_block = sub;
                    found = true;
                    break;
                }
            }
            assert(found);
            found = found; // Suppresses unused warnings
        }
    }

    return out;
}

std::vector<uint128_t> PieceArrangementGenerator::ArrangementIds(torch::Tensor arrangements) const
{
    MUSTRATEGO_CHECK(arrangements.dim() == 2, "Input `arrangements` tensor should be 2D (found: %zd dimensions)",
                     arrangements.dim());
    MUSTRATEGO_CHECK(arrangements.size(1) == 40, "Input `arrangements` tensor should have [*, 40] shape. Found: [%zd, %zd]",
                     arrangements.size(0), arrangements.size(1));
    MUSTRATEGO_CHECK(arrangements.size(0) > 0, "Empty tensor");
    MUSTRATEGO_CHECK(arrangements.dtype() == torch::kUInt8, "Unexpected tensor dtype (expected: torch::kUint8)");

    if (arrangements.is_cuda())
    {
        // Move tensor to CPU since the algorithm is implemented on CPU (we need uint128_t support)
        arrangements = arrangements.cpu();
    }
    const uint32_t num_arrangements = arrangements.size(0);
    const uint8_t *data = arrangements.data_ptr<uint8_t>();
    std::vector<uint128_t> out(num_arrangements, 0);

#pragma omp parallel for
    for (uint32_t i = 0; i < num_arrangements; ++i)
    {
        std::array<uint8_t, 14> counts = PieceArrangementToTypeCounts(data + (i * 40));
        MUSTRATEGO_CHECK(counts == type_counts_, "Piece count for board index %d does not match the generator's type count", i);

        uint128_t arrangements_in_block = num_arrangements_;

        for (uint32_t j = 0; j < 40; ++j)
        {
            const uint32_t num_remaining_pieces = 40 - j;
            for (uint8_t type = SPY; type < data[i * 40 + j]; ++type)
            {
                if (!counts[type])
                    continue;

                uint128_t sub = arrangements_in_block;
                sub *= counts[type];
                assert(sub % num_remaining_pieces == 0);
                sub /= num_remaining_pieces;

                out[i] += sub;
            }

            // Finally, complete the induction step.
            arrangements_in_block *= counts[data[i * 40 + j]];
            assert(arrangements_in_block % num_remaining_pieces == 0);
            arrangements_in_block /= num_remaining_pieces;

            assert(counts[data[i * 40 + j]] > 0);
            --counts[data[i * 40 + j]];
        }
    }

    return out;
}

void PieceArrangementGenerator::InitializeCache_()
{
    for (uint32_t n = 0; n <= 40; ++n)
    {
        cache_[n][0] = 1;
        for (uint32_t k = 1; k <= n; ++k)
        {
            cache_[n][k] = (cache_[n][k - 1] * (n - k + 1));
            assert(cache_[n][k] % k == 0);
            cache_[n][k] /= k;
        }
    }
}

torch::Tensor ArrangementTensorFromStrings(const StringArrangements &arrangements)
{
    CheckStringArrangementsValidity(arrangements);

    const uint32_t num_arrangements = arrangements.size();
    torch::TensorOptions options = torch::TensorOptions()
                                       .device(torch::kCPU)
                                       .dtype(torch::kUInt8)
                                       .layout(torch::kStrided);
    torch::Tensor arrangements_tensor = torch::zeros({(long)num_arrangements, 40}, options);
    uint8_t *data = arrangements_tensor.data_ptr<uint8_t>();

#pragma omp parallel for
    for (uint32_t i = 0; i < num_arrangements; ++i)
        for (uint32_t j = 0; j < 40; ++j)
        {
            assert((arrangements[i][j] - 'A') >= 0 &&
                   (arrangements[i][j] - 'A') < ::CHAR_TO_PIECE_TYPE.size());
            data[40 * i + j] = ::CHAR_TO_PIECE_TYPE[arrangements[i][j] - 'A'];
        }

    return arrangements_tensor;
}

std::vector<bool> IsTerminalArrangement(const StringArrangements &arrangements)
{
    CheckStringArrangementsValidity(arrangements);
    // WARNING: This cannot be directly a vector of bools, because vector<bool>
    // is NOT thread safe due to bit packing.
    std::vector<uint8_t> out(arrangements.size(), 1);

#define IS_MOVABLE(c) (((c) > 'B') && ((c) != 'M'))

#pragma omp parallel for
    for (uint32_t i = 0; i < arrangements.size(); ++i)
    {
        const std::string &a = arrangements[i];
        for (uint32_t j = 0; j < 40; ++j)
        {
            const bool movable_close_to_empty = ((a[j] == 'A') && ((j % 10 != 0 && IS_MOVABLE(a[j - 1])) ||
                                                                   (j % 10 != 9 && IS_MOVABLE(a[j + 1])) ||
                                                                   (j >= 10 && IS_MOVABLE(a[j - 10])) ||
                                                                   (j <= 29 && IS_MOVABLE(a[j + 10]))));
            const bool movable_close_to_corridor = IS_MOVABLE(a[j]) && (j == 30 || j == 31 || j == 34 || j == 35 || j >= 38);

            if (movable_close_to_empty || movable_close_to_corridor)
            {
                out[i] = 0;
            }
        }
    }
#undef IS_MOVABLE

    return {out.begin(), out.end()};
}

std::vector<std::string> ArrangementStringsFromTensor(torch::Tensor arrangements)
{
    MUSTRATEGO_CHECK(arrangements.dim() == 2, "Input `arrangements` tensor should be 2D (found: %zd dimensions)",
                     arrangements.dim());
    MUSTRATEGO_CHECK(arrangements.size(1) == 40, "Input `arrangements` tensor should have [*, 40] shape. Found: [%zd, %zd]",
                     arrangements.size(0), arrangements.size(1));
    MUSTRATEGO_CHECK(arrangements.size(0) > 0, "Empty tensor");
    MUSTRATEGO_CHECK(arrangements.dtype() == torch::kUInt8, "Unexpected tensor dtype (expected: torch::kUint8)");

    if (arrangements.is_cuda())
    {
        // Move tensor to CPU since the algorithm is implemented on CPU
        arrangements = arrangements.cpu();
    }
    const uint32_t num_arrangements = arrangements.size(0);
    const uint8_t *data = arrangements.data_ptr<uint8_t>();
    std::vector<std::string> out(num_arrangements, std::string(40, '?'));

#pragma omp parallel for
    for (uint32_t i = 0; i < num_arrangements; ++i)
        for (uint32_t j = 0; j < 40; ++j)
        {
            const uint8_t type = data[40 * i + j];
            MUSTRATEGO_CHECK(type < NUM_PIECE_TYPES, "Invalid piece type `%d` in tensor at index [%u, %u]", int(type), i, j);
            MUSTRATEGO_CHECK(type != LAKE, "Unexpected lake in tensor arrangement");

            assert(out.at(i).length() == 40);
            out[i][j] = "CDEFGHIJKLMB_A"[type];
        }

    return out;
}

torch::Tensor GenerateInitializationBoards(torch::Tensor red_arrangements,
                                           torch::Tensor blue_arrangements,
                                           const int32_t cuda_device)
{
    MUSTRATEGO_CUDA_CHECK(cudaSetDevice(cuda_device));

    MUSTRATEGO_CHECK(red_arrangements.dim() == 2, "Input `red_arrangements` tensor should be 2D (found: %zd dimensions)",
                     red_arrangements.dim());
    MUSTRATEGO_CHECK(red_arrangements.size(1) == 40, "Input `red_arrangements` tensor should have [*, 40] shape. Found: [%zd, %zd]",
                     red_arrangements.size(0), red_arrangements.size(1));
    MUSTRATEGO_CHECK(blue_arrangements.dim() == 2, "Input `blue_arrangements` tensor should be 2D (found: %zd dimensions)",
                     blue_arrangements.dim());
    MUSTRATEGO_CHECK(blue_arrangements.size(1) == 40, "Input `blue_arrangements` tensor should have [*, 40] shape. Found: [%zd, %zd]",
                     blue_arrangements.size(0), blue_arrangements.size(1));
    MUSTRATEGO_CHECK(red_arrangements.size(0) == blue_arrangements.size(0),
                     "Red and blue input arrangement tensors mush have same first dimensions (found red: %zd, blue: %zd)",
                     red_arrangements.size(0), blue_arrangements.size(0));
    MUSTRATEGO_CHECK(red_arrangements.dtype() == torch::kUInt8, "Red arrangement tensor must have dtype `torch::kUInt8`");
    MUSTRATEGO_CHECK(blue_arrangements.dtype() == torch::kUInt8, "Blue arrangement tensor must have dtype `torch::kUInt8`");

    if (red_arrangements.is_cuda())
        red_arrangements = red_arrangements.cpu();
    if (blue_arrangements.is_cuda())
        blue_arrangements = blue_arrangements.cpu();

    const uint32_t num_arrangements = red_arrangements.size(0);
    const uint8_t *red_data = red_arrangements.data_ptr<uint8_t>();
    const uint8_t *blue_data = blue_arrangements.data_ptr<uint8_t>();

    const std::array<uint8_t, NUM_PIECE_TYPES> counts = PieceArrangementToTypeCounts(red_data);
    uint8_t num_nonempty = 0;
    for (uint32_t i = 0; i < 12; ++i)
        num_nonempty += counts[i];

// Make sure that all boards have the same piece counts.
#pragma omp parallel for
    for (uint32_t i = 0; i < num_arrangements; ++i)
    {
        MUSTRATEGO_CHECK(counts == PieceArrangementToTypeCounts(red_data + (i * 40)), "Red arrangements type count mismatch at index %d", i);
        MUSTRATEGO_CHECK(counts == PieceArrangementToTypeCounts(blue_data + (i * 40)), "Blue arrangements type count mismatch at index %d", i);
    }

    MUSTRATEGO_LOG("Generating initialization boards (num arrangements: %d)", num_arrangements);
    StrategoBoard *boards = new StrategoBoard[num_arrangements];
    memset(boards, 0, num_arrangements * sizeof(StrategoBoard));

#pragma omp parallel for
    for (uint32_t i = 0; i < num_arrangements; ++i)
    {
        uint8_t *board = (uint8_t *)(boards + i);
        constexpr const int SP = 16;

        // Start from lakes
        board[0 + SP * 40] = board[0 + SP * 41] = board[0 + SP * 44] = board[0 + SP * 45] = board[0 + SP * 48] = board[0 + SP * 49] = 0x4d; // Empty: 13 + 0 + 64
        board[0 + SP * 50] = board[0 + SP * 51] = board[0 + SP * 54] = board[0 + SP * 55] = board[0 + SP * 58] = board[0 + SP * 59] = 0x4d; // Empty: 13 + 0 + 64
        board[1 + SP * 40] = board[1 + SP * 41] = board[1 + SP * 44] = board[1 + SP * 45] = board[1 + SP * 48] = board[1 + SP * 49] = 0xff; // Piece ID
        board[1 + SP * 50] = board[1 + SP * 51] = board[1 + SP * 54] = board[1 + SP * 55] = board[1 + SP * 58] = board[1 + SP * 59] = 0xff; // Piece ID

        board[0 + SP * 42] = board[0 + SP * 43] = board[0 + SP * 46] = board[0 + SP * 47] = 0x7c; // Lake: 12 + 3 * 16 + 1 * 64
        board[0 + SP * 52] = board[0 + SP * 53] = board[0 + SP * 56] = board[0 + SP * 57] = 0x7c; // Lake: 12 + 3 * 16 + 1 * 64
        board[1 + SP * 42] = board[1 + SP * 43] = board[1 + SP * 46] = board[1 + SP * 47] = 0xff; // Piece ID
        board[1 + SP * 52] = board[1 + SP * 53] = board[1 + SP * 56] = board[1 + SP * 57] = 0xff; // Piece ID

        // Copy the red pieces into position without flipping. Arrangement[0] will go to cell (row 0, col 0), ...,
        // arrangement[39] will go to cell (row 3, col 9).
        for (uint32_t j = 0; j < 40; ++j)
        {
            const PieceType type = (PieceType)red_data[40 * i + j];
            assert(type != LAKE);
            *(Piece *)(&board[SP * j]) = ((type == EMPTY) ? Piece _E(type) : Piece _R(type, j));
        }

        // Copy the blue pieces into position. Arrangement[0] will go to cell (row 9, col 9), ...,
        // arrangement[39] will go to cell (row 6, col 0).
        for (uint32_t j = 0; j < 40; ++j)
        {
            const PieceType type = (PieceType)blue_data[40 * i + j];
            assert(type != LAKE);
            *(Piece *)(&board[SP * (99 - j)]) = ((type == EMPTY) ? Piece _E(type) : Piece _B(type, j));
        }

        // Fill in the piece counts for each player (`num_hidden`)
        for (uint32_t j = 0; j < 12; ++j)
            board[1600 + j] = board[1612 + j] = counts[j];

        // Fill in `num_hidden_unmoved`
        board[1624] = board[1625] = num_nonempty;
        board[1628] = board[1629] = board[1630] = 0xff; // prev_dst_abs, prev_prev_dst_abs, last_moved_piece_type
    }

    const torch::Tensor boards_tensor =
        torch::from_blob((uint8_t *)boards,
                         {num_arrangements, sizeof(StrategoBoard)},
                         torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8).layout(torch::kStrided))
            .to(torch::Device(torch::kCUDA, cuda_device));
    delete[] boards;
    MUSTRATEGO_CHECK((size_t)boards_tensor.data_ptr<uint8_t>() % 128 == 0, "Unexpected alignment of `boards_tensor` is not a multiple of 128!");

    // As a final postprocessing, we check that none of the initialization boards are terminal.
    {
        torch::Tensor scratch;
        torch::Tensor has_legal_moves;
        MUSTRATEGO_CREATE_CUDA_TENSOR(scratch, cuda_device, torch::kUInt8, {num_arrangements, 100});
        MUSTRATEGO_CREATE_CUDA_TENSOR(has_legal_moves, cuda_device, torch::kBool, {num_arrangements});

        const uint32_t num_threads = 1024;
        const uint32_t num_blocks = ceil(100ll * num_arrangements, num_threads);

        SaturatedNumMovementDirectionsKernel<<<num_blocks, num_threads>>>(
            scratch.data_ptr<uint8_t>(),
            (const StrategoBoard *)boards_tensor.data_ptr<uint8_t>(),
            num_arrangements,
            1 /* player 0 */);

        torch::any_out(
            has_legal_moves,
            scratch,
            /* dim */ 1,
            /* keepdim */ false);
        if (!has_legal_moves.all().item<bool>())
        {
            torch::Tensor offending = red_arrangements.index({~has_legal_moves.cpu()});
            assert(offending.size(0) > 0);

            std::stringstream terminal;
            terminal << offending;
            MUSTRATEGO_FATAL("The following red arrangements are terminal:\n%s", terminal.str().c_str());
        }
    }
    MUSTRATEGO_LOG("...done");
    return boards_tensor;
}

torch::Tensor InitializationBoardFromString(std::string s, const int32_t cuda_device)
{
    std::transform(s.begin(), s.end(), s.begin(), toupper);
    MUSTRATEGO_CHECK(s.length() == 100, "Board string encoding must be of size 100 (found: %zu). Did you forget the movement characters `@.` in it?", s.length());
    MUSTRATEGO_CHECK(s.substr(40, 10) == "AA__AA__AA", "Row 5 must contains empty pieces and lakes only (expected: `AA__AA__AA`, found: `%s`)", s.substr(40, 10).c_str());
    MUSTRATEGO_CHECK(s.substr(50, 10) == "AA__AA__AA", "Row 6 must contains empty pieces and lakes only (expected: `AA__AA__AA`, found: `%s`)", s.substr(50, 10).c_str());

    torch::Tensor red_arrangement = torch::zeros({1, 40}, torch::TensorOptions().dtype(torch::kUInt8));
    torch::Tensor blue_arrangement = torch::zeros({1, 40}, torch::TensorOptions().dtype(torch::kUInt8));

    for (int i = 0; i < 40; ++i)
    {
        MUSTRATEGO_CHECK(s[i] != '_' && s[i] >= 'A' && s[i] <= 'M', "Red arrangement must contain chars 'A' .. 'M' only (found: %c)", s[i]);
        red_arrangement.index_put_({0, i}, CHAR_TO_PIECE_TYPE.at(s[i] - 'A'));
    }

    for (int i = 99; i >= 60; --i)
    {
        MUSTRATEGO_CHECK(s[i] != '_' && (s[i] == 'A' || (s[i] >= 'N' && s[i] <= 'Y')), "Blue arrangement must contain chars 'A', 'N' .. 'Y' only (found: %c)", s[i]);
        if (s[i] == 'A')
            blue_arrangement.index_put_({0, 99 - i}, EMPTY);
        else
            blue_arrangement.index_put_({0, 99 - i}, CHAR_TO_PIECE_TYPE.at(s[i] - 'M'));
    }

    return GenerateInitializationBoards(red_arrangement, blue_arrangement, cuda_device);
}

__global__ void SaturatedNumMovementDirectionsKernel(
    uint8_t *d_out,
    const StrategoBoard *d_boards,
    const uint32_t num_envs,
    const uint8_t for_player)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;

    if (env_idx >= num_envs)
        return;

    const int32_t row_idx = (index % 100) / 10;
    const int32_t col_idx = index % 10;

    // Do not copy, only keep as reference
    const StrategoBoard *board = d_boards + env_idx;
    const Piece piece = board->pieces[row_idx][col_idx];

    uint8_t num_directions = (row_idx > 0 && !(board->pieces[row_idx - 1][col_idx].color & for_player)) +
                             (row_idx < 9 && !(board->pieces[row_idx + 1][col_idx].color & for_player)) +
                             (col_idx > 0 && !(board->pieces[row_idx][col_idx - 1].color & for_player)) +
                             (col_idx < 9 && !(board->pieces[row_idx][col_idx + 1].color & for_player));
    num_directions = (num_directions <= 1) ? num_directions : 2;
    num_directions = (piece.type < FLAG && piece.color == for_player) ? num_directions : 0;

    d_out[index] = num_directions;
}