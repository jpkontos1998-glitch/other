#include <ATen/ATen.h>
#include <torch/types.h>
#include <torch/torch.h>

#include "src/env/cuda/kernels.h"
#include "src/env/env_state.h"
#include "src/env/rules/chase_state.h"
#include "src/env/stratego_board.h"
#include "src/util.h"

namespace
{
    inline void ReplicateRow(torch::Tensor data, uint32_t row)
    {
        const auto sizes = data.sizes();
        const uint32_t n_rows = sizes.at(0);
        assert(sizes.size() == 2 && row < n_rows);

        torch::Tensor slice = data.index({(int)row, torch::indexing::Slice()}).unsqueeze(0);
        torch::repeat_out(data, slice, {n_rows, 1});
    }

    inline void ReplicateColumn(torch::Tensor data, uint32_t col)
    {
        const auto sizes = data.sizes();
        if (sizes.size() == 1)
        {
            assert(col < sizes.at(0));
            data.fill_(data[col]);
        }
        else if (sizes.size() == 2)
        {
            const uint32_t n_cols = sizes.at(1);
            assert(col < n_cols);

            torch::Tensor slice = data.index({torch::indexing::Slice(), (int)col}).unsqueeze(1);
            torch::repeat_out(data, slice, {1, n_cols});
        }
        else
        {
            const uint32_t n_cols = sizes.at(1);
            assert(sizes.size() == 3 && col < n_cols);

            torch::Tensor slice = data.index({torch::indexing::Slice(), (int)col, torch::indexing::Slice()}).unsqueeze(1);
            torch::repeat_out(data, slice, {1, n_cols, 1});
        }
    }
} // namespace

std::vector<std::string> EnvState::BoardStrs() const
{
    MUSTRATEGO_CHECK((size_t)boards.data_ptr<uint8_t>() % 128 == 0, "Unexpected alignment of board tensor");
    return ::BoardStrs((const StrategoBoard *)boards.data_ptr<uint8_t>(), num_envs, boards.device().index());
}

std::vector<std::string> EnvState::ZeroBoardStrs() const
{
    MUSTRATEGO_CHECK((size_t)boards.data_ptr<uint8_t>() % 128 == 0, "Unexpected alignment of zero board tensor");
    return ::BoardStrs((const StrategoBoard *)zero_boards.data_ptr<uint8_t>(), num_envs, zero_boards.device().index());
}

void EnvState::ReplicateEnv(const uint32_t idx)
{
    MUSTRATEGO_CHECK(idx < num_envs, "Index out of range (passed `idx` was %d, but `num_envs` is %d)", idx, num_envs);
    ReplicateRow(boards, idx);
    ReplicateRow(zero_boards, idx);
    ReplicateColumn(num_moves, idx);
    ReplicateColumn(num_moves_since_last_attack, idx);
    ReplicateColumn(terminated_since, idx);
    ReplicateColumn(has_legal_movement, idx);
    ReplicateColumn(flag_captured, idx);
    ReplicateColumn(action_history, idx);
    ReplicateColumn(board_history, idx);
    ReplicateColumn(move_summary_history, idx);
    if (chase_state)
    {
        ReplicateColumn(chase_state->last_dst_pos[0], idx);
        ReplicateColumn(chase_state->last_dst_pos[1], idx);
        ReplicateColumn(chase_state->last_src_pos[0], idx);
        ReplicateColumn(chase_state->last_src_pos[1], idx);
        ReplicateColumn(chase_state->chase_length[0], idx);
        ReplicateColumn(chase_state->chase_length[1], idx);
    }
}

void EnvState::Tile(const uint32_t num_tiles)
{
    MUSTRATEGO_CHECK(num_tiles >= 1, "The number of tiles cannot be 0.");

    if (num_tiles > 1)
    {
        num_envs *= num_tiles;
        boards = boards.tile({num_tiles, 1});
        zero_boards = zero_boards.tile({num_tiles, 1});
        num_moves = num_moves.tile({num_tiles});
        num_moves_since_last_attack = num_moves_since_last_attack.tile({num_tiles});
        terminated_since = terminated_since.tile({num_tiles});
        has_legal_movement = has_legal_movement.tile({num_tiles});
        flag_captured = flag_captured.tile({num_tiles});
        action_history = action_history.tile({1, num_tiles});
        board_history = board_history.tile({1, num_tiles, 1});
        move_summary_history = move_summary_history.tile({1, num_tiles, 1});
        if (chase_state)
        {
            chase_state->last_dst_pos[0] = chase_state->last_dst_pos[0].tile({num_tiles});
            chase_state->last_dst_pos[1] = chase_state->last_dst_pos[1].tile({num_tiles});
            chase_state->last_src_pos[0] = chase_state->last_src_pos[0].tile({num_tiles});
            chase_state->last_src_pos[1] = chase_state->last_src_pos[1].tile({num_tiles});
            chase_state->chase_length[0] = chase_state->chase_length[0].tile({num_tiles});
            chase_state->chase_length[1] = chase_state->chase_length[1].tile({num_tiles});
        }
    }
}

EnvState EnvState::Clone() const
{
    auto _chase_state = chase_state;
    if (_chase_state)
    {
        _chase_state->last_dst_pos[0] = _chase_state->last_dst_pos[0].clone();
        _chase_state->last_dst_pos[1] = _chase_state->last_dst_pos[1].clone();
        _chase_state->last_src_pos[0] = _chase_state->last_src_pos[0].clone();
        _chase_state->last_src_pos[1] = _chase_state->last_src_pos[1].clone();
        _chase_state->chase_length[0] = _chase_state->chase_length[0].clone();
        _chase_state->chase_length[1] = _chase_state->chase_length[1].clone();
    }

    return EnvState{
        .num_envs = num_envs,
        .to_play = to_play,
        .boards = boards.clone(),
        .zero_boards = zero_boards.clone(),
        .num_moves = num_moves.clone(),
        .num_moves_since_last_attack = num_moves_since_last_attack.clone(),
        .terminated_since = terminated_since.clone(),
        .has_legal_movement = has_legal_movement.clone(),
        .flag_captured = flag_captured.clone(),
        .action_history = action_history.clone(),
        .board_history = board_history.clone(),
        .move_summary_history = move_summary_history.clone(),
        .chase_state = _chase_state,
    };
}

int32_t EnvState::CudaDevice() const
{
    MUSTRATEGO_CHECK(boards.device().is_cuda(), "The EnvState is not stored on a CUDA device");
    const int32_t device = boards.device().index();
    assert(device == zero_boards.device().index());
    assert(device == num_moves.device().index());
    assert(device == num_moves_since_last_attack.device().index());
    assert(device == flag_captured.device().index());
    assert(device == action_history.device().index());
    assert(device == board_history.device().index());
    assert(device == move_summary_history.device().index());
    if (chase_state)
    {
        assert(device == chase_state->last_dst_pos[0].device().index());
        assert(device == chase_state->last_dst_pos[1].device().index());
        assert(device == chase_state->last_src_pos[0].device().index());
        assert(device == chase_state->last_src_pos[1].device().index());
        assert(device == chase_state->chase_length[0].device().index());
        assert(device == chase_state->chase_length[1].device().index());
    }

    return device;
}

EnvState EnvState::Cat(const EnvState &other) const
{
    MUSTRATEGO_CHECK(to_play == other.to_play, "Acting player must match.");
    MUSTRATEGO_CHECK((!chase_state && !other.chase_state) || (chase_state && other.chase_state), "Both env states must have chase state or neither.");
    // TODO(gfarina): Check that the dimensions match.

    TensorChaseState chase_state;
    chase_state.last_dst_pos[0] = torch::cat({this->chase_state->last_dst_pos[0], other.chase_state->last_dst_pos[0]}, 0);
    chase_state.last_dst_pos[1] = torch::cat({this->chase_state->last_dst_pos[1], other.chase_state->last_dst_pos[1]}, 0);
    chase_state.last_src_pos[0] = torch::cat({this->chase_state->last_src_pos[0], other.chase_state->last_src_pos[0]}, 0);
    chase_state.last_src_pos[1] = torch::cat({this->chase_state->last_src_pos[1], other.chase_state->last_src_pos[1]}, 0);
    chase_state.chase_length[0] = torch::cat({this->chase_state->chase_length[0], other.chase_state->chase_length[0]}, 0);
    chase_state.chase_length[1] = torch::cat({this->chase_state->chase_length[1], other.chase_state->chase_length[1]}, 0);

    return EnvState{
        .num_envs = num_envs + other.num_envs,
        .to_play = to_play,
        .boards = torch::cat({boards, other.boards}, 0),
        .zero_boards = torch::cat({zero_boards, other.zero_boards}, 0),
        .num_moves = torch::cat({num_moves, other.num_moves}, 0),
        .num_moves_since_last_attack = torch::cat({num_moves_since_last_attack, other.num_moves_since_last_attack}, 0),
        .terminated_since = torch::cat({terminated_since, other.terminated_since}, 0),
        .has_legal_movement = torch::cat({has_legal_movement, other.has_legal_movement}, 0),
        .flag_captured = torch::cat({flag_captured, other.flag_captured}, 0),
        .action_history = torch::cat({action_history, other.action_history}, 1),
        .board_history = torch::cat({board_history, other.board_history}, 1),
        .move_summary_history = torch::cat({move_summary_history, other.move_summary_history}, 1),
        .chase_state = chase_state};
}

EnvState EnvState::Slice(uint32_t begin, uint32_t end) const {
    MUSTRATEGO_CHECK(0 <= begin && begin < end && end <= num_envs,
                     "Invalid slice range.");

    const torch::indexing::Slice s(begin, end); // [begin, end)

    std::optional<TensorChaseState> out_chase;
    if (chase_state) {
        TensorChaseState cs;
        for (int i = 0; i < 2; ++i) {
            cs.last_dst_pos[i] = chase_state->last_dst_pos[i].index({s});
            cs.last_src_pos[i] = chase_state->last_src_pos[i].index({s});
            cs.chase_length[i] = chase_state->chase_length[i].index({s});
        }
        out_chase = cs;
    }

    return EnvState{
        .num_envs = end - begin,
        .to_play = to_play,
        .boards = boards.index({s}),
        .zero_boards = zero_boards.index({s}),
        .num_moves = num_moves.index({s}),
        .num_moves_since_last_attack = num_moves_since_last_attack.index({s}),
        .terminated_since = terminated_since.index({s}),
        .has_legal_movement = has_legal_movement.index({s}),
        .flag_captured = flag_captured.index({s}),
        .action_history = action_history.index({torch::indexing::Slice(), s}),
        .board_history = board_history.index({torch::indexing::Slice(), s}),
        .move_summary_history = move_summary_history.index({torch::indexing::Slice(), s}),
        .chase_state = out_chase,
    };
}

EnvState AssignOpponentHiddenPieces(
    const EnvState &env_state,
    const torch::Tensor &opponent_hidden)
{
    MUSTRATEGO_CHECK(env_state.CudaDevice() == opponent_hidden.device().index(), "Opponent hidden must be on the same device as the env state.");
    MUSTRATEGO_CHECK(env_state.num_envs == 1, "The env state should only have one env");
    MUSTRATEGO_CHECK(opponent_hidden.dtype() == torch::kByte, "Opponent hidden must be of type uint8.");
    MUSTRATEGO_CHECK(opponent_hidden.dim() == 3, "Opponent hidden must be a 3D tensor.");
    MUSTRATEGO_CHECK(opponent_hidden.size(1) == 8 || opponent_hidden.size(1) == 40, "Second dimension can only be 8 or 40.");
    MUSTRATEGO_CHECK(opponent_hidden.size(2) == NUM_PIECE_TYPES, "Third dimension must be equal to NUM_PIECE_TYPES (=%d).", NUM_PIECE_TYPES);

    MUSTRATEGO_DEBUG("Assigning opponent hidden pieces (num_assignments: %d)...", opponent_hidden.size(0));

    const torch::Tensor assignments = opponent_hidden.argmax(-1);
    const uint8_t opponent = 2 - env_state.to_play;

    torch::Tensor hidden_ids = env_state.boards.index({0, torch::indexing::Slice(0, 1600, 16)}).flatten().clone();
    hidden_ids.bitwise_and_(0b01110000).eq_(opponent << 4);
    hidden_ids = hidden_ids.cpu();

    for (int i = 0, j = 0; i < 100; ++i)
    {
        uint8_t *hidden_id = hidden_ids.data_ptr<uint8_t>() + (opponent == 1 ? 99 - i : i);
        if (*hidden_id)
        {
            *hidden_id = ++j;
        }
    }
    const torch::Tensor piece_ids = env_state.boards.index({0, torch::indexing::Slice(1, 1600, 16)}).flatten().cpu();
    torch::Tensor id_to_hidden = torch::zeros(40, torch::kInt64);
    id_to_hidden.fill_(-1); // -1 means no hidden piece for that id
    for (int i = 0; i < 100; ++i)
    {
        const uint8_t hidden_id = hidden_ids[i].item<uint8_t>();
        if (hidden_id)
        {
            const uint8_t piece_id = piece_ids[i].item<uint8_t>();
            assert(piece_id < 40);
            id_to_hidden[piece_id] = hidden_id - 1;
        }
    }
    id_to_hidden = id_to_hidden.to(torch::Device(torch::kCUDA, env_state.CudaDevice()));
    const torch::Tensor masked_id = id_to_hidden.eq(-1);

    id_to_hidden.clamp_min_(0);
    torch::Tensor id_to_type = torch::gather(
        assignments,
        /* dim */ 1,
        id_to_hidden
            .unsqueeze(0)
            .expand({opponent_hidden.size(0), -1}));
    id_to_type.masked_fill_(
        masked_id.unsqueeze(0),
        -1);
    id_to_type = id_to_type.to(torch::kInt32);

    assert(id_to_type.dim() == 2);
    assert(id_to_type.size(0) == opponent_hidden.size(0));
    assert(id_to_type.size(1) == 40);

    auto clone = env_state.Clone();
    clone.Tile(opponent_hidden.size(0));

    {
        const uint32_t num_threads = 512;
        const uint32_t num_blocks = ceil(100 * clone.num_envs, num_threads);

        AssignBoardPiecesKernel<<<num_blocks, num_threads>>>(
            clone.boards.data_ptr<uint8_t>(),
            id_to_type.data_ptr<int32_t>(),
            1, // num_boards
            clone.num_envs,
            opponent);
        AssignBoardPiecesKernel<<<num_blocks, num_threads>>>(
            clone.zero_boards.data_ptr<uint8_t>(),
            id_to_type.data_ptr<int32_t>(),
            1, // num_boards
            clone.num_envs,
            opponent);
    }
    {
        const uint32_t move_memory = env_state.board_history.size(0);
        const uint32_t num_threads = 512;
        const uint32_t num_blocks = ceil(100ll * move_memory * clone.num_envs, num_threads);

        AssignBoardPiecesKernel<<<num_blocks, num_threads>>>(
            clone.board_history.data_ptr<uint8_t>(),
            id_to_type.data_ptr<int32_t>(),
            move_memory,
            clone.num_envs,
            opponent);
    }

    MUSTRATEGO_DEBUG("...done");
    return clone;
}

EnvState LegacyAssignOpponentHiddenPieces(
    const EnvState &env_state,
    const torch::Tensor &opponent_hidden)
{
    MUSTRATEGO_CHECK(env_state.CudaDevice() == opponent_hidden.device().index(), "Opponent hidden must be on the same device as the env state.");
    MUSTRATEGO_CHECK(env_state.num_envs == 1, "The env state should only have one env");
    MUSTRATEGO_CHECK(opponent_hidden.dtype() == torch::kByte, "Opponent hidden must be of type uint8.");
    MUSTRATEGO_CHECK(opponent_hidden.dim() == 3, "Opponent hidden must be a 3D tensor.");
    MUSTRATEGO_CHECK(opponent_hidden.size(1) == 8 || opponent_hidden.size(1) == 40, "Second dimension can only be 8 or 40.");
    MUSTRATEGO_CHECK(opponent_hidden.size(2) == NUM_PIECE_TYPES, "Third dimension must be equal to NUM_PIECE_TYPES (=%d).", NUM_PIECE_TYPES);

    const torch::Tensor assignments = opponent_hidden.argmax(-1).cpu();
    torch::Tensor out_boards = env_state.boards.cpu();

    // This makes a copy
    StrategoBoard original_board;
    memcpy((void *)&original_board, (void *)out_boards.data_ptr<uint8_t>(), sizeof(StrategoBoard));

    out_boards = out_boards.tile({assignments.size(0), 1});
    torch::Tensor out_zero_boards = env_state.zero_boards.tile({assignments.size(0), 1}).cpu();
    torch::Tensor out_board_history = env_state.board_history.tile({1, assignments.size(0), 1}).cpu();
    const uint32_t move_memory = out_board_history.size(0);

    const uint8_t opponent = 2 - env_state.to_play;

    const auto rewrite_board = [opponent](uint8_t *const ptr, const std::vector<uint8_t> &assignment)
    {
        for (int cell = 0; cell < 100; ++cell)
        {
            const uint8_t piece = ptr[16 * cell];
            const uint8_t piece_id = ptr[16 * cell + 1];
            if ((piece & 0b01110000) == (opponent << 4))
            {
                assert(piece_id < 100);

                // When processing past boards, there might be more unknown pieces
                // than at current time. In that case, we do not overwrite the type.
                if (assignment.at(piece_id) != 0xff)
                {
                    assert(assignment.at(piece_id) < LAKE);

                    ptr[16 * cell] &= 0b11110000;
                    ptr[16 * cell] |= assignment.at(piece_id);
                }
            }
        }
    };

#pragma omp parallel for
    for (int i = 0; i < assignments.size(0); ++i)
    {
        std::vector<uint8_t> assignment(100, 0xff); // piece_id -> PieceType

        int hidden_index = 0;
        for (int j = 0; j < 100; ++j)
        {
            MUSTRATEGO_CHECK(hidden_index < assignments.size(1), "Dimension 1 of assignments tensor is too small");
            const int cell = (opponent == 1) ? 99 - j : j;
            const Piece &piece = original_board.pieces[cell / 10][cell % 10];
            if (!piece.visible && piece.color == opponent)
            {
                const uint8_t assign_piece_type = assignments.index({i, hidden_index}).item<uint8_t>();
                MUSTRATEGO_CHECK(assign_piece_type < LAKE, "Invalid piece type (%d) for cell %d", assign_piece_type, cell);
                MUSTRATEGO_CHECK(assign_piece_type < FLAG /* piece type is movable */ || !piece.has_moved,
                                 "[env %d] Tried to assign an unmovable piece to a cell (%d) that has moved", i, cell);
                assert(piece.piece_id < 100);
                assignment.at(piece.piece_id) = assign_piece_type;
                ++hidden_index;
            }
        }
        for (int j = hidden_index; j < assignments.size(1); ++j)
        {
            MUSTRATEGO_CHECK(assignments.index({i, j}).item<uint8_t>() == 0,
                             "Assignment past num hidden pieces (%d)", hidden_index);
        }

        for (uint8_t pt = SPY; pt < LAKE; ++pt)
        {
            int assigned_cnt = 0;
            for (const uint8_t t : assignment)
            {
                if (t == pt)
                    ++assigned_cnt;
            }
            MUSTRATEGO_CHECK(assigned_cnt == original_board.num_hidden[opponent - 1][pt],
                             "Mismatch between assignment count (%d) and board count (%d) for piece type %d",
                             assigned_cnt, original_board.num_hidden[opponent - 1][pt], pt);
        }

        assert(out_boards.numel() == assignments.size(0) * sizeof(StrategoBoard));
        assert(out_zero_boards.numel() == assignments.size(0) * sizeof(StrategoBoard));
        assert(out_board_history.numel() == move_memory * assignments.size(0) * sizeof(StrategoBoard));

        rewrite_board(out_boards.data_ptr<uint8_t>() + i * sizeof(StrategoBoard), assignment);
        rewrite_board(out_zero_boards.data_ptr<uint8_t>() + i * sizeof(StrategoBoard), assignment);
        for (int j = 0; j < move_memory; ++j)
        {
            rewrite_board(out_board_history.data_ptr<uint8_t>() + j * assignments.size(0) * sizeof(StrategoBoard) + i * sizeof(StrategoBoard), assignment);
        }
    }

    EnvState out{
        .num_envs = assignments.size(0),
        .to_play = env_state.to_play,
        .boards = out_boards.to(torch::Device(torch::kCUDA, env_state.CudaDevice())),
        .zero_boards = out_zero_boards.to(torch::Device(torch::kCUDA, env_state.CudaDevice())),
        .num_moves = env_state.num_moves.tile({assignments.size(0)}),
        .num_moves_since_last_attack = env_state.num_moves_since_last_attack.tile({assignments.size(0)}),
        .terminated_since = env_state.terminated_since.tile({assignments.size(0)}),
        .has_legal_movement = env_state.has_legal_movement.tile({assignments.size(0)}),
        .flag_captured = env_state.flag_captured.tile({assignments.size(0)}),
        .action_history = env_state.action_history.tile({1, assignments.size(0)}),
        .board_history = out_board_history.to(torch::Device(torch::kCUDA, env_state.CudaDevice())),
        .move_summary_history = env_state.move_summary_history.tile({1, assignments.size(0), 1}),
    };
    if (env_state.chase_state)
    {
        out.chase_state.emplace();
        out.chase_state->last_dst_pos[0] = env_state.chase_state->last_dst_pos[0].tile({assignments.size(0)});
        out.chase_state->last_dst_pos[1] = env_state.chase_state->last_dst_pos[1].tile({assignments.size(0)});
        out.chase_state->last_src_pos[0] = env_state.chase_state->last_src_pos[0].tile({assignments.size(0)});
        out.chase_state->last_src_pos[1] = env_state.chase_state->last_src_pos[1].tile({assignments.size(0)});
        out.chase_state->chase_length[0] = env_state.chase_state->chase_length[0].tile({assignments.size(0)});
        out.chase_state->chase_length[1] = env_state.chase_state->chase_length[1].tile({assignments.size(0)});
    }
    return out;
}