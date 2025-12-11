#include "src/env/cuda/kernels.h"
#include "src/env/stratego.h"
#include "src/env/stratego_board.h"

__global__ void LegalActionsMaskKernel(
    bool *d_out,
    const uint8_t *d_terminated_since,
    const uint32_t num_envs,
    const uint8_t for_player,
    const StrategoBoard *d_boards,
    const bool handle_terminated)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t env_idx = index / 100;

    if (env_idx >= num_envs)
        return;

    if (handle_terminated && d_terminated_since[env_idx])
    {
        // We make the first action playable even if the environment has terminated
        //
        // (Any action will be ignored by ApplyActions anyway.)
        d_out[env_idx * NUM_ACTIONS] = true;
        return;
    }

    const int32_t cell_idx = index % 100;
    const int32_t row_idx = cell_idx / 10;
    const int32_t col_idx = cell_idx % 10;

    const StrategoBoard *board = d_boards + env_idx;
    const Piece piece = board->pieces[row_idx][col_idx];
    if (piece.type >= FLAG || piece.color != for_player)
        return;

    const int32_t pov_row_idx = (for_player == 2) ? 9 - row_idx : row_idx;
    const int32_t pov_col_idx = (for_player == 2) ? 9 - col_idx : col_idx;
    const int32_t pov_cell_idx = (for_player == 2) ? 99 - cell_idx : cell_idx;
    bool *out = d_out + env_idx * NUM_ACTIONS;

    int8_t left = col_idx - 1;
    int8_t right = col_idx + 1;
    int8_t up = row_idx + 1;
    int8_t down = row_idx - 1;
    {
        // At this stage, we operate in absolute (non-relative) coordinates.
        // We will reverse this array later if the player is blue.
        while (piece.type == SCOUT && left > 0 && !board->pieces[row_idx][left].color)
            --left;
        // At this point we know that left is the right column to the left of col_idx
        // with a nonempty cell.
        //
        // If the current piece belongs to us or is a lake, then we cannot move up to left and
        // we must add 1.
        left += !!(left < 0 || board->pieces[row_idx][left].color & for_player);

        while (piece.type == SCOUT && right < 9 && !board->pieces[row_idx][right].color)
            ++right;
        right -= !!(right > 9 || board->pieces[row_idx][right].color & for_player);

        while (piece.type == SCOUT && up < 9 && !board->pieces[up][col_idx].color)
            ++up;
        up -= !!(up > 9 || board->pieces[up][col_idx].color & for_player);

        while (piece.type == SCOUT && down > 0 && !board->pieces[down][col_idx].color)
            --down;
        down += !!(down < 0 || board->pieces[down][col_idx].color & for_player);

        int8_t temp;
        if (for_player == 2)
        {
            // Swap left and right, and up and down.
            temp = left;
            left = 9 - right;
            right = 9 - temp;
            temp = down;
            down = 9 - up;
            up = 9 - temp;
        }
    }

    assert(pov_cell_idx >= 0 && pov_cell_idx < 100);

    // Vertical
    out[pov_cell_idx] = (down <= 0 + (pov_row_idx <= 0) && up >= 0 + (pov_row_idx <= 0));
    out[100 + pov_cell_idx] = (down <= 1 + (pov_row_idx <= 1) && up >= 1 + (pov_row_idx <= 1));
    out[200 + pov_cell_idx] = (down <= 2 + (pov_row_idx <= 2) && up >= 2 + (pov_row_idx <= 2));
    out[300 + pov_cell_idx] = (down <= 3 + (pov_row_idx <= 3) && up >= 3 + (pov_row_idx <= 3));
    out[400 + pov_cell_idx] = (down <= 4 + (pov_row_idx <= 4) && up >= 4 + (pov_row_idx <= 4));
    out[500 + pov_cell_idx] = (down <= 5 + (pov_row_idx <= 5) && up >= 5 + (pov_row_idx <= 5));
    out[600 + pov_cell_idx] = (down <= 6 + (pov_row_idx <= 6) && up >= 6 + (pov_row_idx <= 6));
    out[700 + pov_cell_idx] = (down <= 7 + (pov_row_idx <= 7) && up >= 7 + (pov_row_idx <= 7));
    out[800 + pov_cell_idx] = (down <= 8 + (pov_row_idx <= 8) && up >= 8 + (pov_row_idx <= 8));
    // Horizontal
    out[900 + pov_cell_idx] = (left <= 0 + (pov_col_idx <= 0) && right >= 0 + (pov_col_idx <= 0));
    out[1000 + pov_cell_idx] = (left <= 1 + (pov_col_idx <= 1) && right >= 1 + (pov_col_idx <= 1));
    out[1100 + pov_cell_idx] = (left <= 2 + (pov_col_idx <= 2) && right >= 2 + (pov_col_idx <= 2));
    out[1200 + pov_cell_idx] = (left <= 3 + (pov_col_idx <= 3) && right >= 3 + (pov_col_idx <= 3));
    out[1300 + pov_cell_idx] = (left <= 4 + (pov_col_idx <= 4) && right >= 4 + (pov_col_idx <= 4));
    out[1400 + pov_cell_idx] = (left <= 5 + (pov_col_idx <= 5) && right >= 5 + (pov_col_idx <= 5));
    out[1500 + pov_cell_idx] = (left <= 6 + (pov_col_idx <= 6) && right >= 6 + (pov_col_idx <= 6));
    out[1600 + pov_cell_idx] = (left <= 7 + (pov_col_idx <= 7) && right >= 7 + (pov_col_idx <= 7));
    out[1700 + pov_cell_idx] = (left <= 8 + (pov_col_idx <= 8) && right >= 8 + (pov_col_idx <= 8));
}

__global__ void ApplyActionsKernel(
    StrategoBoard *d_boards,
    int32_t *d_num_moves,
    int32_t *d_num_moves_since_last_attack,
    int32_t *d_num_moves_since_reset,
    uint8_t *d_flag_captured,
    uint8_t *d_move_summary_out,
    uint8_t *d_red_death,  // Cell where  red dies or 0xff
    uint8_t *d_blue_death, // Cell where blue dies or 0xff
    const uint8_t *d_terminated_since,
    const uint32_t num_envs,
    const uint8_t player, // the acting player
    const int32_t *d_actions)
{
    int32_t env_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_idx >= num_envs)
        return;

    ++d_num_moves[env_idx];
    ++d_num_moves_since_last_attack[env_idx];
    ++d_num_moves_since_reset[env_idx];

    const int32_t action = d_actions[env_idx];
    const int32_t from_cell_idx = action % 100;
    const int32_t pov_from_row_idx = from_cell_idx / 10;
    const int32_t pov_from_col_idx = from_cell_idx % 10;

    int32_t new_coord = action / 100;
    const int32_t direction = new_coord >= 9; // 0 = vertical, 1 = horizontal
    new_coord %= 9;
    new_coord +=
        (direction && new_coord >= pov_from_col_idx) ||
        (!direction && new_coord >= pov_from_row_idx);
    assert(0 <= new_coord && new_coord <= 9);
    const int32_t pov_to_row_idx = direction ? pov_from_row_idx : new_coord;
    const int32_t pov_to_col_idx = direction ? new_coord : pov_from_col_idx;

    const int32_t from_row_idx = (player == 2) ? 9 - pov_from_row_idx : pov_from_row_idx;
    const int32_t from_col_idx = (player == 2) ? 9 - pov_from_col_idx : pov_from_col_idx;
    const int32_t to_row_idx = (player == 2) ? 9 - pov_to_row_idx : pov_to_row_idx;
    const int32_t to_col_idx = (player == 2) ? 9 - pov_to_col_idx : pov_to_col_idx;

    assert(to_row_idx >= 0 && to_row_idx <= 9);
    assert(to_col_idx >= 0 && to_col_idx <= 9);

    Piece from_piece = d_boards[env_idx].pieces[from_row_idx][from_col_idx];
    Piece to_piece = d_boards[env_idx].pieces[to_row_idx][to_col_idx];
    const bool to_piece_was_visible = to_piece.visible;

    assert(from_piece.piece_id == 0xff || !d_boards[env_idx].death_status[from_piece.color - 1][from_piece.piece_id].death_location);
    assert(to_piece.piece_id == 0xff || !d_boards[env_idx].death_status[to_piece.color - 1][to_piece.piece_id].death_location);

    d_move_summary_out[6 * env_idx + 0] = from_cell_idx;
    d_move_summary_out[6 * env_idx + 1] = 10 * pov_to_row_idx + pov_to_col_idx;
    d_move_summary_out[6 * env_idx + 2] = ((to_piece.type == EMPTY && !from_piece.visible) ? HIDDEN_PIECE : from_piece.type) + (from_piece.visible << 4) + (from_piece.has_moved << 5);
    d_move_summary_out[6 * env_idx + 3] = to_piece.type + (to_piece.visible << 4) + (to_piece.has_moved << 5);
    d_move_summary_out[6 * env_idx + 4] = from_piece.piece_id;
    d_move_summary_out[6 * env_idx + 5] = to_piece.piece_id;

    if (d_terminated_since[env_idx])
        return;

    // Update `actively_adjacent`
    const uint8_t last_moved_piece_type = d_boards[env_idx].last_moved_piece_type;
    const uint8_t prev_dst_abs = d_boards[env_idx].prev_dst_abs;
    const uint8_t prev_prev_dst_abs = d_boards[env_idx].prev_prev_dst_abs;
    if (last_moved_piece_type != 0xff && prev_dst_abs != 0xff)
    {
        assert(last_moved_piece_type < 16);
        assert(prev_dst_abs < 100);

#define UPDATE_ACT_ADJ(r, c)                                                                      \
    {                                                                                             \
        Piece p = d_boards[env_idx].pieces[r][c];                                                 \
        if (p.color == player)                                                                    \
        {                                                                                         \
            assert(last_moved_piece_type < 16);                                                   \
            p.actively_adjacent[last_moved_piece_type / 8] |= (1 << (last_moved_piece_type % 8)); \
            d_boards[env_idx].pieces[r][c] = p;                                                   \
        }                                                                                         \
    }
        if (prev_dst_abs > 9)
            UPDATE_ACT_ADJ(prev_dst_abs / 10 - 1, prev_dst_abs % 10);
        if (prev_dst_abs < 90)
            UPDATE_ACT_ADJ(prev_dst_abs / 10 + 1, prev_dst_abs % 10);
        if (prev_dst_abs % 10 > 0)
            UPDATE_ACT_ADJ(prev_dst_abs / 10, prev_dst_abs % 10 - 1);
        if (prev_dst_abs % 10 < 9)
            UPDATE_ACT_ADJ(prev_dst_abs / 10, prev_dst_abs % 10 + 1);

#undef UPDATE_ACT_ADJ
    }
    if (prev_prev_dst_abs != 0xff)
    {
        assert(prev_prev_dst_abs < 100);
        const Piece prev_prev_dst_piece = d_boards[env_idx].pieces[prev_prev_dst_abs / 10][prev_prev_dst_abs % 10];

        // First case: our piece is still alieve, and is now actively adjacent to its neighbors.
        if (prev_prev_dst_piece.color == player)
        {
#define UPDATE_ACT_ADJ(r, c)                                                                                                       \
    {                                                                                                                              \
        Piece p = d_boards[env_idx].pieces[r][c];                                                                                  \
        if (p.color == 3 - player)                                                                                                 \
        {                                                                                                                          \
            const uint8_t pt = p.visible ? p.type : HIDDEN_PIECE;                                                                  \
            assert(prev_prev_dst_abs < 100);                                                                                       \
            assert(pt < 16);                                                                                                       \
            d_boards[env_idx].pieces[prev_prev_dst_abs / 10][prev_prev_dst_abs % 10].actively_adjacent[pt / 8] |= (1 << (pt % 8)); \
        }                                                                                                                          \
    }

            if (prev_prev_dst_abs > 9)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10 - 1, prev_prev_dst_abs % 10);
            if (prev_prev_dst_abs < 90)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10 + 1, prev_prev_dst_abs % 10);
            if (prev_prev_dst_abs % 10 > 0)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10, prev_prev_dst_abs % 10 - 1);
            if (prev_prev_dst_abs % 10 < 9)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10, prev_prev_dst_abs % 10 + 1);

#undef UPDATE_ACT_ADJ
        }
        // Second case: the piece is dead, perhaps killed by the piece currently in that cell. We need to update
        // active adjacency for all nearby pieces
        else if (prev_prev_dst_piece.color == 3 - player && prev_prev_dst_piece.visible)
        {
            const uint8_t pt = prev_prev_dst_piece.type;
            assert(pt < 16);

#define UPDATE_ACT_ADJ(r, c)                                                             \
    {                                                                                    \
        if (d_boards[env_idx].pieces[r][c].color == player)                              \
            d_boards[env_idx].pieces[r][c].actively_adjacent[pt / 8] |= (1 << (pt % 8)); \
    }

            if (prev_prev_dst_abs > 9)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10 - 1, prev_prev_dst_abs % 10);
            if (prev_prev_dst_abs < 90)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10 + 1, prev_prev_dst_abs % 10);
            if (prev_prev_dst_abs % 10 > 0)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10, prev_prev_dst_abs % 10 - 1);
            if (prev_prev_dst_abs % 10 < 9)
                UPDATE_ACT_ADJ(prev_prev_dst_abs / 10, prev_prev_dst_abs % 10 + 1);

#undef UPDATE_ACT_ADJ
        }
    }

    // Refresh pieces as the previous code might have modified them.
    from_piece = d_boards[env_idx].pieces[from_row_idx][from_col_idx];
    to_piece = d_boards[env_idx].pieces[to_row_idx][to_col_idx];

    // Update `evaded`
    if (last_moved_piece_type != 0xff &&
        10 * to_row_idx + to_col_idx != prev_dst_abs &&
        IS_ADJACENT((10 * from_row_idx + from_col_idx), prev_dst_abs))
    {
        assert(prev_dst_abs < 100);
        assert(last_moved_piece_type < FLAG || last_moved_piece_type == HIDDEN_PIECE);
        from_piece.evaded[last_moved_piece_type / 8] |= (1 << (last_moved_piece_type % 8));
    }

    assert(from_piece.color && from_piece.color < 3);
    assert(from_piece.type < FLAG);

    if (!from_piece.has_moved && !from_piece.visible)
    {
        --d_boards[env_idx].num_hidden_unmoved[player - 1];
    }
    from_piece.has_moved = true;

    // If we move to a place of the opponent, we both become visible
    if (to_piece.color ^ player == 3)
    {
        // This was an attack, so reset the counter.
        d_num_moves_since_last_attack[env_idx] = 0;

        if (!from_piece.visible)
        {
            assert(from_piece.type < 12);
            assert(player == 1 || player == 2);
            --d_boards[env_idx].num_hidden[player - 1][from_piece.type];
            from_piece.visible = true;
        }
        if (!to_piece.visible)
        {
            assert(to_piece.type < 12);
            assert(to_piece.color == 1 || to_piece.color == 2);
            --d_boards[env_idx].num_hidden[to_piece.color - 1][to_piece.type];
            to_piece.visible = true;

            if (!to_piece.has_moved)
            {
                --d_boards[env_idx].num_hidden_unmoved[to_piece.color - 1];
            }
        }
        assert(from_piece.visible && to_piece.visible);
    }

    const int32_t step_length = abs(to_row_idx - from_row_idx) + abs(to_col_idx - from_col_idx);
    assert(abs(step_length) == 1 || from_piece.type == SCOUT);

    // If the piece moves by more than 1 unit, then we know it is a scout, so we might as well
    // mark it as visible!
    if (abs(step_length) >= 2 && !from_piece.visible)
    {
        --d_boards[env_idx].num_hidden[player - 1][from_piece.type];
        from_piece.visible = true;
    }

    // Empty the source cell.
    d_boards[env_idx].pieces[from_row_idx][from_col_idx] = _E(EMPTY);

    // The destination cell is:
    // * from_piece, if either:
    //   - the destination is empty
    //   - the source is a miner (2) and the destination is a bomb (11)
    //   - the source is a spy (0) and the destination is a marshal (9)
    //   - the source type is > the destination type.
    //   - the destination is a FLAG
    // * empty if
    //   - the source matches the destination
    // * to_piece if
    //   - the destination is a bomb and the source not a miner
    //   - the destination piece has value > the source piece
    const bool to_wins =
        (to_piece.type < FLAG &&
         to_piece.type > from_piece.type &&
         !(to_piece.type == MARSHAL && from_piece.type == SPY)) ||
        (to_piece.type == BOMB && from_piece.type != MINER);
    const bool tie = (to_piece.type == from_piece.type);

    // If to_wins, nothing more happens.
    Piece xpiece;
    if (to_wins)
    {
        const uint8_t from_piece_id = from_piece.piece_id;
        assert(from_piece_id < 40);
        assert(from_piece.color == player);

        d_boards[env_idx].deaths[from_piece.color - 1][from_piece_id / 8] |= (1 << (from_piece_id % 8));
        xpiece = d_boards[env_idx].pieces[to_row_idx][to_col_idx] = to_piece;

        // Mark the from_piece as dead.
        DeathStatus ds = d_boards[env_idx].death_status[from_piece.color - 1][from_piece_id];
        assert(!ds.is_dead);
        ds.is_dead = true;
        ds.death_reason = to_piece_was_visible ? ATTACKED_VISIBLE_STRONGER : ATTACKED_HIDDEN;
        ds.piece_type = from_piece.type;
        ds.death_location = to_row_idx * 10 + to_col_idx;
        d_boards[env_idx].death_status[from_piece.color - 1][from_piece_id] = ds;
    }
    else if (tie)
    {
        uint8_t from_piece_id = from_piece.piece_id;
        uint8_t to_piece_id = to_piece.piece_id;
        assert(from_piece_id < 40);
        assert(to_piece_id < 40);
        assert(from_piece.color == player);
        assert(to_piece.color == 3 - player);

        d_boards[env_idx].deaths[from_piece.color - 1][from_piece_id / 8] |= (1 << (from_piece_id % 8));
        d_boards[env_idx].deaths[to_piece.color - 1][to_piece_id / 8] |= (1 << (to_piece_id % 8));

        xpiece = d_boards[env_idx].pieces[to_row_idx][to_col_idx] = _E(EMPTY);

        // Mark the from_piece as dead.
        {
            DeathStatus ds = d_boards[env_idx].death_status[from_piece.color - 1][from_piece_id];
            assert(!ds.is_dead);
            ds.is_dead = true;
            ds.death_reason = to_piece_was_visible ? ATTACKED_VISIBLE_TIE : ATTACKED_HIDDEN;
            ds.piece_type = from_piece.type;
            ds.death_location = to_row_idx * 10 + to_col_idx;
            d_boards[env_idx].death_status[from_piece.color - 1][from_piece_id] = ds;
        }
        // Mark the to_piece as dead.
        {
            assert(to_piece.color == 3 - from_piece.color);
            DeathStatus ds = d_boards[env_idx].death_status[to_piece.color - 1][to_piece_id];
            assert(!ds.is_dead);
            ds.is_dead = true;
            ds.death_reason = to_piece_was_visible ? VISIBLE_DEFENDED_TIE : HIDDEN_DEFENDED;
            ds.piece_type = to_piece.type;
            ds.death_location = to_row_idx * 10 + to_col_idx;
            d_boards[env_idx].death_status[to_piece.color - 1][to_piece_id] = ds;
        }
    }
    else
    {
        if (to_piece.type != EMPTY)
        {
            uint8_t to_piece_id = to_piece.piece_id;
            assert(to_piece_id < 40);
            assert(to_piece.color == 3 - player);

            d_boards[env_idx].deaths[to_piece.color - 1][to_piece_id / 8] |= (1 << (to_piece_id % 8));

            // Mark the to_piece as dead.
            {
                assert(to_piece.color == 3 - from_piece.color);
                DeathStatus ds = d_boards[env_idx].death_status[to_piece.color - 1][to_piece_id];
                assert(!ds.is_dead);
                ds.is_dead = true;
                ds.death_reason = to_piece_was_visible ? VISIBLE_DEFENDED_WEAKER : HIDDEN_DEFENDED;
                ds.piece_type = to_piece.type;
                ds.death_location = to_row_idx * 10 + to_col_idx;
                d_boards[env_idx].death_status[to_piece.color - 1][to_piece_id] = ds;
            }
        }
        else
        {
#define UPDATE_THREAT(r, c)                                       \
    {                                                             \
        const Piece p = d_boards[env_idx].pieces[r][c];           \
        if (p.color == 3 - player)                                \
        {                                                         \
            const uint8_t pt = p.visible ? p.type : HIDDEN_PIECE; \
            assert(pt < 16);                                      \
            from_piece.threatened[pt / 8] |= (1 << (pt % 8));     \
        }                                                         \
    }

            // Update `threatened`
            // The moving piece is still alive. We mark all the pieces adjacent to the
            // destination as being threatened.
            if (to_row_idx > 0)
                UPDATE_THREAT(to_row_idx - 1, to_col_idx);
            if (to_row_idx < 9)
                UPDATE_THREAT(to_row_idx + 1, to_col_idx);
            if (to_col_idx > 0)
                UPDATE_THREAT(to_row_idx, to_col_idx - 1);
            if (to_col_idx < 9)
                UPDATE_THREAT(to_row_idx, to_col_idx + 1);
#undef UPDATE_THREAT
        }

        xpiece = d_boards[env_idx].pieces[to_row_idx][to_col_idx] = from_piece;
    }

#define UPDATE_PROTECT(a, b, c)                                                                \
    {                                                                                          \
        assert((a) >= 0 && (b) >= 0 && (c) >= 0);                                              \
        assert((a) < 100 && (b) < 100 && (c) < 100 && (a) != (b) && (a) != (c) && (b) != (c)); \
        Piece *protector = &d_boards[env_idx].pieces[(a) / 10][(a) % 10];                      \
        Piece *protectee = &d_boards[env_idx].pieces[(b) / 10][(b) % 10];                      \
        Piece *aggressor = &d_boards[env_idx].pieces[(c) / 10][(c) % 10];                      \
        if ((aggressor->color == 3 - player) &&                                                \
            (protectee->color == player || protectee->color == 0) &&                           \
            (protector->color == player))                                                      \
        {                                                                                      \
            const uint8_t protector_pt = protector->visible ? protector->type : HIDDEN_PIECE;  \
            const uint8_t protectee_pt = protectee->visible ? protectee->type : HIDDEN_PIECE;  \
            const uint8_t aggressor_pt = aggressor->visible ? aggressor->type : HIDDEN_PIECE;  \
            assert(protector_pt < 16 && protectee_pt < 16 && aggressor_pt < 16);               \
            protector->protected_[protectee_pt / 8] |= (1 << (protectee_pt % 8));              \
            protector->protected_against[aggressor_pt / 8] |= (1 << (aggressor_pt % 8));       \
            protectee->was_protected_by[protector_pt / 8] |= (1 << (protector_pt % 8));        \
            protectee->was_protected_against[aggressor_pt / 8] |= (1 << (aggressor_pt % 8));   \
        }                                                                                      \
    }
    // Case 1: Previously moved piece is aggressor
    if (last_moved_piece_type != 0xff)
    {
        assert(prev_dst_abs < 100);

        // Two down
        if ((prev_dst_abs / 10) >= 2)
        {
            UPDATE_PROTECT(prev_dst_abs - 20, prev_dst_abs - 10, prev_dst_abs)
        }
        // Two up
        if ((prev_dst_abs / 10) < 8)
        {
            UPDATE_PROTECT(prev_dst_abs + 20, prev_dst_abs + 10, prev_dst_abs)
        }
        // Two left
        if ((prev_dst_abs % 10) >= 2)
        {
            UPDATE_PROTECT(prev_dst_abs - 2, prev_dst_abs - 1, prev_dst_abs)
        }
        // Two right
        if ((prev_dst_abs % 10) < 8)
        {
            UPDATE_PROTECT(prev_dst_abs + 2, prev_dst_abs + 1, prev_dst_abs)
        }
        // One up one left
        if ((prev_dst_abs / 10) < 9 && (prev_dst_abs % 10) >= 1)
        {
            UPDATE_PROTECT(prev_dst_abs + 9, prev_dst_abs - 1, prev_dst_abs)
            UPDATE_PROTECT(prev_dst_abs + 9, prev_dst_abs + 10, prev_dst_abs)
        }
        // One up one right
        if ((prev_dst_abs / 10) < 9 && (prev_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(prev_dst_abs + 11, prev_dst_abs + 1, prev_dst_abs)
            UPDATE_PROTECT(prev_dst_abs + 11, prev_dst_abs + 10, prev_dst_abs)
        }
        // One down one left
        if ((prev_dst_abs / 10) > 0 && (prev_dst_abs % 10) >= 1)
        {
            UPDATE_PROTECT(prev_dst_abs - 11, prev_dst_abs - 1, prev_dst_abs)
            UPDATE_PROTECT(prev_dst_abs - 11, prev_dst_abs - 10, prev_dst_abs)
        }
        // One down one right
        if ((prev_dst_abs / 10) > 0 && (prev_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(prev_dst_abs - 9, prev_dst_abs + 1, prev_dst_abs)
            UPDATE_PROTECT(prev_dst_abs - 9, prev_dst_abs - 10, prev_dst_abs)
        }
    }
    // Case 2: Moving piece is protector
    if (!(to_wins || tie))
    {
        const uint8_t to_dst_abs = 10 * to_row_idx + to_col_idx;
        // Two down
        if ((to_dst_abs / 10) >= 2)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs - 10, to_dst_abs - 20)
        }
        // Two up
        if ((to_dst_abs / 10) < 8)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs + 10, to_dst_abs + 20)
        }
        // Two left
        if ((to_dst_abs % 10) >= 2)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs - 1, to_dst_abs - 2)
        }
        // Two right
        if ((to_dst_abs % 10) < 8)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs + 1, to_dst_abs + 2)
        }
        // One up one left
        if ((to_dst_abs / 10) < 9 && (to_dst_abs % 10) >= 1)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs - 1, to_dst_abs + 9)
            UPDATE_PROTECT(to_dst_abs, to_dst_abs + 10, to_dst_abs + 9)
        }
        // One up one right
        if ((to_dst_abs / 10) < 9 && (to_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs + 1, to_dst_abs + 11)
            UPDATE_PROTECT(to_dst_abs, to_dst_abs + 10, to_dst_abs + 11)
        }
        // One down one left
        if ((to_dst_abs / 10) > 0 && (to_dst_abs % 10) >= 1)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs - 1, to_dst_abs - 11)
            UPDATE_PROTECT(to_dst_abs, to_dst_abs - 10, to_dst_abs - 11)
        }
        // One down one right
        if ((to_dst_abs / 10) > 0 && (to_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(to_dst_abs, to_dst_abs + 1, to_dst_abs - 9)
            UPDATE_PROTECT(to_dst_abs, to_dst_abs - 10, to_dst_abs - 9)
        }
    }
    // Case 3: Moving piece is protectee OR empty square is protectee via tie
    if (!(to_wins))
    {
        const uint8_t to_dst_abs = 10 * to_row_idx + to_col_idx;
        // One up other down
        if ((to_dst_abs / 10) < 9 && (to_dst_abs / 10) > 0)
        {
            UPDATE_PROTECT(to_dst_abs + 10, to_dst_abs, to_dst_abs - 10)
            UPDATE_PROTECT(to_dst_abs - 10, to_dst_abs, to_dst_abs + 10)
        }
        // One up other left
        if ((to_dst_abs / 10) < 9 && (to_dst_abs % 10) > 0)
        {
            UPDATE_PROTECT(to_dst_abs + 10, to_dst_abs, to_dst_abs - 1)
            UPDATE_PROTECT(to_dst_abs - 1, to_dst_abs, to_dst_abs + 10)
        }
        // One up other right
        if ((to_dst_abs / 10) < 9 && (to_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(to_dst_abs + 10, to_dst_abs, to_dst_abs + 1)
            UPDATE_PROTECT(to_dst_abs + 1, to_dst_abs, to_dst_abs + 10)
        }
        // One down other left
        if ((to_dst_abs / 10) > 0 && (to_dst_abs % 10) > 0)
        {
            UPDATE_PROTECT(to_dst_abs - 10, to_dst_abs, to_dst_abs - 1)
            UPDATE_PROTECT(to_dst_abs - 1, to_dst_abs, to_dst_abs - 10)
        }
        // One down other right
        if ((to_dst_abs / 10) > 0 && (to_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(to_dst_abs - 10, to_dst_abs, to_dst_abs + 1)
            UPDATE_PROTECT(to_dst_abs + 1, to_dst_abs, to_dst_abs - 10)
        }
        // One left other right
        if ((to_dst_abs % 10) < 9 && (to_dst_abs % 10) > 0)
        {
            UPDATE_PROTECT(to_dst_abs - 1, to_dst_abs, to_dst_abs + 1)
            UPDATE_PROTECT(to_dst_abs + 1, to_dst_abs, to_dst_abs - 1)
        }
    }
    // Case 4: Protection against newly revealed defender
    if (to_wins)
    {
        const uint8_t to_dst_abs = 10 * to_row_idx + to_col_idx;
        // Two down
        if ((to_dst_abs / 10) >= 2)
        {
            UPDATE_PROTECT(to_dst_abs - 20, to_dst_abs - 10, to_dst_abs)
        }
        // Two up
        if ((to_dst_abs / 10) < 8)
        {
            UPDATE_PROTECT(to_dst_abs + 20, to_dst_abs + 10, to_dst_abs)
        }
        // Two left
        if ((to_dst_abs % 10) >= 2)
        {
            UPDATE_PROTECT(to_dst_abs - 2, to_dst_abs - 1, to_dst_abs)
        }
        // Two right
        if ((to_dst_abs % 10) < 8)
        {
            UPDATE_PROTECT(to_dst_abs + 2, to_dst_abs + 1, to_dst_abs)
        }
        // One up one left
        if ((to_dst_abs / 10) < 9 && (to_dst_abs % 10) >= 1)
        {
            UPDATE_PROTECT(to_dst_abs + 9, to_dst_abs - 1, to_dst_abs)
            UPDATE_PROTECT(to_dst_abs + 9, to_dst_abs + 10, to_dst_abs)
        }
        // One up one right
        if ((to_dst_abs / 10) < 9 && (to_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(to_dst_abs + 11, to_dst_abs + 1, to_dst_abs)
            UPDATE_PROTECT(to_dst_abs + 11, to_dst_abs + 10, to_dst_abs)
        }
        // One down one left
        if ((to_dst_abs / 10) > 0 && (to_dst_abs % 10) >= 1)
        {
            UPDATE_PROTECT(to_dst_abs - 11, to_dst_abs - 1, to_dst_abs)
            UPDATE_PROTECT(to_dst_abs - 11, to_dst_abs - 10, to_dst_abs)
        }
        // One down one right
        if ((to_dst_abs / 10) > 0 && (to_dst_abs % 10) < 9)
        {
            UPDATE_PROTECT(to_dst_abs - 9, to_dst_abs + 1, to_dst_abs)
            UPDATE_PROTECT(to_dst_abs - 9, to_dst_abs - 10, to_dst_abs)
        }
    }
    // Case 5: Empty square is protectee via abdication by moving piece
    const uint8_t from_src_abs = 10 * from_row_idx + from_col_idx;
    // One up other down
    if ((from_src_abs / 10) < 9 && (from_src_abs / 10) > 0)
    {
        UPDATE_PROTECT(from_src_abs + 10, from_src_abs, from_src_abs - 10)
        UPDATE_PROTECT(from_src_abs - 10, from_src_abs, from_src_abs + 10)
    }
    // One up other left
    if ((from_src_abs / 10) < 9 && (from_src_abs % 10) > 0)
    {
        UPDATE_PROTECT(from_src_abs + 10, from_src_abs, from_src_abs - 1)
        UPDATE_PROTECT(from_src_abs - 1, from_src_abs, from_src_abs + 10)
    }
    // One up other right
    if ((from_src_abs / 10) < 9 && (from_src_abs % 10) < 9)
    {
        UPDATE_PROTECT(from_src_abs + 10, from_src_abs, from_src_abs + 1)
        UPDATE_PROTECT(from_src_abs + 1, from_src_abs, from_src_abs + 10)
    }
    // One down other left
    if ((from_src_abs / 10) > 0 && (from_src_abs % 10) > 0)
    {
        UPDATE_PROTECT(from_src_abs - 10, from_src_abs, from_src_abs - 1)
        UPDATE_PROTECT(from_src_abs - 1, from_src_abs, from_src_abs - 10)
    }
    // One down other right
    if ((from_src_abs / 10) > 0 && (from_src_abs % 10) < 9)
    {
        UPDATE_PROTECT(from_src_abs - 10, from_src_abs, from_src_abs + 1)
        UPDATE_PROTECT(from_src_abs + 1, from_src_abs, from_src_abs - 10)
    }
    // One left other right
    if ((from_src_abs % 10) < 9 && (from_src_abs % 10) > 0)
    {
        UPDATE_PROTECT(from_src_abs - 1, from_src_abs, from_src_abs + 1)
        UPDATE_PROTECT(from_src_abs + 1, from_src_abs, from_src_abs - 1)
    }
#undef UPDATE_PROTECT

    // If the opponent has lost the flag, the game is over.
    if (!to_wins && to_piece.type == FLAG)
    {
        d_flag_captured[env_idx] = player;
    }

    d_red_death[env_idx] = ((to_piece.color == 1 || player == 1) && xpiece.color != 1) ? 10 * to_row_idx + to_col_idx : 0xff;
    d_blue_death[env_idx] = ((to_piece.color == 2 || player == 2) && xpiece.color != 2) ? 99 - (10 * to_row_idx + to_col_idx) : 0xff;

    // Update last_moved_piece_type and prev_dst_abs
    d_boards[env_idx].last_moved_piece_type = (to_wins || tie) ? 0xff : (from_piece.visible ? from_piece.type : HIDDEN_PIECE);
    d_boards[env_idx].prev_prev_dst_abs = d_boards[env_idx].prev_dst_abs;
    d_boards[env_idx].prev_dst_abs = 10 * to_row_idx + to_col_idx;
}
