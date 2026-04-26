/*
  Fast PSV decoder in C for Python (via ctypes).

  Decodes PackedSfenValue (40 bytes) → input planes (48×9×9 float)
                                      + policy index (int)
                                      + WDL target (3 floats)

  ~100x faster than pure Python Huffman decoding.

  Build:
    gcc -O3 -shared -fPIC -o psv_decode_c.so psv_decode_c.c -lm
*/

#include <math.h>
#include <stdint.h>
#include <string.h>

// =====================================================================
// Piece types (YaneuraOu convention)
// =====================================================================

#define NO_PIECE    0
#define PAWN        1
#define LANCE       2
#define KNIGHT      3
#define SILVER      4
#define BISHOP      5
#define ROOK        6
#define GOLD        7
#define KING        8

#define PRO_PAWN    9
#define PRO_LANCE   10
#define PRO_KNIGHT  11
#define PRO_SILVER  12
#define HORSE       13
#define DRAGON      14

#define COLOR_BLACK 0
#define COLOR_WHITE 1

// =====================================================================
// Huffman table for board pieces
// =====================================================================

// Decode table: for each (code, bits) pair, what piece type
static const struct { int code; int bits; int piece; } huffman_table[] = {
    {0x00, 1, NO_PIECE},  // 0
    {0x01, 2, PAWN},      // 01
    {0x03, 4, LANCE},     // 0011
    {0x0b, 4, KNIGHT},    // 1011
    {0x07, 4, SILVER},    // 0111
    {0x1f, 6, BISHOP},    // 011111
    {0x3f, 6, ROOK},      // 111111
    {0x0f, 5, GOLD},      // 01111
};
#define HUFFMAN_TABLE_SIZE 8

// =====================================================================
// BitStream reader
// =====================================================================

typedef struct {
    const uint8_t* data;
    int cursor;
} BitStream;

static inline int read_one_bit(BitStream* bs) {
    if (bs->cursor >= 256) return 0;  // bounds check
    int byte_idx = bs->cursor >> 3;
    int bit_idx = bs->cursor & 7;
    bs->cursor++;
    return (bs->data[byte_idx] >> bit_idx) & 1;
}

static inline int read_n_bits(BitStream* bs, int n) {
    int val = 0;
    for (int i = 0; i < n; i++)
        val |= read_one_bit(bs) << i;
    return val;
}

static int decode_piece(BitStream* bs) {
    int code = 0;
    int bits = 0;
    while (bits < 6) {
        code |= read_one_bit(bs) << bits;
        bits++;
        for (int i = 0; i < HUFFMAN_TABLE_SIZE; i++) {
            if (huffman_table[i].bits == bits && huffman_table[i].code == code)
                return huffman_table[i].piece;
        }
    }
    return NO_PIECE;
}

// =====================================================================
// Direction table for policy encoding
// =====================================================================

// 10 directions (same as shogi_model_v2.py)
// Direction: delta_file, delta_rank patterns
static int g_direction_table[81 * 81];
static int g_direction_inited = 0;

// Directions:
// 0: Up (file same, rank decreasing)
// 1: Down
// 2: Left (file increasing)
// 3: Right (file decreasing)
// 4: Up-Left
// 5: Up-Right
// 6: Down-Left
// 7: Down-Right
// 8: Knight-Left (file+1, rank-2)
// 9: Knight-Right (file-1, rank-2)

static void init_direction_table(void) {
    if (g_direction_inited) return;

    memset(g_direction_table, -1, sizeof(g_direction_table));

    for (int from_sq = 0; from_sq < 81; from_sq++) {
        int from_f = from_sq / 9;
        int from_r = from_sq % 9;

        for (int to_sq = 0; to_sq < 81; to_sq++) {
            int to_f = to_sq / 9;
            int to_r = to_sq % 9;

            int df = to_f - from_f;
            int dr = to_r - from_r;

            if (df == 0 && dr == 0) continue;

            int dir = -1;

            // Knight moves
            if (df == 1 && dr == -2) { dir = 8; }
            else if (df == -1 && dr == -2) { dir = 9; }
            // Straight and diagonal
            else if (df == 0 && dr < 0) { // Up
                // Check all squares between are empty (sliding)
                dir = 0;
            }
            else if (df == 0 && dr > 0) { dir = 1; } // Down
            else if (df > 0 && dr == 0) { dir = 2; } // Left
            else if (df < 0 && dr == 0) { dir = 3; } // Right
            else if (df > 0 && dr < 0 && df == -dr) { dir = 4; } // Up-Left
            else if (df < 0 && dr < 0 && df == dr) { dir = 5; } // Up-Right
            else if (df > 0 && dr > 0 && df == dr) { dir = 6; } // Down-Left
            else if (df < 0 && dr > 0 && -df == dr) { dir = 7; } // Down-Right

            if (dir >= 0)
                g_direction_table[from_sq * 81 + to_sq] = dir;
        }
    }
    g_direction_inited = 1;
}

// =====================================================================
// YaneuraOu move decoding
// =====================================================================

#define MOVE_DROP_FLAG 81
#define NUM_DIRECTIONS 10
#define NUM_PROMO_DIRECTIONS 10

static int decode_move_to_policy(uint16_t move_raw, int turn) {
    if (move_raw == 0) return -1;

    int to_sq = move_raw & 0x7F;
    int from_raw = (move_raw >> 7) & 0x7F;
    int promote = (move_raw >> 14) & 1;

    if (to_sq >= 81) return -1;

    int flip = (turn == COLOR_WHITE);

    if (from_raw >= MOVE_DROP_FLAG) {
        // Drop move
        int piece_type = from_raw - MOVE_DROP_FLAG;
        if (piece_type < 1 || piece_type > 7) return -1;
        int pt = piece_type - 1; // 0-6
        int to = to_sq;
        if (flip) {
            int to_f = to / 9, to_r = to % 9;
            to = (8 - to_f) * 9 + (8 - to_r);
        }
        int direction = NUM_DIRECTIONS + NUM_PROMO_DIRECTIONS + pt; // 20-26
        return direction * 81 + to;
    } else {
        // Board move
        if (from_raw >= 81) return -1;
        int from = from_raw;
        int to = to_sq;
        if (flip) {
            int ff = from / 9, fr = from % 9;
            from = (8 - ff) * 9 + (8 - fr);
            int tf = to / 9, tr = to % 9;
            to = (8 - tf) * 9 + (8 - tr);
        }
        int dir = g_direction_table[from * 81 + to];
        if (dir < 0) return -1;
        if (promote) dir += NUM_DIRECTIONS; // 0-9 → 10-19
        return dir * 81 + to;
    }
}

// =====================================================================
// Main decode function: PSV → planes + policy + WDL
// =====================================================================

// Piece type to plane index
static const int piece_to_plane[] = {
    -1, // NO_PIECE
    0, 1, 2, 3, 4, 5, 6, 7,  // PAWN..KING
    8, 9, 10, 11, 12, 13,     // PRO_PAWN..DRAGON
};

/*
  Decode one PSV record (40 bytes) into training data.

  Args:
    record:     40-byte PSV record
    planes:     output float[48*9*9] (will be zeroed and filled)
    policy_idx: output int (policy index, -1 if invalid)
    wdl:        output float[3] (win, draw, loss)
    eval_coef:  sigmoid coefficient for score→WDL (default 600)

  Returns: 0 on success, -1 on error
*/
int decode_psv_record(
    const uint8_t* record,
    float* planes,       // [48 * 9 * 9]
    int* policy_idx,
    float* wdl,          // [3]
    float eval_coef)
{
    init_direction_table();

    const uint8_t* sfen_bytes = record;
    int16_t score;
    uint16_t move_raw, game_ply;
    int8_t game_result;

    memcpy(&score, record + 32, 2);
    memcpy(&move_raw, record + 34, 2);
    memcpy(&game_ply, record + 36, 2);
    memcpy(&game_result, record + 38, 1);

    // Zero planes
    memset(planes, 0, 48 * 9 * 9 * sizeof(float));

    // Decode packed sfen
    BitStream bs = {sfen_bytes, 0};

    // 1. Turn
    int turn = read_one_bit(&bs);
    int flip = (turn == COLOR_WHITE);

    // 2. King squares
    int king_sq_black = read_n_bits(&bs, 7);
    int king_sq_white = read_n_bits(&bs, 7);

    // Board array
    int board[81];
    memset(board, 0, sizeof(board));

    if (king_sq_black < 81) board[king_sq_black] = KING;           // BLACK king
    if (king_sq_white < 81) board[king_sq_white] = (16 + KING);    // WHITE king

    // 3. Decode board pieces
    for (int sq = 0; sq < 81; sq++) {
        if (sq == king_sq_black || sq == king_sq_white) continue;
        if (bs.cursor >= 256) break;

        int piece = decode_piece(&bs);
        if (piece == NO_PIECE) continue;
        if (bs.cursor >= 256) break;

        int promoted = 0;
        if (piece != GOLD) {
            promoted = read_one_bit(&bs);
        }
        if (bs.cursor >= 256) break;
        int color = read_one_bit(&bs);

        if (promoted) piece += 8;
        board[sq] = piece + (color == COLOR_WHITE ? 16 : 0);
    }

    // 4. Set board planes
    for (int sq = 0; sq < 81; sq++) {
        int piece_val = board[sq];
        if (piece_val == 0) continue;

        int color = (piece_val >= 16) ? COLOR_WHITE : COLOR_BLACK;
        int piece_type = piece_val & 15;
        int plane_idx = piece_to_plane[piece_type];
        if (plane_idx < 0) continue;

        int file = sq / 9;
        int rank = sq % 9;

        int is_ours, sq_f, sq_r;
        if (flip) {
            is_ours = (color == COLOR_WHITE);
            sq_f = 8 - file;
            sq_r = 8 - rank;
        } else {
            is_ours = (color == COLOR_BLACK);
            sq_f = file;
            sq_r = rank;
        }

        int offset = is_ours ? 0 : 14;
        planes[(offset + plane_idx) * 81 + sq_r * 9 + sq_f] = 1.0f;
    }

    // 5. Hand pieces
    int hand_counts[2][7]; // [color][P,L,N,S,B,R,G]
    memset(hand_counts, 0, sizeof(hand_counts));

    while (bs.cursor < 256) {
        int piece = decode_piece(&bs);
        if (piece == NO_PIECE) break;
        if (bs.cursor >= 256) break;
        int color = read_one_bit(&bs);
        if (piece >= 1 && piece <= 7) {
            hand_counts[color][piece - 1]++;
        }
    }

    for (int color = 0; color < 2; color++) {
        for (int pt = 0; pt < 7; pt++) {
            int count = hand_counts[color][pt];
            if (count > 0) {
                int is_ours;
                if (flip) is_ours = (color == COLOR_WHITE);
                else      is_ours = (color == COLOR_BLACK);
                int base = is_ours ? 29 : 36;
                float val = count / 18.0f;
                for (int i = 0; i < 81; i++)
                    planes[(base + pt) * 81 + i] = val;
            }
        }
    }

    // Plane 43: all ones
    for (int i = 0; i < 81; i++)
        planes[43 * 81 + i] = 1.0f;

    // 6. Policy target
    *policy_idx = decode_move_to_policy(move_raw, turn);

    // 7. WDL target (blend score + game result)
    float win_rate = 1.0f / (1.0f + expf(-(float)score / eval_coef));

    float hard_w, hard_d, hard_l;
    if (game_result == 1)       { hard_w = 1.0f; hard_d = 0.0f; hard_l = 0.0f; }
    else if (game_result == 0)  { hard_w = 0.0f; hard_d = 1.0f; hard_l = 0.0f; }
    else                        { hard_w = 0.0f; hard_d = 0.0f; hard_l = 1.0f; }

    wdl[0] = 0.7f * win_rate       + 0.3f * hard_w;
    wdl[1] = 0.0f                  + 0.3f * hard_d;
    wdl[2] = 0.7f * (1.0f - win_rate) + 0.3f * hard_l;

    return 0;
}

/*
  Batch decode: decode N records at once.

  Args:
    records:     N×40 byte array
    n:           number of records
    planes:      output float[N * 48 * 9 * 9]
    policy_idxs: output int[N]
    wdls:        output float[N * 3]
    eval_coef:   sigmoid coefficient

  Returns: number of successfully decoded records
*/
int decode_psv_batch(
    const uint8_t* records,
    int n,
    float* planes,
    int* policy_idxs,
    float* wdls,
    float eval_coef)
{
    init_direction_table();
    int ok = 0;
    for (int i = 0; i < n; i++) {
        int ret = decode_psv_record(
            records + i * 40,
            planes + i * 48 * 81,
            &policy_idxs[i],
            &wdls[i * 3],
            eval_coef);
        if (ret == 0) ok++;
    }
    return ok;
}
