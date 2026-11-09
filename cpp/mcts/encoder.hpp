// Input encoding (8, 8, 36) and policy indexing (4672), bit-for-bit compatible with
// core/chess_environment.fast_fen_to_example and core/move_encoding.move_to_index.
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../third_party/chess-library/chess.hpp"

namespace nash {

constexpr int kPlanes = 36;
constexpr int kTensorSize = 8 * 8 * kPlanes;  // 2304 floats
constexpr int kPolicySize = 8 * 8 * 73;       // 4672

using U64 = std::uint64_t;

// Everything fast_fen_to_example needs, in tensor coordinates: bit = row * 8 + col, where row 0 is
// the rank furthest from the side to move (rank 8 for white, rank 1 for black) and col 0 is the a-file.
struct EncodeInput {
    // Channels P, N, B (even (row + col) in tensor coordinates), D (odd), R, Q, K.
    std::array<U64, 7> hero{};
    std::array<U64, 7> villain{};
    bool castle[4] = {false, false, false, false};  // hero K, hero Q, villain K, villain Q
    int ep = -1;                                    // tensor bit, or -1
    float material = 0.0f;                          // (hero - villain) / 40.2, computed in double like Python
};

inline int popcount(U64 b) { return __builtin_popcountll(b); }
inline int lsb(U64 b) { return __builtin_ctzll(b); }

constexpr U64 kColA = 0x0101010101010101ULL;
constexpr U64 kColH = 0x8080808080808080ULL;
constexpr U64 kRow2 = 0x0000000000FF0000ULL;  // tensor row 2
constexpr U64 kRow5 = 0x0000FF0000000000ULL;  // tensor row 5

inline void set_plane(float* out, U64 bits, int plane) {
    while (bits) {
        const int s = lsb(bits);
        bits &= bits - 1;
        out[s * kPlanes + plane] = 1.0f;
    }
}

// Mirrors generate_pseudo_legal_moves() plus the piece / castling / en passant / material planes.
// `out` must hold kTensorSize floats; it is fully overwritten.
inline void fill_tensor(const EncodeInput& in, float* out) {
    std::memset(out, 0, sizeof(float) * kTensorSize);

    U64 occ = 0;
    for (int i = 0; i < 7; ++i) occ |= in.hero[i] | in.villain[i];

    for (int i = 0; i < 7; ++i) {
        set_plane(out, in.hero[i], i);
        set_plane(out, in.villain[i], i + 9);
    }
    const int castle_planes[4] = {7, 8, 16, 17};
    for (int i = 0; i < 4; ++i)
        if (in.castle[i]) set_plane(out, ~0ULL, castle_planes[i]);
    if (in.ep >= 0) out[in.ep * kPlanes + 18] = 1.0f;

    // Slider / leaper attack tables are geometric, so chess-library's a1=0 tables work unchanged on
    // tensor coordinates (bit = row * 8 + col is just another labelling of the same 8x8 grid).
    for (int side = 0; side < 2; ++side) {
        const auto& p = side == 0 ? in.hero : in.villain;
        const int base = side == 0 ? 19 : 26;
        U64 moves[7] = {0, 0, 0, 0, 0, 0, 0};

        const U64 pawns = p[0];
        U64 attacks;
        if (side == 0) {  // hero pawns move toward row 0
            attacks = ((pawns >> 7) & ~kColA) | ((pawns >> 9) & ~kColH);
            const U64 single = (pawns >> 8) & ~occ;
            moves[0] = single | (((single & kRow5) >> 8) & ~occ);
        } else {  // villain pawns move toward row 7
            attacks = ((pawns << 9) & ~kColA) | ((pawns << 7) & ~kColH);
            const U64 single = (pawns << 8) & ~occ;
            moves[0] = single | (((single & kRow2) << 8) & ~occ);
        }

        const chess::Bitboard occ_bb(occ);
        for (U64 b = p[1]; b; b &= b - 1) moves[1] |= chess::attacks::knight(chess::Square(lsb(b))).getBits();
        for (U64 b = p[2]; b; b &= b - 1) moves[2] |= chess::attacks::bishop(chess::Square(lsb(b)), occ_bb).getBits();
        for (U64 b = p[3]; b; b &= b - 1) moves[3] |= chess::attacks::bishop(chess::Square(lsb(b)), occ_bb).getBits();
        for (U64 b = p[4]; b; b &= b - 1) moves[4] |= chess::attacks::rook(chess::Square(lsb(b)), occ_bb).getBits();
        for (U64 b = p[5]; b; b &= b - 1) moves[5] |= chess::attacks::queen(chess::Square(lsb(b)), occ_bb).getBits();
        for (U64 b = p[6]; b; b &= b - 1) moves[6] |= chess::attacks::king(chess::Square(lsb(b))).getBits();

        for (int i = 0; i < 7; ++i) set_plane(out, moves[i], base + i);
        set_plane(out, attacks, side == 0 ? 33 : 34);
    }

    for (int s = 0; s < 64; ++s) out[s * kPlanes + 35] = in.material;
}

inline double piece_value(char upper) {
    switch (upper) {
        case 'P': return 1;
        case 'N': return 3;
        case 'B': return 3.1;
        case 'D': return 3.1;
        case 'R': return 5;
        case 'Q': return 9;
        default: return 0;  // K
    }
}

// Literal re-implementation of fast_fen_to_example's FEN parsing (castling and en passant fields are
// taken verbatim from the string, exactly as the Python does).
inline EncodeInput encode_input_from_fen(std::string_view fen) {
    std::vector<std::string_view> parts;
    size_t i = 0;
    while (i < fen.size()) {
        while (i < fen.size() && std::isspace(static_cast<unsigned char>(fen[i]))) ++i;
        size_t j = i;
        while (j < fen.size() && !std::isspace(static_cast<unsigned char>(fen[j]))) ++j;
        if (j > i) parts.push_back(fen.substr(i, j - i));
        i = j;
    }
    if (parts.size() < 4) throw std::invalid_argument("FEN needs at least 4 fields: " + std::string(fen));

    const std::string_view board = parts[0], turn = parts[1], castling = parts[2], ep = parts[3];
    const bool white = turn == "w";

    EncodeInput in;
    double material_hero = 0, material_villain = 0;
    U64 hero[7] = {}, villain[7] = {};  // pre-flip coordinates (row 0 = rank 8)

    int row = 0, col = 0;
    for (char ch : board) {
        if (ch == '/') {
            ++row;
            col = 0;
            continue;
        }
        if (ch >= '0' && ch <= '9') {
            col += ch - '0';
            continue;
        }
        const char up = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        int idx = -1;
        switch (up) {
            case 'P': idx = 0; break;
            case 'N': idx = 1; break;
            case 'B': idx = 2; break;
            case 'R': idx = 4; break;
            case 'Q': idx = 5; break;
            case 'K': idx = 6; break;
            default: break;
        }
        if (idx >= 0) {
            if (row < 0 || row > 7 || col < 0 || col > 7) throw std::invalid_argument("FEN board out of range");
            char value_char = up;
            if (up == 'B') {
                const bool even = (row + col) % 2 == 0;
                // White: even -> B (2), odd -> D (3). Black: swapped, because the rows get flipped.
                const bool light_plane = white ? even : !even;
                idx = light_plane ? 2 : 3;
                value_char = light_plane ? 'B' : 'D';
            }
            const bool upper = std::isupper(static_cast<unsigned char>(ch));
            const U64 bit = 1ULL << (row * 8 + col);
            if ((upper && white) || (!upper && !white)) {
                hero[idx] |= bit;
                material_hero += piece_value(value_char);
            } else {
                villain[idx] |= bit;
                material_villain += piece_value(value_char);
            }
        }
        ++col;
    }

    auto has = [&](char c) { return castling.find(c) != std::string_view::npos; };
    in.castle[0] = white ? has('K') : has('k');
    in.castle[1] = white ? has('Q') : has('q');
    in.castle[2] = white ? has('k') : has('K');
    in.castle[3] = white ? has('q') : has('Q');

    int ep_bit = -1;
    if (ep != "-") {
        static constexpr std::string_view rows = "87654321", cols = "abcdefgh";
        if (ep.size() < 2) throw std::invalid_argument("bad en passant field");
        const auto c = cols.find(ep[0]);
        const auto r = rows.find(ep[1]);
        if (c == std::string_view::npos || r == std::string_view::npos)
            throw std::invalid_argument("bad en passant field");
        ep_bit = static_cast<int>(r * 8 + c);
    }

    const int flip = white ? 0 : 56;  // np.flip(axis=0) maps row r -> 7 - r, i.e. bit ^ 56
    for (int k = 0; k < 7; ++k) {
        in.hero[k] = white ? hero[k] : __builtin_bswap64(hero[k]);
        in.villain[k] = white ? villain[k] : __builtin_bswap64(villain[k]);
    }
    in.ep = ep_bit >= 0 ? (ep_bit ^ flip) : -1;
    in.material = static_cast<float>((material_hero - material_villain) / 40.2);
    return in;
}

// ---------------------------------------------------------------------------------------------
// Board helpers shared by the encoder and the tree.

// Castling rights as python-chess's clean_castling_rights() reports them: the right must exist,
// the king must be on e1/e8 and the rook on its corner. Bits: 1 = WK, 2 = WQ, 4 = BK, 8 = BQ.
inline int clean_castling(const chess::Board& b) {
    using chess::Color;
    using chess::PieceType;
    using Side = chess::Board::CastlingRights::Side;
    const auto cr = b.castlingRights();
    const U64 wk = b.pieces(PieceType::KING, Color::WHITE).getBits();
    const U64 bk = b.pieces(PieceType::KING, Color::BLACK).getBits();
    const U64 wr = b.pieces(PieceType::ROOK, Color::WHITE).getBits();
    const U64 br = b.pieces(PieceType::ROOK, Color::BLACK).getBits();
    int out = 0;
    if (wk & (1ULL << 4)) {
        if (cr.has(Color::WHITE, Side::KING_SIDE) && (wr & (1ULL << 7))) out |= 1;
        if (cr.has(Color::WHITE, Side::QUEEN_SIDE) && (wr & (1ULL << 0))) out |= 2;
    }
    if (bk & (1ULL << 60)) {
        if (cr.has(Color::BLACK, Side::KING_SIDE) && (br & (1ULL << 63))) out |= 4;
        if (cr.has(Color::BLACK, Side::QUEEN_SIDE) && (br & (1ULL << 56))) out |= 8;
    }
    return out;
}

// En passant square (a1 = 0) if an en passant capture is legal in `moves`, else -1 (python-chess's
// "legal" en passant, which is what Board.fen() prints and what the repetition key uses).
inline int legal_ep_square(const chess::Board& b, const chess::Movelist& moves) {
    for (const auto& m : moves)
        if (m.typeOf() == chess::Move::ENPASSANT) return b.enpassantSq().index();
    return -1;
}

// What fast_fen_to_example(python_board.fen()) sees for this position.
inline EncodeInput encode_input_from_board(const chess::Board& b, int legal_ep) {
    using chess::Color;
    using chess::PieceType;
    const bool white = b.sideToMove() == Color::WHITE;
    const Color hero_c = white ? Color::WHITE : Color::BLACK;

    EncodeInput in;
    // Tensor bit for a1-based square s: white -> (7 - rank) * 8 + file = s ^ 56; black -> s.
    auto to_tensor = [&](U64 bb) { return white ? __builtin_bswap64(bb) : bb; };

    const PieceType types[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                                PieceType::ROOK, PieceType::QUEEN,  PieceType::KING};
    const int chans[6] = {0, 1, 2, 4, 5, 6};
    // In tensor coordinates the B plane is (row + col) even. Bit parity of row*8+col equals col parity
    // plus row parity, so "even" squares are the 0x55AA... pattern of the tensor grid.
    constexpr U64 kEvenTensor = 0xAA55AA55AA55AA55ULL;  // bit 0 (row 0, col 0) is even
    for (int c = 0; c < 2; ++c) {
        const Color color = c == 0 ? Color::WHITE : Color::BLACK;
        auto& dst = color == hero_c ? in.hero : in.villain;
        for (int t = 0; t < 6; ++t) {
            const U64 bb = to_tensor(b.pieces(types[t], color).getBits());
            if (types[t] == PieceType::BISHOP) {
                dst[2] = bb & kEvenTensor;
                dst[3] = bb & ~kEvenTensor;
            } else {
                dst[chans[t]] = bb;
            }
        }
    }

    const int cr = clean_castling(b);
    const bool wk = cr & 1, wq = cr & 2, bk = cr & 4, bq = cr & 8;
    in.castle[0] = white ? wk : bk;
    in.castle[1] = white ? wq : bq;
    in.castle[2] = white ? bk : wk;
    in.castle[3] = white ? bq : wq;
    in.ep = legal_ep >= 0 ? (white ? (legal_ep ^ 56) : legal_ep) : -1;

    // Material in FEN order (rank 8 -> 1, file a -> h) so the double sum is identical to Python's.
    double mh = 0, mv = 0;
    for (int rank = 7; rank >= 0; --rank) {
        for (int file = 0; file < 8; ++file) {
            const auto piece = b.at(chess::Square(rank * 8 + file));
            if (piece == chess::Piece::NONE) continue;
            const auto pt = piece.type();
            double v = 0;
            if (pt == PieceType::PAWN) v = 1;
            else if (pt == PieceType::KNIGHT) v = 3;
            else if (pt == PieceType::BISHOP) v = 3.1;
            else if (pt == PieceType::ROOK) v = 5;
            else if (pt == PieceType::QUEEN) v = 9;
            if (piece.color() == hero_c) mh += v;
            else mv += v;
        }
    }
    in.material = static_cast<float>((mh - mv) / 40.2);
    return in;
}

// ---------------------------------------------------------------------------------------------
// Policy index (core/move_encoding.move_to_index) and UCI strings.

// King destination for chess-library's king-takes-rook castling encoding.
inline int move_to_square(const chess::Move& m) {
    if (m.typeOf() == chess::Move::CASTLING) {
        const int from = m.from().index(), to = m.to().index();
        return (from & ~7) + (to > from ? 6 : 2);
    }
    return m.to().index();
}

inline int policy_index(const chess::Move& m, bool white) {
    static constexpr int kQueenDirs[8][2] = {{-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}};
    static constexpr int kKnight[8][2] = {{-2, 1}, {-1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}};
    const int from = m.from().index(), to = move_to_square(m);
    const int fr = white ? 7 - (from >> 3) : (from >> 3), fc = from & 7;
    const int tr = white ? 7 - (to >> 3) : (to >> 3), tc = to & 7;
    const int dr = tr - fr, dc = tc - fc;
    int plane = -1;
    if (m.typeOf() == chess::Move::PROMOTION && m.promotionType() != chess::PieceType::QUEEN) {
        const auto pt = m.promotionType();
        const int piece = pt == chess::PieceType::KNIGHT ? 0 : (pt == chess::PieceType::BISHOP ? 1 : 2);
        plane = 64 + piece * 3 + (dc + 1);
    } else {
        for (int k = 0; k < 8; ++k)
            if (kKnight[k][0] == dr && kKnight[k][1] == dc) plane = 56 + k;
        if (plane < 0) {
            const int dist = std::max(std::abs(dr), std::abs(dc));
            const int ur = dr / dist, uc = dc / dist;
            for (int k = 0; k < 8; ++k)
                if (kQueenDirs[k][0] == ur && kQueenDirs[k][1] == uc) plane = k * 7 + (dist - 1);
        }
    }
    if (plane < 0) throw std::logic_error("move has no policy plane");
    return (fr * 8 + fc) * 73 + plane;
}

inline std::string move_uci(const chess::Move& m) {
    const int from = m.from().index(), to = move_to_square(m);
    std::string s;
    s += static_cast<char>('a' + (from & 7));
    s += static_cast<char>('1' + (from >> 3));
    s += static_cast<char>('a' + (to & 7));
    s += static_cast<char>('1' + (to >> 3));
    if (m.typeOf() == chess::Move::PROMOTION) {
        const auto pt = m.promotionType();
        s += pt == chess::PieceType::KNIGHT ? 'n' : pt == chess::PieceType::BISHOP ? 'b' : pt == chess::PieceType::ROOK ? 'r' : 'q';
    }
    return s;
}

}  // namespace nash
