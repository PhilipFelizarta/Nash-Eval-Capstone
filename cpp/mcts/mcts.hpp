// Batched PUCT Monte Carlo tree search. Network evaluation happens outside (Python):
// gather() returns the encoded leaves that need the network, backup() expands them and backs up.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "encoder.hpp"

namespace nash {

// Position equality key with python-chess's _transposition_key semantics: piece placement, side to
// move, *cleaned* castling rights and the en passant square only if an en passant capture is legal.
inline U64 mix64(U64 x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline U64 position_key(const chess::Board& b, int legal_ep) {
    using chess::Color;
    using chess::PieceType;
    U64 h = 0x6A09E667F3BCC908ULL;
    auto add = [&h](U64 v) { h = mix64(h ^ mix64(v)); };
    add(b.pieces(PieceType::PAWN).getBits());
    add(b.pieces(PieceType::KNIGHT).getBits());
    add(b.pieces(PieceType::BISHOP).getBits());
    add(b.pieces(PieceType::ROOK).getBits());
    add(b.pieces(PieceType::QUEEN).getBits());
    add(b.pieces(PieceType::KING).getBits());
    add(b.us(Color::WHITE).getBits());
    add(b.us(Color::BLACK).getBits());
    add(static_cast<U64>(b.sideToMove() == Color::WHITE) | (static_cast<U64>(clean_castling(b)) << 1) |
        (static_cast<U64>(legal_ep + 1) << 8));
    return h;
}

// python-chess Board.has_insufficient_material / is_insufficient_material, verbatim.
inline bool has_insufficient_material(const chess::Board& b, chess::Color color) {
    using chess::PieceType;
    const U64 us = b.us(color).getBits(), them = b.us(~color).getBits();
    const U64 pawns = b.pieces(PieceType::PAWN).getBits(), knights = b.pieces(PieceType::KNIGHT).getBits();
    const U64 bishops = b.pieces(PieceType::BISHOP).getBits(), rooks = b.pieces(PieceType::ROOK).getBits();
    const U64 queens = b.pieces(PieceType::QUEEN).getBits(), kings = b.pieces(PieceType::KING).getBits();
    constexpr U64 kDark = 0xAA55AA55AA55AA55ULL, kLight = 0x55AA55AA55AA55AAULL;
    if (us & (pawns | rooks | queens)) return false;
    if (us & knights) return popcount(us) <= 2 && !(them & ~kings & ~queens);
    if (us & bishops) {
        const bool same_color = !(bishops & kDark) || !(bishops & kLight);
        return same_color && !pawns && !knights;
    }
    return true;
}

inline bool is_insufficient_material(const chess::Board& b) {
    return has_insufficient_material(b, chess::Color::WHITE) && has_insufficient_material(b, chess::Color::BLACK);
}

// Builds a board and normalises castling rights / en passant the way python-chess would see them,
// so chess-library and python-chess agree on the legal moves and on every later position.
inline chess::Board make_board(const std::string& fen) {
    chess::Board b;
    if (!b.setFen(fen)) throw std::invalid_argument("invalid FEN: " + fen);
    if (b.pieces(chess::PieceType::KING, chess::Color::WHITE).count() != 1 ||
        b.pieces(chess::PieceType::KING, chess::Color::BLACK).count() != 1)
        throw std::invalid_argument("FEN must have exactly one king per side: " + fen);
    const int cr = clean_castling(b);
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, b);
    const int ep = legal_ep_square(b, moves);

    // getFen() = "<placement> <stm> <castling> <ep> <hfm> <fullmove>"
    const std::string full = b.getFen(true);
    std::vector<std::string> f;
    size_t i = 0;
    while (i < full.size()) {
        size_t j = full.find(' ', i);
        if (j == std::string::npos) j = full.size();
        f.push_back(full.substr(i, j - i));
        i = j + 1;
    }
    if (f.size() < 6) throw std::invalid_argument("invalid FEN: " + fen);
    std::string castling;
    if (cr & 1) castling += 'K';
    if (cr & 2) castling += 'Q';
    if (cr & 4) castling += 'k';
    if (cr & 8) castling += 'q';
    if (castling.empty()) castling = "-";
    std::string eps = "-";
    if (ep >= 0) {
        eps = std::string(1, static_cast<char>('a' + (ep & 7))) + static_cast<char>('1' + (ep >> 3));
    }
    // chess-library keeps the half-move clock in a uint8 for unmakeMove(); anything >= 100 means the same.
    const std::string hfm = std::to_string(std::min<long>(std::strtol(f[4].c_str(), nullptr, 10), 200));
    const std::string normalized = f[0] + " " + f[1] + " " + castling + " " + eps + " " + hfm + " " + f[5];
    if (!b.setFen(normalized)) throw std::invalid_argument("invalid FEN: " + fen);
    return b;
}

struct Config {
    float c_puct = 2.0f;
    float fpu_reduction = 0.25f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    int virtual_loss = 1;
    std::uint64_t seed = 0;
};

enum NodeState : std::uint8_t { UNVISITED = 0, PENDING = 1, EXPANDED = 2, TERMINAL = 3 };
// Proven game-theoretic value from the view of the side to move AT the node.
enum Proven : std::int8_t { UNKNOWN = 0, WIN = 1, DRAW = 2, LOSS = 3 };

inline double proven_value(std::int8_t p) { return p == WIN ? 1.0 : (p == DRAW ? 0.5 : 0.0); }

struct Node {
    double w = 0.0;      // sum of backed-up values, from the view of the player who made `move`
    U64 key = 0;         // repetition key, set when the node is first reached
    std::int32_t parent = -1;
    std::int32_t first_child = -1;
    std::int32_t n = 0;  // real visits
    std::int32_t vl = 0; // in-flight selections through this node (virtual loss count)
    float prior = 0.0f;
    std::uint16_t move = 0;  // chess::Move bits of the move from parent to this node
    std::uint16_t policy = 0;
    std::uint16_t num_children = 0;
    std::uint8_t state = UNVISITED;
    std::int8_t proven = UNKNOWN;
};

struct RootStat {
    std::string uci;
    int policy;
    int visits;
    double q;
    float prior;
};

struct GatherStats {
    int selections = 0, leaves = 0, terminals = 0, collisions = 0;
};

class MCTS {
   public:
    explicit MCTS(const Config& cfg) : cfg_(cfg), rng_(cfg.seed) { reset(std::string(chess::constants::STARTPOS), {}, true); }

    const Config& config() const { return cfg_; }
    bool noise_enabled() const { return noise_enabled_; }
    void set_noise_enabled(bool v) { noise_enabled_ = v; }

    void reset(const std::string& fen, const std::vector<std::string>& history, bool add_noise) {
        board_ = make_board(fen);
        history_keys_.clear();
        chess::Movelist moves;
        for (const auto& uci : history) {
            chess::movegen::legalmoves(moves, board_);
            history_keys_.push_back(position_key(board_, legal_ep_square(board_, moves)));
            board_.makeMove(find_move(moves, uci));
        }
        noise_enabled_ = add_noise;
        new_root();
    }

    // Runs up to n selections. Returns the number of leaves written to `out` (k * kTensorSize floats).
    int gather(int n, std::vector<float>& out) {
        if (!pending_.empty() || !collisions_.empty())
            throw std::runtime_error("gather() called again before backup()");
        stats_ = GatherStats{};
        out.resize(static_cast<size_t>(std::max(n, 0)) * kTensorSize);
        for (int i = 0; i < n; ++i) {
            ++stats_.selections;
            simulate(out.data() + pending_.size() * kTensorSize);
        }
        out.resize(pending_.size() * kTensorSize);
        return static_cast<int>(pending_.size());
    }

    int num_pending() const { return static_cast<int>(pending_.size()); }

    // policy: (k, 4672) logits, wdl: (k, 3), rows in gather() order.
    void backup(const float* policy, const float* wdl, int k) {
        if (k != static_cast<int>(pending_.size()))
            throw std::invalid_argument("backup() got " + std::to_string(k) + " rows, expected " +
                                        std::to_string(pending_.size()));
        std::vector<double> scratch;
        for (int i = 0; i < k; ++i) {
            const Pending& p = pending_[i];
            const float* logits = policy + static_cast<size_t>(i) * kPolicySize;
            expand(p, logits, scratch);
            const float* v3 = wdl + static_cast<size_t>(i) * 3;
            double v = static_cast<double>(v3[0]) + 0.5 * static_cast<double>(v3[1]);
            if (!std::isfinite(v)) v = 0.5;
            v = std::min(1.0, std::max(0.0, v));
            remove_vl(p.node);
            backup_value(p.node, v);
        }
        for (int leaf : collisions_) remove_vl(leaf);
        pending_.clear();
        collisions_.clear();
        pending_moves_.clear();
    }

    // Drops the in-flight batch (e.g. if the evaluator failed), restoring the tree.
    void cancel() {
        for (const auto& p : pending_) {
            remove_vl(p.node);
            nodes_[p.node].state = UNVISITED;
        }
        for (int leaf : collisions_) remove_vl(leaf);
        pending_.clear();
        collisions_.clear();
        pending_moves_.clear();
    }

    int root_visits() const { return nodes_[0].n; }

    // Expected score for the side to move at the root.
    double root_value() const { return nodes_[0].n > 0 ? 1.0 - nodes_[0].w / nodes_[0].n : 0.5; }

    std::int8_t root_proven() const { return nodes_[0].proven; }
    bool root_terminal() const { return nodes_[0].state == TERMINAL; }

    std::vector<RootStat> root_stats() const {
        std::vector<RootStat> out;
        const Node& r = nodes_[0];
        if (r.state != EXPANDED) return out;
        for (int j = 0; j < r.num_children; ++j) {
            const Node& c = nodes_[r.first_child + j];
            double q;
            if (c.proven != UNKNOWN) q = 1.0 - proven_value(c.proven);
            else if (c.n > 0) q = c.w / c.n;
            else q = std::numeric_limits<double>::quiet_NaN();
            out.push_back({move_uci(chess::Move(c.move)), c.policy, c.n, q, c.prior});
        }
        return out;
    }

    // Proven status of each root move from the root side-to-move's view (root_stats() order).
    std::vector<std::int8_t> root_child_proven() const {
        std::vector<std::int8_t> out;
        const Node& r = nodes_[0];
        if (r.state != EXPANDED) return out;
        for (int j = 0; j < r.num_children; ++j) {
            const std::int8_t c = nodes_[r.first_child + j].proven;
            std::int8_t mine = c;
            if (c == WIN) mine = LOSS;
            else if (c == LOSS) mine = WIN;
            out.push_back(mine);
        }
        return out;
    }

    std::string select_move(double temperature) {
        const Node& r = nodes_[0];
        if (r.state != EXPANDED || r.num_children == 0)
            throw std::runtime_error("root is not expanded (no search yet, or no legal moves)");

        // A child that is a proven loss for its side to move wins for us: play it (mate-in-1 first).
        int best = -1;
        for (int j = 0; j < r.num_children; ++j) {
            const int ci = r.first_child + j;
            const Node& c = nodes_[ci];
            if (c.proven != LOSS) continue;
            if (best < 0) { best = ci; continue; }
            const Node& b = nodes_[best];
            const bool c_mate = c.state == TERMINAL, b_mate = b.state == TERMINAL;
            if ((c_mate && !b_mate) || (c_mate == b_mate && c.n > b.n)) best = ci;
        }
        if (best >= 0) return move_uci(chess::Move(nodes_[best].move));

        // Otherwise never pick a proven loss for us unless every move loses.
        std::vector<int> cand;
        for (int j = 0; j < r.num_children; ++j)
            if (nodes_[r.first_child + j].proven != WIN) cand.push_back(r.first_child + j);
        if (cand.empty())
            for (int j = 0; j < r.num_children; ++j) cand.push_back(r.first_child + j);

        int nmax = 0;
        for (int ci : cand) nmax = std::max(nmax, nodes_[ci].n);
        if (temperature <= 1e-6 || nmax == 0) {
            best = cand[0];
            for (int ci : cand) {
                const Node &c = nodes_[ci], &b = nodes_[best];
                if (c.n > b.n || (c.n == b.n && c.prior > b.prior)) best = ci;
            }
            return move_uci(chess::Move(nodes_[best].move));
        }
        std::vector<double> weights;
        for (int ci : cand) {
            const int n = nodes_[ci].n;
            weights.push_back(n > 0 ? std::exp((std::log(static_cast<double>(n)) - std::log(static_cast<double>(nmax))) / temperature) : 0.0);
        }
        std::discrete_distribution<int> pick(weights.begin(), weights.end());
        return move_uci(chess::Move(nodes_[cand[pick(rng_)]].move));
    }

    // Tree reuse: keep the subtree of `uci` as the new root, or start fresh if it was never expanded.
    void advance(const std::string& uci) {
        if (!pending_.empty() || !collisions_.empty()) throw std::runtime_error("advance() with a pending batch");
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);
        const chess::Move mv = find_move(moves, uci);

        int child = -1;
        const Node& r = nodes_[0];
        if (r.state == EXPANDED)
            for (int j = 0; j < r.num_children; ++j)
                if (nodes_[r.first_child + j].move == mv.move()) child = r.first_child + j;

        history_keys_.push_back(position_key(board_, legal_ep_square(board_, moves)));
        board_.makeMove(mv);

        if (child < 0 || nodes_[child].state != EXPANDED) {
            new_root();
            return;
        }
        // Copy the subtree breadth-first so every node's children stay contiguous.
        std::vector<Node> fresh;
        std::vector<std::int32_t> old_of_new;
        fresh.reserve(nodes_.size());
        fresh.push_back(nodes_[child]);
        old_of_new.push_back(child);
        fresh[0].parent = -1;
        for (size_t i = 0; i < fresh.size(); ++i) {
            const Node& o = nodes_[old_of_new[i]];
            if (o.num_children == 0) continue;
            const auto first = static_cast<std::int32_t>(fresh.size());
            for (int j = 0; j < o.num_children; ++j) {
                fresh.push_back(nodes_[o.first_child + j]);
                fresh.back().parent = static_cast<std::int32_t>(i);
                old_of_new.push_back(o.first_child + j);
            }
            fresh[i].first_child = first;
        }
        nodes_.swap(fresh);
        root_noised_ = false;
        maybe_noise_root();
    }

    std::vector<std::vector<std::string>> pending_paths() const {
        std::vector<std::vector<std::string>> out;
        for (const auto& p : pending_) out.push_back(path_to(p.node));
        return out;
    }

    long long total_virtual_loss() const {
        long long s = 0;
        for (const auto& nd : nodes_) s += nd.vl;
        return s;
    }

    size_t num_nodes() const { return nodes_.size(); }
    const GatherStats& last_gather() const { return stats_; }
    std::string root_fen() const { return board_.getFen(true); }

   private:
    struct Pending {
        std::int32_t node;
        std::int32_t offset;  // into pending_moves_
        std::int32_t count;
    };
    struct PendingMove {
        std::uint16_t move;
        std::uint16_t policy;
    };

    static chess::Move find_move(const chess::Movelist& moves, const std::string& uci) {
        for (const auto& m : moves)
            if (move_uci(m) == uci) return m;
        throw std::invalid_argument("illegal move: " + uci);
    }

    void new_root() {
        nodes_.assign(1, Node{});
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);
        nodes_[0].key = position_key(board_, legal_ep_square(board_, moves));
        root_noised_ = false;
        pending_.clear();
        collisions_.clear();
        pending_moves_.clear();
    }

    void maybe_noise_root() {
        Node& r = nodes_[0];
        if (root_noised_ || !noise_enabled_ || cfg_.dirichlet_epsilon <= 0.0f || r.state != EXPANDED ||
            r.num_children == 0)
            return;
        std::gamma_distribution<double> gamma(cfg_.dirichlet_alpha, 1.0);
        std::vector<double> eta(r.num_children);
        double sum = 0.0;
        for (auto& e : eta) sum += (e = gamma(rng_));
        root_noised_ = true;
        if (!(sum > 0.0)) return;
        const double eps = cfg_.dirichlet_epsilon;
        for (int j = 0; j < r.num_children; ++j) {
            Node& c = nodes_[r.first_child + j];
            c.prior = static_cast<float>((1.0 - eps) * c.prior + eps * eta[j] / sum);
        }
    }

    std::vector<std::string> path_to(int node) const {
        std::vector<std::string> rev;
        for (int x = node; x > 0; x = nodes_[x].parent) rev.push_back(move_uci(chess::Move(nodes_[x].move)));
        return {rev.rbegin(), rev.rend()};
    }

    void add_vl(int node) {
        for (int x = node; x >= 0; x = nodes_[x].parent) nodes_[x].vl += 1;
    }
    void remove_vl(int node) {
        for (int x = node; x >= 0; x = nodes_[x].parent) nodes_[x].vl -= 1;
    }

    // v = expected score for the side to move at `node`.
    void backup_value(int node, double v) {
        double val = 1.0 - v;  // stored from the view of the player who moved into the node
        for (int x = node; x >= 0; x = nodes_[x].parent) {
            nodes_[x].n += 1;
            nodes_[x].w += val;
            val = 1.0 - val;
        }
    }

    // Simple solver: a node is a proven win if some child is a proven loss (for the child's side to
    // move), a proven loss if every child is a proven win, a proven draw if every child is proven and
    // the best of them is a draw.
    void propagate_proof(int node) {
        for (int x = node; x >= 0; x = nodes_[x].parent) {
            Node& p = nodes_[x];
            if (p.state != EXPANDED) return;
            bool all = true, any_draw = false;
            std::int8_t res = UNKNOWN;
            for (int j = 0; j < p.num_children; ++j) {
                const std::int8_t c = nodes_[p.first_child + j].proven;
                if (c == LOSS) { res = WIN; break; }
                if (c == UNKNOWN) all = false;
                if (c == DRAW) any_draw = true;
            }
            if (res == UNKNOWN && all) res = any_draw ? DRAW : LOSS;
            if (res == p.proven) return;
            p.proven = res;
        }
    }

    bool is_threefold(int node, U64 key) const {
        const int limit = static_cast<int>(board_.halfMoveClock());
        int count = 1, step = 0;
        int x = node;
        // Ancestors in the tree, then the game history before the root.
        while (step < limit && x > 0) {
            x = nodes_[x].parent;
            ++step;
            if ((step & 1) == 0 && nodes_[x].key == key && ++count >= 3) return true;
        }
        if (step < limit) {
            for (int h = static_cast<int>(history_keys_.size()) - 1; h >= 0 && step < limit; --h) {
                ++step;
                if ((step & 1) == 0 && history_keys_[h] == key && ++count >= 3) return true;
            }
        }
        return false;
    }

    int select_child(int node) const {
        const Node& p = nodes_[node];
        const double n_parent = static_cast<double>(p.n) + static_cast<double>(p.vl) * cfg_.virtual_loss;
        const double sqrt_n = std::sqrt(std::max(n_parent, 1.0));
        const double parent_q = p.n > 0 ? 1.0 - p.w / p.n : 0.5;  // selecting player's view
        const double fpu = parent_q - cfg_.fpu_reduction;
        int best = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < p.num_children; ++j) {
            const int ci = p.first_child + j;
            const Node& c = nodes_[ci];
            if (c.proven == LOSS) return ci;  // the opponent is lost (mate found): always take it
            const double n_eff = static_cast<double>(c.n) + static_cast<double>(c.vl) * cfg_.virtual_loss;
            double q;
            if (c.proven != UNKNOWN) q = 1.0 - proven_value(c.proven);
            else q = n_eff > 0 ? c.w / n_eff : fpu;  // virtual losses count as visits worth 0
            double score = q + cfg_.c_puct * c.prior * sqrt_n / (1.0 + n_eff);
            if (c.proven == WIN) score -= 1e9;  // proven loss for us: only if every move loses
            if (score > best_score) {
                best_score = score;
                best = ci;
            }
        }
        return best;
    }

    void simulate(float* out_slot) {
        int node = 0;
        path_.clear();
        for (;;) {
            Node& nd = nodes_[node];
            if (nd.state == TERMINAL || (node != 0 && nd.proven != UNKNOWN)) {
                backup_value(node, proven_value(nd.proven));
                ++stats_.terminals;
                break;
            }
            if (nd.state == PENDING) {
                add_vl(node);  // keep the collision's virtual loss until backup() so later picks diversify
                collisions_.push_back(node);
                ++stats_.collisions;
                break;
            }
            if (nd.state == UNVISITED) {
                visit_leaf(node, out_slot);
                break;
            }
            const int child = select_child(node);
            const chess::Move mv(nodes_[child].move);
            board_.makeMove(mv);
            path_.push_back(mv);
            node = child;
        }
        for (auto it = path_.rbegin(); it != path_.rend(); ++it) board_.unmakeMove(*it);
    }

    void visit_leaf(int node, float* out_slot) {
        chess::movegen::legalmoves(moves_, board_);
        const int ep = legal_ep_square(board_, moves_);
        const U64 key = position_key(board_, ep);
        nodes_[node].key = key;

        std::int8_t terminal = UNKNOWN;
        if (moves_.empty()) {
            terminal = board_.inCheck() ? LOSS : DRAW;  // checkmate / stalemate
        } else if (node != 0 && (is_insufficient_material(board_) || board_.halfMoveClock() >= 100 ||
                                 is_threefold(node, key))) {
            terminal = DRAW;  // draw rules are not applied to the root, so a move can always be chosen
        }
        if (terminal != UNKNOWN) {
            Node& nd = nodes_[node];
            nd.state = TERMINAL;
            nd.proven = terminal;
            propagate_proof(nd.parent);
            backup_value(node, proven_value(terminal));
            ++stats_.terminals;
            return;
        }

        nodes_[node].state = PENDING;
        add_vl(node);
        const bool white = board_.sideToMove() == chess::Color::WHITE;
        Pending p{node, static_cast<std::int32_t>(pending_moves_.size()), static_cast<std::int32_t>(moves_.size())};
        for (const auto& m : moves_)
            pending_moves_.push_back({m.move(), static_cast<std::uint16_t>(policy_index(m, white))});
        pending_.push_back(p);
        ++stats_.leaves;
        fill_tensor(encode_input_from_board(board_, ep), out_slot);
    }

    void expand(const Pending& p, const float* logits, std::vector<double>& scratch) {
        scratch.resize(p.count);
        double mx = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < p.count; ++j) {
            const double l = logits[pending_moves_[p.offset + j].policy];
            scratch[j] = std::isfinite(l) ? l : -1e30;
            mx = std::max(mx, scratch[j]);
        }
        double sum = 0.0;
        for (int j = 0; j < p.count; ++j) sum += (scratch[j] = std::exp(scratch[j] - mx));

        const auto first = static_cast<std::int32_t>(nodes_.size());
        nodes_.resize(nodes_.size() + p.count);  // may reallocate: index, don't hold references
        for (int j = 0; j < p.count; ++j) {
            Node& c = nodes_[first + j];
            c.parent = p.node;
            c.move = pending_moves_[p.offset + j].move;
            c.policy = pending_moves_[p.offset + j].policy;
            c.prior = static_cast<float>(sum > 0 ? scratch[j] / sum : 1.0 / p.count);
        }
        Node& nd = nodes_[p.node];
        nd.first_child = first;
        nd.num_children = static_cast<std::uint16_t>(p.count);
        nd.state = EXPANDED;
        if (p.node == 0) maybe_noise_root();
    }

    Config cfg_;
    std::mt19937_64 rng_;
    chess::Board board_;  // root position; simulate() always restores it
    std::vector<U64> history_keys_;  // game positions before the root, oldest first
    std::vector<Node> nodes_;
    std::vector<Pending> pending_;
    std::vector<PendingMove> pending_moves_;
    std::vector<std::int32_t> collisions_;
    std::vector<chess::Move> path_;
    chess::Movelist moves_;
    GatherStats stats_;
    bool noise_enabled_ = true;
    bool root_noised_ = false;
};

}  // namespace nash
