// pybind11 module `nash_mcts`: encoder, legal move indexing and the batched PUCT tree.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "encoder.hpp"
#include "mcts.hpp"

namespace py = pybind11;
using nash::kPolicySize;
using nash::kTensorSize;

namespace {

py::array_t<float> to_numpy(const std::vector<float>& buf, py::ssize_t k) {
    py::array_t<float> arr({k, py::ssize_t(8), py::ssize_t(8), py::ssize_t(nash::kPlanes)});
    if (k > 0) std::memcpy(arr.mutable_data(), buf.data(), sizeof(float) * static_cast<size_t>(k) * kTensorSize);
    return arr;
}

py::array_t<float> encode(const std::string& fen) {
    std::vector<float> buf(kTensorSize);
    nash::fill_tensor(nash::encode_input_from_fen(fen), buf.data());
    return to_numpy(buf, 1).reshape({8, 8, nash::kPlanes});
}

py::array_t<float> encode_batch(const std::vector<std::string>& fens) {
    const auto k = static_cast<py::ssize_t>(fens.size());
    py::array_t<float> arr({k, py::ssize_t(8), py::ssize_t(8), py::ssize_t(nash::kPlanes)});
    float* out = arr.mutable_data();
    for (py::ssize_t i = 0; i < k; ++i) nash::fill_tensor(nash::encode_input_from_fen(fens[i]), out + i * kTensorSize);
    return arr;
}

std::vector<std::pair<std::string, int>> legal_moves(const std::string& fen) {
    const chess::Board b = nash::make_board(fen);
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, b);
    const bool white = b.sideToMove() == chess::Color::WHITE;
    std::vector<std::pair<std::string, int>> out;
    out.reserve(moves.size());
    for (const auto& m : moves) out.emplace_back(nash::move_uci(m), nash::policy_index(m, white));
    return out;
}

const char* proven_name(std::int8_t p) {
    switch (p) {
        case nash::WIN: return "win";
        case nash::DRAW: return "draw";
        case nash::LOSS: return "loss";
        default: return "unknown";
    }
}

}  // namespace

PYBIND11_MODULE(nash_mcts, m) {
    m.doc() = "C++ PUCT MCTS for the Nash-Eval chess network (input encoding, move indexing, tree search).";
    m.attr("POLICY_SIZE") = kPolicySize;
    m.attr("INPUT_SHAPE") = py::make_tuple(8, 8, nash::kPlanes);

    m.def("encode", &encode, py::arg("fen"),
          "(8, 8, 36) float32 encoding, identical to core.chess_environment.fast_fen_to_example(fen).");
    m.def("encode_batch", &encode_batch, py::arg("fens"), "Stacked encode() of several FENs: (k, 8, 8, 36).");
    m.def("legal_moves", &legal_moves, py::arg("fen"),
          "Legal moves as (uci, policy_index) with core.move_encoding.move_to_index indices.");

    py::class_<nash::MCTS>(m, "MCTS")
        .def(py::init([](float c_puct, float fpu_reduction, float dirichlet_alpha, float dirichlet_epsilon,
                         int virtual_loss, std::uint64_t seed) {
                 nash::Config cfg;
                 cfg.c_puct = c_puct;
                 cfg.fpu_reduction = fpu_reduction;
                 cfg.dirichlet_alpha = dirichlet_alpha;
                 cfg.dirichlet_epsilon = dirichlet_epsilon;
                 cfg.virtual_loss = virtual_loss;
                 cfg.seed = seed;
                 return new nash::MCTS(cfg);
             }),
             py::arg("c_puct") = 2.0f, py::arg("fpu_reduction") = 0.25f, py::arg("dirichlet_alpha") = 0.3f,
             py::arg("dirichlet_epsilon") = 0.25f, py::arg("virtual_loss") = 1, py::arg("seed") = 0)
        .def("reset", &nash::MCTS::reset, py::arg("fen"), py::arg("history_uci") = std::vector<std::string>{},
             py::arg("add_noise") = true,
             "Set the root to `fen` after playing `history_uci` from it. add_noise=False disables root "
             "Dirichlet noise (evaluation games).")
        .def(
            "gather",
            [](nash::MCTS& t, int n) {
                std::vector<float> buf;
                int k;
                {
                    py::gil_scoped_release release;
                    k = t.gather(n, buf);
                }
                return to_numpy(buf, k);
            },
            py::arg("n"),
            "Run up to n selections; returns (k, 8, 8, 36) encodings of the k distinct leaves needing the network.")
        .def(
            "backup",
            [](nash::MCTS& t, py::array_t<float, py::array::c_style | py::array::forcecast> policy,
               py::array_t<float, py::array::c_style | py::array::forcecast> wdl) {
                const int k = t.num_pending();
                if (policy.ndim() != 2 || policy.shape(0) != k || policy.shape(1) != kPolicySize)
                    throw std::invalid_argument("policy_logits must have shape (" + std::to_string(k) + ", 4672)");
                if (wdl.ndim() != 2 || wdl.shape(0) != k || wdl.shape(1) != 3)
                    throw std::invalid_argument("wdl must have shape (" + std::to_string(k) + ", 3)");
                const float* p = policy.data();
                const float* v = wdl.data();
                py::gil_scoped_release release;
                t.backup(p, v, k);
            },
            py::arg("policy_logits"), py::arg("wdl"))
        .def("cancel", &nash::MCTS::cancel, "Discard the pending batch and its virtual loss.")
        .def("root_stats",
             [](const nash::MCTS& t) {
                 std::vector<std::tuple<std::string, int, int, double, float>> out;
                 for (const auto& s : t.root_stats()) out.emplace_back(s.uci, s.policy, s.visits, s.q, s.prior);
                 return out;
             },
             "(uci, policy_index, visits, Q, prior) per root move. Q is the expected score for the root's side "
             "to move (NaN if unvisited; exact 1/0.5/0 if proven).")
        .def("root_move_proven",
             [](const nash::MCTS& t) {
                 std::vector<std::string> out;
                 for (auto p : t.root_child_proven()) out.emplace_back(proven_name(p));
                 return out;
             },
             "Per root move (root_stats() order): 'win', 'draw', 'loss' or 'unknown' for the side to move at "
             "the root after playing it.")
        .def("root_visits", &nash::MCTS::root_visits)
        .def("root_value", &nash::MCTS::root_value, "Expected score W + D/2 for the side to move at the root.")
        .def("root_proven", [](const nash::MCTS& t) { return std::string(proven_name(t.root_proven())); },
             "'win', 'draw', 'loss' or 'unknown' for the side to move at the root.")
        .def("root_terminal", &nash::MCTS::root_terminal, "True if the root has no legal moves.")
        .def("root_fen", &nash::MCTS::root_fen)
        .def(
            "select_move",
            [](nash::MCTS& t, double temperature) { return t.select_move(temperature); },
            py::arg("temperature") = 0.0)
        .def("advance", &nash::MCTS::advance, py::arg("uci"))
        .def("pending_paths", &nash::MCTS::pending_paths, "UCI paths from the root to each pending leaf.")
        .def("num_pending", &nash::MCTS::num_pending)
        .def("total_virtual_loss", &nash::MCTS::total_virtual_loss)
        .def("num_nodes", &nash::MCTS::num_nodes)
        .def("last_gather_stats",
             [](const nash::MCTS& t) {
                 const auto& s = t.last_gather();
                 py::dict d;
                 d["selections"] = s.selections;
                 d["leaves"] = s.leaves;
                 d["terminals"] = s.terminals;
                 d["collisions"] = s.collisions;
                 return d;
             })
        .def_property("noise_enabled", &nash::MCTS::noise_enabled, &nash::MCTS::set_noise_enabled);
}
