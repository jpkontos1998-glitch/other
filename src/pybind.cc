#include <optional>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/extension.h>

#include "src/env/env_state.h"
#include "src/env/inits/jb_boards.h"
#include "src/env/boardstate_channels.h"
#include "src/env/stratego.h"
#include "src/env/stratego_board.h"
#include "src/env/stratego_conf.h"
#include "src/util.h"

namespace py = pybind11;

#if PY_MAJOR_VERSION < 3
#error Only Python 3 is supported
#endif

namespace
{
#define ATTEMPT_CAST(FIELD, ...)                        \
    {                                                   \
        if (key == #FIELD)                              \
        {                                               \
            conf.FIELD = kv.second.cast<__VA_ARGS__>(); \
            continue;                                   \
        }                                               \
    }

    // Constructs a StrategoConf from a py::dict object by parsing the keys
    StrategoConf ParseStrategoConf(const py::dict &dict)
    {
        StrategoConf conf{};
        for (const auto &kv : dict)
        {
            const std::string key = kv.first.cast<std::string>();
            ATTEMPT_CAST(max_num_moves, uint32_t);
            ATTEMPT_CAST(max_num_moves_between_attacks, uint32_t);
            ATTEMPT_CAST(move_memory, uint32_t);
            ATTEMPT_CAST(reset_behavior, ResetBehavior);
            ATTEMPT_CAST(reset_state, std::optional<EnvState>);
            ATTEMPT_CAST(initial_arrangements, std::optional<std::pair<StringArrangements, StringArrangements>>);
            ATTEMPT_CAST(initial_arrangements_distrib, std::optional<std::pair<torch::Tensor, torch::Tensor>>);
            ATTEMPT_CAST(verbose, bool);
            ATTEMPT_CAST(two_square_rule, bool);
            ATTEMPT_CAST(continuous_chasing_rule, bool);
            ATTEMPT_CAST(enable_src_dst_planes, bool);
            ATTEMPT_CAST(enable_hidden_and_types_planes, bool);
            ATTEMPT_CAST(enable_dm_planes, bool);
            ATTEMPT_CAST(cuda_device, int32_t);
            ATTEMPT_CAST(quiet, uint8_t);
            ATTEMPT_CAST(nonsteppable, bool);
            MUSTRATEGO_FATAL("Unrecognized configuration key %s", key.c_str());
        }
        return conf;
    }

#undef ATTEMPT_CAST
} // namespace

namespace PYBIND11_NAMESPACE
{
    namespace detail
    {
        template <>
        struct type_caster<uint128_t>
        {
        public:
            PYBIND11_TYPE_CASTER(uint128_t, const_name("uint128_t"));

            bool load(handle src, bool)
            {
                // Extract PyObject from handle
                PyObject *obj = src.ptr();

                if (PyLong_Check(obj))
                {
                    auto mask = PyLong_FromUnsignedLongLong(0xFFFFFFFFFFFFFFFF);
                    auto val64 = PyLong_FromLong(64);
                    auto high_bits_obj = PyNumber_Rshift(obj, val64);
                    auto low_bits_obj = PyNumber_And(obj, mask);

                    auto lo = PyLong_AsUnsignedLongLong(low_bits_obj);
                    auto hi = PyLong_AsUnsignedLongLong(high_bits_obj);
                    value = uint128_t(hi);
                    value <<= 64;
                    value += lo;

                    Py_DECREF(low_bits_obj);
                    Py_DECREF(high_bits_obj);
                    Py_DECREF(val64);
                    Py_DECREF(mask);
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "expecting int");
                    return false;
                }
                return !PyErr_Occurred();
            }

            static handle cast(uint128_t src, return_value_policy /* policy */, handle /* parent */)
            {
                auto src_hi = PyLong_FromUnsignedLongLong((uint64_t)(src >> 64));
                auto val64 = PyLong_FromLong(64);
                auto hi = PyNumber_Lshift(src_hi, val64);
                auto lo = PyLong_FromUnsignedLongLong((uint64_t)src);
                auto ans = PyNumber_Add(hi, lo);

                Py_DECREF(lo);
                Py_DECREF(hi);
                Py_DECREF(val64);
                Py_DECREF(src_hi);
                return ans;
            }
        };
    }
} // namespace PYBIND11_NAMESPACE::detail

PYBIND11_MODULE(pystratego, m)
{
    py::enum_<ResetBehavior>(m, "ResetBehavior")
        .value("CUSTOM_ENV_STATE", ResetBehavior::CUSTOM_ENV_STATE)
        .value("RANDOM_CUSTOM_INITIAL_ARRANGEMENT", ResetBehavior::RANDOM_CUSTOM_INITIAL_ARRANGEMENT)
        .value("STEP_CUSTOM_INITIAL_ARRANGEMENT", ResetBehavior::STEP_CUSTOM_INITIAL_ARRANGEMENT)
        .value("FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT", ResetBehavior::FULLINFO_RANDOM_CUSTOM_INITIAL_ARRANGEMENT)
        .value("FULLINFO_STEP_CUSTOM_INITIAL_ARRANGEMENT", ResetBehavior::FULLINFO_STEP_CUSTOM_INITIAL_ARRANGEMENT)
        .value("RANDOM_JB_CLASSIC_BOARD", ResetBehavior::RANDOM_JB_CLASSIC_BOARD)
        .value("RANDOM_JB_BARRAGE_BOARD", ResetBehavior::RANDOM_JB_BARRAGE_BOARD)
        .value("FULLINFO_RANDOM_JB_CLASSIC_BOARD", ResetBehavior::FULLINFO_RANDOM_JB_CLASSIC_BOARD)
        .value("FULLINFO_RANDOM_JB_BARRAGE_BOARD", ResetBehavior::FULLINFO_RANDOM_JB_BARRAGE_BOARD);

    py::enum_<BoardVariant>(m, "BoardVariant")
        .value("CLASSIC", BoardVariant::CLASSIC)
        .value("BARRAGE", BoardVariant::BARRAGE);

    py::class_<EnvState::TensorChaseState>(m, "ChaseState")
        .def_readonly("last_src_pos", &EnvState::TensorChaseState::last_src_pos)
        .def_readonly("last_dst_pos", &EnvState::TensorChaseState::last_dst_pos)
        .def_readonly("chase_length", &EnvState::TensorChaseState::chase_length);

    py::class_<EnvState>(m, "EnvState")
        .def_readonly("num_envs", &EnvState::num_envs)
        .def_readonly("to_play", &EnvState::to_play)
        .def_readonly("boards", &EnvState::boards)
        .def_readonly("zero_boards", &EnvState::zero_boards)
        .def_readonly("num_moves", &EnvState::num_moves)
        .def_readonly("num_moves_since_last_attack", &EnvState::num_moves_since_last_attack)
        .def_readonly("terminated_since", &EnvState::terminated_since)
        .def_readonly("has_legal_movement", &EnvState::has_legal_movement)
        .def_readonly("flag_captured", &EnvState::flag_captured)
        .def_readonly("action_history", &EnvState::action_history)
        .def_readonly("board_history", &EnvState::board_history)
        .def_readonly("move_summary_history", &EnvState::move_summary_history)
        .def_readonly("chase_state", &EnvState::chase_state)
        .def("replicate_env", &EnvState::ReplicateEnv)
        .def("tile", &EnvState::Tile)
        .def("board_strs", &EnvState::BoardStrs)
        .def("zero_board_strs", &EnvState::ZeroBoardStrs)
        .def("cat", &EnvState::Cat)
        .def("clone", &EnvState::Clone)
        .def("slice", &EnvState::Slice);

    py::class_<StrategoConf>(m, "StrategoConf")
        .def(py::init<>())
        .def("__init__", [](StrategoConf &instance, const py::kwargs &kwargs)
             { new (&instance) StrategoConf(ParseStrategoConf(kwargs)); })
        .def_readwrite("max_num_moves", &StrategoConf::max_num_moves)
        .def_readwrite("max_num_moves_between_attacks", &StrategoConf::max_num_moves_between_attacks)
        .def_readwrite("move_memory", &StrategoConf::move_memory)
        .def_readwrite("reset_behavior", &StrategoConf::reset_behavior)
        .def_readwrite("reset_state", &StrategoConf::reset_state)
        .def_readwrite("initial_arrangements", &StrategoConf::initial_arrangements)
        .def_readwrite("verbose", &StrategoConf::verbose)
        .def_readwrite("two_square_rule", &StrategoConf::two_square_rule)
        .def_readwrite("continuous_chasing_rule", &StrategoConf::continuous_chasing_rule)
        .def_readwrite("enable_src_dst_planes", &StrategoConf::enable_src_dst_planes)
        .def_readwrite("enable_hidden_and_types_planes", &StrategoConf::enable_hidden_and_types_planes)
        .def_readwrite("enable_dm_planes", &StrategoConf::enable_dm_planes)
        .def_readwrite("cuda_device", &StrategoConf::cuda_device);

    py::class_<StrategoRolloutBuffer>(m, "StrategoRolloutBuffer")
        .def(py::init<uint32_t, uint32_t>())               // Construct with default configuration
        .def(py::init<uint32_t, uint32_t, StrategoConf>()) // Construct with explicit configuration
        .def("__init__",                                   // Custom constructor with kwargs
             [](StrategoRolloutBuffer &instance, uint32_t buf_size, uint32_t num_envs, const py::kwargs &kwargs)
             { new (&instance) StrategoRolloutBuffer(buf_size, num_envs, ParseStrategoConf(kwargs)); })
        .def("save_games", &StrategoRolloutBuffer::SaveGames)
        .def("stop_saving_games", &StrategoRolloutBuffer::StopSavingGames)
        .def("change_reset_behavior",
             py::overload_cast<const ResetBehavior,
                               const std::optional<std::pair<torch::Tensor, torch::Tensor>> &>(&StrategoRolloutBuffer::ChangeResetBehavior),
             py::arg("reset_behavior"),
             py::arg("initial_arrangement_distrib") = std::nullopt)
        .def("change_reset_behavior", py::overload_cast<const EnvState &>(&StrategoRolloutBuffer::ChangeResetBehavior), py::arg("env_state"))
        .def("change_reset_behavior",
             py::overload_cast<const std::pair<StringArrangements, StringArrangements> &,
                               const std::optional<std::pair<torch::Tensor, torch::Tensor>> &,
                               const bool,
                               const bool>(&StrategoRolloutBuffer::ChangeResetBehavior),
             py::arg("initial_arrangements"),
             py::arg("initial_arrangements_distrib") = std::nullopt,
             py::arg("randomize") = true,
             py::arg("fullinfo") = false)
        .def("reset", &StrategoRolloutBuffer::Reset)
        .def("current_step", &StrategoRolloutBuffer::CurrentStep)
        .def("current_player", &StrategoRolloutBuffer::CurrentPlayer)
        .def("acting_player", &StrategoRolloutBuffer::ActingPlayer)
        .def("apply_actions", &StrategoRolloutBuffer::ApplyActions)
        .def("compute_legal_action_mask", &StrategoRolloutBuffer::ComputeLegalActionMask)
        .def("compute_infostate_tensor", &StrategoRolloutBuffer::ComputeInfostateTensor)
        .def("compute_reward_pl0", &StrategoRolloutBuffer::ComputeRewardPl0)
        .def("compute_is_unknown_piece", &StrategoRolloutBuffer::ComputeIsUnknownPiece)
        .def("compute_piece_type_onehot", &StrategoRolloutBuffer::ComputePieceTypeOnehot)
        .def("compute_two_square_rule_applies", &StrategoRolloutBuffer::ComputeTwoSquareRuleApplies)
        .def("compute_unknown_piece_type_onehot", &StrategoRolloutBuffer::ComputeUnknownPieceTypeOnehot)
        .def("compute_unknown_piece_has_moved", &StrategoRolloutBuffer::ComputeUnknownPieceHasMoved)
        .def("compute_unknown_piece_position_onehot", &StrategoRolloutBuffer::ComputeUnknownPiecePositionOnehot)
        .def("get_terminated_since", &StrategoRolloutBuffer::GetTerminatedSince)
        .def("get_has_legal_movement", &StrategoRolloutBuffer::GetHasLegalMovement)
        .def("get_flag_captured", &StrategoRolloutBuffer::GetFlagCaptured)
        .def("get_num_moves", &StrategoRolloutBuffer::GetNumMoves)
        .def("get_num_moves_since_last_attack", &StrategoRolloutBuffer::GetNumMovesSinceLastAttack)
        .def("get_num_moves_since_reset", &StrategoRolloutBuffer::GetNumMovesSinceReset)
        .def("get_twosquare_state", &StrategoRolloutBuffer::GetTwosquareState)
        .def("get_illegal_chase_actions", &StrategoRolloutBuffer::GetIllegalChaseActions)
        .def("get_played_actions", &StrategoRolloutBuffer::GetPlayedActions)
        .def("get_move_summary", &StrategoRolloutBuffer::GetMoveSummary)
        .def("get_board_tensor", py::overload_cast<const uint64_t>(&StrategoRolloutBuffer::GetBoardTensor, py::const_))
        .def("get_zero_board_tensor", py::overload_cast<const uint64_t>(&StrategoRolloutBuffer::GetZeroBoardTensor, py::const_))
        .def("snapshot_env_history", &StrategoRolloutBuffer::SnapshotEnvHistory)
        .def("snapshot_state", &StrategoRolloutBuffer::SnapshotState)
        .def("board_strs", &StrategoRolloutBuffer::BoardStrs)
        .def("zero_board_strs", &StrategoRolloutBuffer::ZeroBoardStrs)
        .def("seed_action_sampler", &StrategoRolloutBuffer::SeedActionSampler)
        .def("sample_random_legal_action", &StrategoRolloutBuffer::SampleRandomLegalAction)
        .def("sample_first_legal_action", &StrategoRolloutBuffer::SampleFirstLegalAction)
        .def_readonly("buf_size", &StrategoRolloutBuffer::buf_size)
        .def_readonly("num_envs", &StrategoRolloutBuffer::num_envs)
        .def_readonly("legal_action_mask", &StrategoRolloutBuffer::legal_action_mask)
        .def_readonly("board_state_tensor", &StrategoRolloutBuffer::board_state_tensor)
        .def_readonly("infostate_tensor", &StrategoRolloutBuffer::infostate_tensor)
        .def_readonly("reward_pl0", &StrategoRolloutBuffer::reward_pl0)
        .def_readonly("is_unknown_piece", &StrategoRolloutBuffer::is_unknown_piece)
        .def_readonly("piece_type_onehot", &StrategoRolloutBuffer::piece_type_onehot)
        .def_readonly("two_square_rule_applies", &StrategoRolloutBuffer::two_square_rule_applies)
        .def_readonly("unknown_piece_type_onehot", &StrategoRolloutBuffer::unknown_piece_type_onehot)
        .def_readonly("unknown_piece_has_moved", &StrategoRolloutBuffer::unknown_piece_has_moved)
        .def_readonly("unknown_piece_position_onehot", &StrategoRolloutBuffer::unknown_piece_position_onehot)
        .def_readonly("conf", &StrategoRolloutBuffer::conf)
        .def_readonly("NUM_INFOSTATE_CHANNELS", &StrategoRolloutBuffer::NUM_INFOSTATE_CHANNELS)
        .def_readonly("INFOSTATE_CHANNEL_DESCRIPTION", &StrategoRolloutBuffer::INFOSTATE_CHANNEL_DESCRIPTION);

    py::class_<PieceArrangementGenerator>(m, "PieceArrangementGenerator")
        .def(py::init<const std::array<uint8_t, NUM_PIECE_TYPES> &>())
        .def(py::init<BoardVariant>())
        .def("generate_string_arrangements", &PieceArrangementGenerator::GenerateStringArrangements)
        .def("generate_arrangements", &PieceArrangementGenerator::GenerateArrangements)
        .def("arrangement_ids", &PieceArrangementGenerator::ArrangementIds)
        .def("num_possible_arrangements", &PieceArrangementGenerator::NumPossibleArrangements);

    {
        py::module_ util = m.def_submodule("util", "Collection of utility functions");
        util.def("arrangement_tensor_from_strings", &ArrangementTensorFromStrings);
        util.def("is_terminal_arrangement", &IsTerminalArrangement);
        util.def("arrangement_strings_from_tensor", &ArrangementStringsFromTensor);
        util.def("actions_to_abs_coordinates", &ActionsToAbsCoordinates, py::arg("actions"), py::arg("player"));
        util.def("abs_coordinates_to_actions", &AbsCoordinatesToActions, py::arg("movements"), py::arg("player"));
        util.def("generate_initialization_boards", &GenerateInitializationBoards, py::arg("red_arrangements"), py::arg("blue_arrangements"), py::arg("cuda_device") = 0);
        util.def("initialization_board_from_string", &InitializationBoardFromString, py::arg("s"), py::arg("cuda_device") = 0);
        util.def("assign_opponent_hidden_pieces", &AssignOpponentHiddenPieces);
        util.def("legacy_assign_opponent_hidden_pieces", &LegacyAssignOpponentHiddenPieces);
    }

    m.attr("JB_INIT_BOARDS_BARRAGE") = JB_INIT_BOARDS_BARRAGE;
    m.attr("JB_INIT_BOARDS_CLASSIC") = JB_INIT_BOARDS_CLASSIC;
    m.attr("NUM_ACTIONS") = py::int_(NUM_ACTIONS);
    m.attr("NUM_PIECE_TYPES") = py::int_(NUM_PIECE_TYPES);
    m.attr("NUM_BOARD_STATE_CHANNELS") = py::int_(NUM_BOARD_STATE_CHANNELS);
    m.attr("BOARDSTATE_CHANNEL_DESCRIPTION") = BOARDSTATE_CHANNEL_DESCRIPTIONS;
}
