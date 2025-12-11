cmake_minimum_required(VERSION 3.25)
set(USE_CUDNN 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(stratego LANGUAGES CUDA CXX)

# Disable response files since they don't play it nicely with `clangd`
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD 17)
set(COMPILATION_FLAGS "-fPIC -g -O3 -fopenmp -Wall -Wextra -Wfatal-errors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS} ${COMPILATION_FLAGS}")

# Set pybind flags of pytorch
execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/get_pybind_flags.py
  OUTPUT_VARIABLE PYBIND_FLAGS
)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)


# find Pytorch
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)

# Torch requires Caffe2, and Caffe2 destroys any sane modern cmake CUDA setup.
# So, we need to undo all the work that was done on the nvcc flags.
set(CUDA_NVCC_FLAGS " ")
# Similarly, we reset the CUDA_FLAGS here to undo the mess that Caffe2 makes
set(CMAKE_CUDA_FLAGS "${COMPILATION_FLAGS}")

set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")

message("---------------------")
message(${TorchPath})
message(${PYBIND_FLAGS})
message(${CMAKE_CXX_FLAGS})
message(${TORCH_INCLUDE_DIRS})
message("---------------------")

# === New libraries ===

# init boards
add_library(jb_init_boards
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/inits/jb_init_boards_barrage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/inits/jb_init_boards_full.cpp
)
target_include_directories(jb_init_boards PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# util
add_library(stratego_util
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cu
)
target_include_directories(stratego_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(stratego_util PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(stratego_util PUBLIC ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})

# CUDA kernels
add_library(
  cuda_kernels
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/cuda/action_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/cuda/board_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/cuda/infostate_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/cuda/snapshot_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/cuda/termination_kernels.cu
)
target_include_directories(cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(cuda_kernels PUBLIC util)

# twosquare rule
add_library(twosquare_state
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/rules/twosquare_state.cu
)
target_include_directories(twosquare_state PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(twosquare_state PUBLIC ${TORCH_INCLUDE_DIRS})

# continuous chasing rule
add_library(chase_state
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/rules/chase_state.cu
)
target_include_directories(chase_state PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(chase_state PUBLIC ${TORCH_INCLUDE_DIRS})

# game saver
add_library(game_saver
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/game_saver.cu
)
target_include_directories(game_saver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(game_saver PUBLIC ${TORCH_INCLUDE_DIRS})

# compile library
add_library(cu_stratego
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/stratego_board.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/env_state.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/stratego.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/env/stratego_conf.cu
)
target_include_directories(cu_stratego PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cu_stratego PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(cu_stratego PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(cu_stratego PUBLIC
  cuda_kernels
  jb_init_boards
  twosquare_state
  chase_state
  game_saver
  stratego_util
  ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

pybind11_add_module(pystratego ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind.cc)
target_link_libraries(pystratego PRIVATE cu_stratego)

# testing
enable_testing()
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
add_test(
  NAME test_tensor_size
  COMMAND python ../tests/test_tensor_size.py -v
)
add_test(
  NAME test_action_transforms
  COMMAND python ../tests/test_action_transforms.py -v
)
add_test(
  NAME test_alt_log_loss
  COMMAND python ../tests/test_alt_log_loss.py -v
)
add_test(
  NAME test_arrangement_distribution_behavior
  COMMAND python ../tests/test_arrangement_distribution_behavior.py -v
)
add_test(
  NAME test_attack_planes
  COMMAND python ../tests/test_attack_planes.py -v
)
add_test(
  NAME test_battle_planes
  COMMAND python ../tests/test_battle_planes.py -v
)
add_test(
  NAME test_belief_mask
  COMMAND python ../tests/test_belief_mask.py -v
)
add_test(
  NAME test_chan_last
  COMMAND python ../tests/test_chan_last.py -v
)
add_test(
  NAME test_circular_buffer
  COMMAND python ../tests/test_circular_buffer.py -v
)
add_test(
  NAME test_cemetery
  COMMAND python ../tests/test_cemetery.py -v
)
add_test(
  NAME test_continuous_chase
  COMMAND python ../tests/test_continuous_chase.py -v
)
add_test(
  NAME test_continuous_chase_new
  COMMAND python ../tests/test_continuous_chase_new.py -v
)
add_test(
  NAME test_cpu_mem_leak
  COMMAND python ../tests/test_cpu_mem_leak.py -v
)
add_test(
  NAME test_dm_planes
  COMMAND python ../tests/test_dm_planes.py -v
)
add_test(
  NAME test_env_state
  COMMAND python ../tests/test_env_state.py -v
)
add_test(
  NAME test_env
  COMMAND python ../tests/test_env.py -v
)
add_test(
  NAME test_far_action_history
  COMMAND python ../tests/test_far_action_history.py -v
)
add_test(
  NAME test_gen_methods
  COMMAND python ../tests/test_gen_methods.py -v
)
add_test(
  NAME test_infostate
  COMMAND python ../tests/test_infostate.py -v
)
add_test(
  NAME test_initial_board_memory
  COMMAND python ../tests/test_initial_board_memory.py -v
)
add_test(
  NAME test_instantiate
  COMMAND python ../tests/test_instantiate.py -v
)
add_test(
  NAME test_is_terminal_arrangement
  COMMAND python ../tests/test_is_terminal_arrangement.py -v
)
add_test(
  NAME test_last_attack
  COMMAND python ../tests/test_last_attack.py -v
)
add_test(
  NAME test_last_moves_encoder
  COMMAND python ../tests/test_last_moves_encoder.py -v
)
add_test(
  NAME test_legal_action_consistency
  COMMAND python ../tests/test_legal_action_consistency.py -v
)
add_test(
  NAME test_log_prob_compression
  COMMAND python ../tests/test_log_prob_compression.py -v
)
add_test(
  NAME test_magnet_policy
  COMMAND python ../tests/test_magnet_policy.py -v
)
add_test(
  NAME test_move_summary
  COMMAND python ../tests/test_move_summary.py -v
)
add_test(
  NAME test_new_planes
  COMMAND python ../tests/test_new_planes.py -v
)
add_test(
  NAME test_piece_arrangement_generator
  COMMAND python ../tests/test_piece_arrangement_generator.py -v
)
add_test(
  NAME test_piece_id
  COMMAND python ../tests/test_piece_id.py -v
)
add_test(
  NAME test_process_data
  COMMAND python ../tests/test_process_data.py -v
)
add_test(
  NAME test_process_arr_data
  COMMAND python ../tests/test_process_arr_data.py -v
)
add_test(
  NAME test_protect_planes
  COMMAND python ../tests/test_protect_planes.py -v
)
add_test(
  NAME test_replay_games
  COMMAND python ../tests/test_replay_games.py -v
)
add_test(
  NAME test_reset_behavior
  COMMAND python ../tests/test_reset_behavior.py -v
)
add_test(
  NAME test_rewards
  COMMAND python ../tests/test_rewards.py -v
)
add_test(
  NAME test_samplers
  COMMAND python ../tests/test_samplers.py -v
)
add_test(
  NAME test_save_games
  COMMAND python ../tests/test_save_games.py -v
)
add_test(
  NAME test_search
  COMMAND python ../tests/test_search.py -v
)
add_test(
  NAME test_snapshot_env_history
  COMMAND python ../tests/test_snapshot_env_history.py -v
)
add_test(
  NAME test_transformer_action_reduction
  COMMAND python ../tests/test_transformer_action_reduction.py -v
)
add_test(
  NAME test_twosquare_rule
  COMMAND python ../tests/test_twosquare_rule.py -v
)
add_test(
  NAME test_unknown_piece_counts
  COMMAND python ../tests/test_unknown_piece_counts.py -v
)
add_test(
  NAME test_uniform_belief
  COMMAND python ../tests/test_uniform_belief.py -v
)
