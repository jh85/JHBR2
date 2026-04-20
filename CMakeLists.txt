cmake_minimum_required(VERSION 3.18)
project(jhbr2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Build type ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

# ---- ONNX Runtime ----
# Download: https://github.com/microsoft/onnxruntime/releases
# Set ONNXRUNTIME_ROOT to the extracted directory, e.g.:
#   cmake -DONNXRUNTIME_ROOT=/home/ei/onnxruntime-linux-x64-gpu-1.24.4 ..
if(NOT ONNXRUNTIME_ROOT)
  # Try common locations
  file(GLOB _ort_candidates
    /home/ei/onnxruntime*
    /opt/onnxruntime*
    /usr/local/onnxruntime*
  )
  list(LENGTH _ort_candidates _ort_len)
  if(_ort_len GREATER 0)
    list(GET _ort_candidates 0 ONNXRUNTIME_ROOT)
  endif()
endif()

if(ONNXRUNTIME_ROOT)
  message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
  # Support both pre-built release layout and from-source build layout
  if(EXISTS ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h)
    # From-source build: headers are in include/onnxruntime/core/session/
    set(ORT_INCLUDE ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
    # Look for the shared library in build/Linux/Release or build/Linux/Debug
    if(EXISTS ${ONNXRUNTIME_ROOT}/build/Linux/Release/libonnxruntime.so)
      set(ORT_LIB ${ONNXRUNTIME_ROOT}/build/Linux/Release)
    elseif(EXISTS ${ONNXRUNTIME_ROOT}/build/Linux/Debug/libonnxruntime.so)
      set(ORT_LIB ${ONNXRUNTIME_ROOT}/build/Linux/Debug)
    else()
      message(FATAL_ERROR "Found ONNX Runtime source tree but no built libraries in build/Linux/")
    endif()
  else()
    # Pre-built release package layout
    set(ORT_INCLUDE ${ONNXRUNTIME_ROOT}/include)
    set(ORT_LIB ${ONNXRUNTIME_ROOT}/lib)
  endif()
else()
  if(NOT USE_TENSORRT)
    message(WARNING "ONNXRUNTIME_ROOT not set. NN evaluator will not build. "
      "Download from: https://github.com/microsoft/onnxruntime/releases")
  endif()
  set(ORT_INCLUDE "")
  set(ORT_LIB "")
endif()

# ---- Sources ----

# Shogi engine (board, bitboard, types, encoder)
set(SHOGI_ENGINE_SRCS
  shogi_engine/board.cc
  shogi_engine/bitboard.cc
  shogi_engine/encoder.cc
)

# MCTS
option(USE_TENSORRT "Use native TensorRT backend (no ONNX Runtime needed)" OFF)

if(USE_TENSORRT)
  set(MCTS_SRCS
    mcts/search.cc
    mcts/nn_tensorrt.cc
  )
else()
  set(MCTS_SRCS
    mcts/search.cc
    mcts/nn_eval.cc
  )
endif()

# Mate solver
set(MATE_SRCS
  mate/dfpn.cc
)

# USI
set(USI_SRCS
  usi/usi_engine.cc
)

# ---- Include directories ----
# The shogi_engine/ headers use #include "shogi/board.h" etc.
# We need to map shogi_engine/ -> shogi/ for includes.
# Solution: add the parent with a symlink, or adjust includes.
# Simplest: use the actual directory as include path.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# For #include "shogi/..." to work, create the mapping
# shogi_engine headers are included as "shogi/board.h" in lc0 style
# We'll fix this by adding an include alias
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/shogi)
  file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/shogi_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/shogi SYMBOLIC)
endif()

# ---- Library: shogi engine core ----
add_library(shogi_core STATIC ${SHOGI_ENGINE_SRCS})
target_include_directories(shogi_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: mate solver ----
add_library(mate_solver STATIC ${MATE_SRCS})
target_link_libraries(mate_solver PUBLIC shogi_core)

# ---- Library: mcts ----
add_library(mcts STATIC ${MCTS_SRCS})
target_link_libraries(mcts PUBLIC shogi_core mate_solver)

if(USE_TENSORRT)
  # Native TensorRT backend.
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(mcts PUBLIC USE_TENSORRT=1)
  target_include_directories(mcts PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(mcts PUBLIC nvinfer CUDA::cudart)
elseif(ORT_INCLUDE)
  # ONNX Runtime backend.
  target_include_directories(mcts PUBLIC ${ORT_INCLUDE})
  target_link_directories(mcts PUBLIC ${ORT_LIB})
  target_link_libraries(mcts PUBLIC onnxruntime)
  target_compile_definitions(mcts PUBLIC HAS_ONNXRUNTIME=1)
endif()

# ---- Executable ----
add_executable(jhbr2 main.cc ${USI_SRCS})
target_link_libraries(jhbr2 PRIVATE mcts mate_solver shogi_core)
if(NOT USE_TENSORRT AND ORT_INCLUDE)
  target_include_directories(jhbr2 PRIVATE ${ORT_INCLUDE})
endif()

# ---- Test: move generation correctness ----
add_executable(test_movegen test/test_movegen.cc)
target_link_libraries(test_movegen PRIVATE shogi_core)

# ---- Test: game replay (DoMove chain) ----
add_executable(test_replay test/test_replay.cc)
target_link_libraries(test_replay PRIVATE shogi_core)

# ---- Benchmark: move generation ----
add_executable(bench_movegen test/bench_movegen.cc)
target_link_libraries(bench_movegen PRIVATE shogi_core)

# ---- Test: mate problem solver ----
add_executable(test_mate test/test_mate.cc)
target_link_libraries(test_mate PRIVATE mate_solver shogi_core)
