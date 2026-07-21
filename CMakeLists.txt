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
  shogi/board.cc
  shogi/bitboard.cc
  shogi/encoder.cc
)

# MCTS (dlshogi-style search + NN backend)
option(USE_TENSORRT "Use native TensorRT backend (no ONNX Runtime needed)" OFF)

set(DLSHOGI_MCTS_SRCS
  dlshogi_mcts/uct_node.cc
  dlshogi_mcts/uct_search.cc
)

if(USE_TENSORRT)
  # The native TRT backend uses a CUDA kernel to expand bit-packed input
  # features on the GPU (dlshogi-style). Enable the CUDA language before the
  # mcts target consumes the .cu source.
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
  endif()
  set(MCTS_SRCS
    ${DLSHOGI_MCTS_SRCS}
    mcts/nn_tensorrt.cc
    shogi/encoder_unpack.cu
  )
else()
  set(MCTS_SRCS
    ${DLSHOGI_MCTS_SRCS}
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
# The shogi/ headers use #include "shogi/board.h" — directory and
# include path now match, no shim needed.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: shogi engine core ----
add_library(shogi_core STATIC ${SHOGI_ENGINE_SRCS})
target_include_directories(shogi_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: opening book ----
add_library(opening_book STATIC book/opening_book.cc)
target_include_directories(opening_book PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: pure move-time calculation ----
add_library(time_manager STATIC usi/time_manager.cc)
target_include_directories(time_manager PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: mate solver ----
add_library(mate_solver STATIC ${MATE_SRCS})
target_link_libraries(mate_solver PUBLIC shogi_core)

# ---- Library: mcts ----
add_library(mcts STATIC ${MCTS_SRCS})
target_link_libraries(mcts PUBLIC shogi_core mate_solver)

if(USE_TENSORRT)
  # Native TensorRT backend.
  find_package(CUDAToolkit REQUIRED)
  set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
  set(CUDNN_ROOT "" CACHE PATH "cuDNN installation root")
  if(NOT TENSORRT_ROOT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
    find_library(TENSORRT_NVINFER nvinfer)
  else()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
      PATHS ${TENSORRT_ROOT}/include NO_DEFAULT_PATH)
    find_library(TENSORRT_NVINFER nvinfer
      PATHS ${TENSORRT_ROOT}/lib NO_DEFAULT_PATH)
  endif()
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=/path/to/TensorRT")
  endif()
  message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
  message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
  target_compile_definitions(mcts PUBLIC USE_TENSORRT=1)
  target_include_directories(mcts PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
  )
  target_link_libraries(mcts PUBLIC ${TENSORRT_NVINFER} CUDA::cudart)
  if(CUDNN_ROOT)
    target_include_directories(mcts PUBLIC ${CUDNN_ROOT}/include)
    target_link_directories(mcts PUBLIC ${CUDNN_ROOT}/lib)
    set_target_properties(mcts PROPERTIES
      BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDNN_ROOT}/lib;${CUDAToolkit_LIBRARY_DIR}"
    )
  else()
    set_target_properties(mcts PROPERTIES
      BUILD_RPATH "${TENSORRT_ROOT}/lib;${CUDAToolkit_LIBRARY_DIR}"
    )
  endif()
elseif(ORT_INCLUDE)
  # ONNX Runtime backend.
  target_include_directories(mcts PUBLIC ${ORT_INCLUDE})
  target_link_directories(mcts PUBLIC ${ORT_LIB})
  target_link_libraries(mcts PUBLIC onnxruntime)
  target_compile_definitions(mcts PUBLIC HAS_ONNXRUNTIME=1)
endif()

# ---- Executable ----
add_executable(jhbr2 main.cc ${USI_SRCS})
target_link_libraries(jhbr2 PRIVATE mcts mate_solver opening_book time_manager
                                    shogi_core)
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

# ---- Benchmark: legal move generation from SFEN file ----
add_executable(bench_legal100 test/bench_legal100.cc)
target_link_libraries(bench_legal100 PRIVATE shogi_core)

# ---- Benchmark: encoder ----
add_executable(bench_encoder test/bench_encoder.cc)
target_link_libraries(bench_encoder PRIVATE shogi_core)

# ---- Test: mate problem solver ----
add_executable(test_mate test/test_mate.cc)
target_link_libraries(test_mate PRIVATE mate_solver shogi_core)

# ---- Test: shallow mate (Phase 0+) ----
add_executable(test_shallow_mate test/test_shallow_mate.cc)
target_link_libraries(test_shallow_mate PRIVATE mate_solver shogi_core)

# ---- Bench: shallow mate vs df-pn ----
add_executable(bench_shallow_mate test/bench_shallow_mate.cc)
target_link_libraries(bench_shallow_mate PRIVATE mate_solver shogi_core)

# ---- Bench: mate-in-1 oracle vs specialized implementation ----
add_executable(bench_mate1 test/bench_mate1.cc)
target_link_libraries(bench_mate1 PRIVATE shogi_core)

# ---- Test: specialized mate-in-1 vs move-generation oracle ----
add_executable(test_mate1 test/test_mate1.cc)
target_link_libraries(test_mate1 PRIVATE shogi_core)

# ---- Test: GenerateCheckingMoves property test ----
add_executable(test_check_movegen test/test_check_movegen.cc)
target_link_libraries(test_check_movegen PRIVATE shogi_core)

# ---- Test: opening book preload and on-the-fly lookup ----
add_executable(test_opening_book test/test_opening_book.cc)
target_link_libraries(test_opening_book PRIVATE opening_book)

# ---- Test: pure move-time calculation ----
add_executable(test_time_manager test/test_time_manager.cc)
target_link_libraries(test_time_manager PRIVATE time_manager)
