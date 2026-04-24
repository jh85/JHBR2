/*
  JHBR2 Shogi Engine — Neural Network Evaluator (ONNX Runtime)

  Input:  (batch, C, 9, 9) float32
  Output: policy (batch, 2187), wdl (batch, 3), mlh (batch, 1)

  Execution provider priority:
    1. TensorRT (FP16, dynamic batch, fastest)
    2. CUDA (FP32, uses multi-ONNX-file batching)
    3. CPU (fallback)
*/

#include "mcts/nn_eval.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

#if HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include "shogi/encoder.h"

namespace jhbr2 {

using namespace lczero;

// =====================================================================
// Softmax helper
// =====================================================================

static void Softmax(float* data, int size) {
  float max_val = *std::max_element(data, data + size);
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    data[i] = std::exp(data[i] - max_val);
    sum += data[i];
  }
  if (sum > 0.0f) {
    for (int i = 0; i < size; i++) data[i] /= sum;
  }
}

// =====================================================================
// Implementation
// =====================================================================

#if HAS_ONNXRUNTIME

// Fixed batch sizes for the multi-ONNX-file fallback (CUDA without TensorRT).
static constexpr int kBatchSizes[] = {1, 2, 4, 8, 16, 32};
static constexpr int kNumBatchSizes = 6;

struct NNEvaluator::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "jhbr2"};
  Ort::SessionOptions session_opts;

  // TensorRT mode: single session handles any batch size.
  std::unique_ptr<Ort::Session> trt_session;

  // CUDA fallback mode: one session per fixed batch size.
  std::unique_ptr<Ort::Session> sessions[kNumBatchSizes];
  bool session_valid[kNumBatchSizes] = {};

  // Input/output names (shared).
  std::vector<std::string> input_names_str;
  std::vector<std::string> output_names_str;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;

  int input_channels = 44;
  bool use_tensorrt = false;
};

NNEvaluator::NNEvaluator(const std::string& onnx_path, bool use_gpu)
    : impl_(std::make_unique<Impl>()) {

  ShogiEncoderTables::Init();

  auto& opts = impl_->session_opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // --- Try TensorRT first (best performance, dynamic batch) ---
  // Create cache directory for TRT engine files.
  std::system("mkdir -p trt_cache 2>/dev/null");
  if (use_gpu) {
    try {
      Ort::SessionOptions trt_opts;
      trt_opts.SetIntraOpNumThreads(1);
      trt_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      // TensorRT provider options via C API (lc0-style dynamic batch).
      OrtTensorRTProviderOptionsV2* trt_provider_opts = nullptr;
      Ort::GetApi().CreateTensorRTProviderOptions(&trt_provider_opts);

      // Dynamic batch profiles: min=1, opt=32, max=128.
      // ONNX Runtime builds a TensorRT engine with these optimization profiles,
      // so any batch size 1-128 runs efficiently without padding.
      std::string min_shapes = "input_planes:1x" + std::to_string(kShogiInputPlanes) + "x9x9";
      std::string opt_shapes = "input_planes:32x" + std::to_string(kShogiInputPlanes) + "x9x9";
      std::string max_shapes = "input_planes:128x" + std::to_string(kShogiInputPlanes) + "x9x9";

      std::vector<const char*> trt_keys = {
        "trt_max_workspace_size",
        "trt_fp16_enable",
        "trt_engine_cache_enable",
        "trt_engine_cache_path",
        "trt_profile_min_shapes",
        "trt_profile_opt_shapes",
        "trt_profile_max_shapes",
        "trt_builder_optimization_level",
      };
      std::vector<const char*> trt_values = {
        "2147483648",           // 2GB workspace
        "1",                    // Enable FP16
        "1",                    // Cache the TRT engine
        "./trt_cache",          // Cache directory
        min_shapes.c_str(),     // Min batch = 1
        opt_shapes.c_str(),     // Optimal batch = 32
        max_shapes.c_str(),     // Max batch = 128
        "3",                    // Builder optimization level (3 = good balance)
      };
      Ort::GetApi().UpdateTensorRTProviderOptions(
          trt_provider_opts, trt_keys.data(), trt_values.data(), trt_keys.size());
      trt_opts.AppendExecutionProvider_TensorRT_V2(*trt_provider_opts);
      Ort::GetApi().ReleaseTensorRTProviderOptions(trt_provider_opts);

      // Also add CUDA as fallback for ops TensorRT doesn't support.
      OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
      Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts);
      trt_opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
      Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);

      impl_->trt_session = std::make_unique<Ort::Session>(
          impl_->env, onnx_path.c_str(), trt_opts);

      impl_->use_tensorrt = true;
      using_gpu_ = true;
      supports_batch_ = true;  // TensorRT handles dynamic batch
      fprintf(stderr, "[NN] Using TensorRT (FP16, dynamic batch)\n");
    } catch (const std::exception& e) {
      fprintf(stderr, "[NN] TensorRT not available: %s\n", e.what());
      impl_->trt_session.reset();
      impl_->use_tensorrt = false;
    }
  }

  // --- Fall back to CUDA ---
  if (!impl_->use_tensorrt && use_gpu) {
    try {
      OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
      Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts);
      opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
      Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);
      using_gpu_ = true;
      fprintf(stderr, "[NN] Using CUDA\n");
    } catch (...) {
      using_gpu_ = false;
      fprintf(stderr, "[NN] Using CPU\n");
    }
  }

  // --- Get session for metadata (TRT or batch=1 CUDA/CPU) ---
  Ort::Session* meta_session = nullptr;

  if (impl_->use_tensorrt) {
    meta_session = impl_->trt_session.get();
  } else {
    impl_->sessions[0] = std::make_unique<Ort::Session>(
        impl_->env, onnx_path.c_str(), opts);
    impl_->session_valid[0] = true;
    meta_session = impl_->sessions[0].get();
  }

  // Get input/output names.
  Ort::AllocatorWithDefaultOptions alloc;
  size_t num_inputs = meta_session->GetInputCount();
  for (size_t i = 0; i < num_inputs; i++) {
    auto name = meta_session->GetInputNameAllocated(i, alloc);
    impl_->input_names_str.push_back(name.get());
  }
  size_t num_outputs = meta_session->GetOutputCount();
  for (size_t i = 0; i < num_outputs; i++) {
    auto name = meta_session->GetOutputNameAllocated(i, alloc);
    impl_->output_names_str.push_back(name.get());
  }
  for (auto& s : impl_->input_names_str) impl_->input_names.push_back(s.c_str());
  for (auto& s : impl_->output_names_str) impl_->output_names.push_back(s.c_str());

  // Detect input channels.
  auto input_shape = meta_session->GetInputTypeInfo(0)
      .GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() >= 2) {
    impl_->input_channels = static_cast<int>(input_shape[1]);
  }

  // --- CUDA fallback: load multi-batch ONNX files ---
  if (!impl_->use_tensorrt) {
    std::string base_path = onnx_path;
    if (base_path.size() > 5 && base_path.substr(base_path.size() - 5) == ".onnx") {
      base_path = base_path.substr(0, base_path.size() - 5);
    }
    for (int si = 1; si < kNumBatchSizes; si++) {
      int bs = kBatchSizes[si];
      std::string batch_path = base_path + "_b" + std::to_string(bs) + ".onnx";
      std::ifstream test_file(batch_path);
      if (!test_file.good()) continue;
      test_file.close();
      try {
        impl_->sessions[si] = std::make_unique<Ort::Session>(
            impl_->env, batch_path.c_str(), opts);
        impl_->session_valid[si] = true;
      } catch (...) {
        impl_->sessions[si].reset();
      }
    }
    int max_batch = 1;
    for (int si = 0; si < kNumBatchSizes; si++) {
      if (impl_->session_valid[si]) max_batch = kBatchSizes[si];
    }
    supports_batch_ = (max_batch > 1);
  }
}

NNEvaluator::~NNEvaluator() = default;

NNOutput NNEvaluator::Evaluate(const ShogiBoard& board,
                                const MoveList& legal_moves) {
  std::vector<std::pair<ShogiBoard, MoveList>> batch;
  batch.emplace_back(board, legal_moves);
  auto results = EvaluateBatch(batch);
  return std::move(results[0]);
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {

  const int batch_size = static_cast<int>(batch.size());
  const int channels = impl_->input_channels;
  constexpr int sq = 81;

  // --- Select session and determine padded batch size ---
  Ort::Session* session = nullptr;
  int padded_size = batch_size;

  if (impl_->use_tensorrt) {
    // TensorRT: single session, use actual batch size (no padding needed).
    session = impl_->trt_session.get();
    padded_size = batch_size;
  } else {
    // CUDA fallback: find best matching fixed-batch session.
    int session_idx = 0;
    for (int si = kNumBatchSizes - 1; si >= 0; si--) {
      if (impl_->session_valid[si] && kBatchSizes[si] >= batch_size) {
        session_idx = si;
      }
    }
    padded_size = kBatchSizes[session_idx];

    // If no session fits, evaluate one at a time.
    if (padded_size < batch_size) {
      std::vector<NNOutput> results;
      results.reserve(batch_size);
      for (const auto& [board, moves] : batch) {
        results.push_back(Evaluate(board, moves));
      }
      return results;
    }
    session = impl_->sessions[session_idx].get();
  }

  // --- Encode input planes ---
  std::vector<float> input_data(padded_size * channels * sq, 0.0f);
  for (int b = 0; b < batch_size; b++) {
    auto planes = EncodeShogiPosition(batch[b].first);
    float* dst = input_data.data() + b * channels * sq;
    for (int c = 0; c < channels && c < kShogiInputPlanes; c++) {
      std::copy(planes[c].data, planes[c].data + sq, dst + c * sq);
    }
  }

  // --- Run inference ---
  std::array<int64_t, 4> input_shape = {padded_size, channels, 9, 9};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  std::vector<Ort::Value> outputs;
  try {
    outputs = session->Run(
        Ort::RunOptions{nullptr},
        impl_->input_names.data(), &input_tensor, 1,
        impl_->output_names.data(), impl_->output_names.size());
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "[NN] ONNX error (batch=%d): %s\n", batch_size, e.what());
    // Fall back to one-at-a-time.
    std::vector<NNOutput> results;
    for (const auto& [board, moves] : batch) {
      results.push_back(Evaluate(board, moves));
    }
    return results;
  }

  // --- Parse outputs ---
  float* policy_data = outputs[0].GetTensorMutableData<float>();
  float* wdl_data = outputs[1].GetTensorMutableData<float>();

  auto policy_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  int policy_size = static_cast<int>(policy_shape.back());

  std::vector<NNOutput> results(batch_size);

  for (int b = 0; b < batch_size; b++) {
    auto& result = results[b];
    const auto& board = batch[b].first;
    const auto& legal_moves = batch[b].second;

    // WDL softmax
    float wdl[3];
    std::copy(wdl_data + b * 3, wdl_data + b * 3 + 3, wdl);
    Softmax(wdl, 3);

    result.wdl[0] = wdl[0];
    result.wdl[1] = wdl[1];
    result.wdl[2] = wdl[2];
    result.value = wdl[0] - wdl[2];
    result.draw = wdl[1];

    // Policy
    float* logits = policy_data + b * policy_size;
    bool is_white = (board.side_to_move() == lczero::WHITE);

    std::vector<float> legal_logits(legal_moves.size());
    float max_logit = -1e10f;

    for (size_t i = 0; i < legal_moves.size(); i++) {
      Move m = legal_moves[i];
      if (is_white) m.Flip();
      int idx = ShogiMoveToNNIndex(m);
      if (idx >= 0 && idx < policy_size) {
        legal_logits[i] = logits[idx];
      } else {
        legal_logits[i] = -1000.0f;
      }
      max_logit = std::max(max_logit, legal_logits[i]);
    }

    // Softmax over legal moves.
    result.policy.resize(legal_moves.size());
    float total = 0.0f;
    for (size_t i = 0; i < legal_moves.size(); i++) {
      result.policy[i] = std::exp(legal_logits[i] - max_logit);
      total += result.policy[i];
    }
    if (total > 0.0f) {
      for (auto& p : result.policy) p /= total;
    }
  }

  return results;
}

#else  // !HAS_ONNXRUNTIME

// Stub implementation.
struct NNEvaluator::Impl {};

NNEvaluator::NNEvaluator(const std::string&, bool)
    : impl_(std::make_unique<Impl>()) {
  ShogiEncoderTables::Init();
}

NNEvaluator::~NNEvaluator() = default;

NNOutput NNEvaluator::Evaluate(const ShogiBoard& board,
                                const MoveList& legal_moves) {
  NNOutput result;
  result.value = 0.0f; result.draw = 0.1f;
  result.wdl[0] = 0.45f; result.wdl[1] = 0.1f; result.wdl[2] = 0.45f;
  float u = legal_moves.empty() ? 0.0f : 1.0f / legal_moves.size();
  result.policy.assign(legal_moves.size(), u);
  return result;
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
  std::vector<NNOutput> results;
  for (auto& [board, moves] : batch) results.push_back(Evaluate(board, moves));
  return results;
}

#endif

}  // namespace jhbr2
