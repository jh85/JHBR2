/*
  JHBR2 Shogi Engine — Neural Network Evaluator (ONNX Runtime)

  Input:  (batch, 44, 9, 9) float32  [note: model uses 44 of 48 input planes]
  Output: policy (batch, 2187), wdl (batch, 3), mlh (batch, 1)

  Batching strategy: create separate ONNX sessions for fixed batch sizes
  (1, 2, 4, 8, 16). At runtime, pick the smallest session that fits the
  actual batch, pad with zeros. This avoids dynamic shape issues with CUDA.
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

// Fixed batch sizes we create sessions for.
static constexpr int kBatchSizes[] = {1, 2, 4, 8, 16};
static constexpr int kNumBatchSizes = 5;

struct NNEvaluator::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "jhbr2"};
  Ort::SessionOptions session_opts;

  // One session per fixed batch size. session[0]=batch1, session[1]=batch2, etc.
  std::unique_ptr<Ort::Session> sessions[kNumBatchSizes];
  bool session_valid[kNumBatchSizes] = {};  // true if session was created successfully

  // Input/output names (shared across all sessions — same model).
  std::vector<std::string> input_names_str;
  std::vector<std::string> output_names_str;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;

  // Input shape: (batch, channels, 9, 9)
  int input_channels = 44;
};

NNEvaluator::NNEvaluator(const std::string& onnx_path, bool use_gpu)
    : impl_(std::make_unique<Impl>()) {

  // Initialize encoder tables.
  ShogiEncoderTables::Init();

  auto& opts = impl_->session_opts;
  opts.SetIntraOpNumThreads(1);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Try GPU first if requested.
  if (use_gpu) {
    try {
      OrtCUDAProviderOptionsV2* cuda_opts = nullptr;
      Ort::GetApi().CreateCUDAProviderOptions(&cuda_opts);
      opts.AppendExecutionProvider_CUDA_V2(*cuda_opts);
      Ort::GetApi().ReleaseCUDAProviderOptions(cuda_opts);
      using_gpu_ = true;
    } catch (...) {
      using_gpu_ = false;
    }
  }

  // Create the batch=1 session first (always needed).
  impl_->sessions[0] = std::make_unique<Ort::Session>(
      impl_->env, onnx_path.c_str(), opts);
  impl_->session_valid[0] = true;

  // Get input/output names from the first session.
  Ort::AllocatorWithDefaultOptions alloc;
  auto& session = *impl_->sessions[0];

  size_t num_inputs = session.GetInputCount();
  for (size_t i = 0; i < num_inputs; i++) {
    auto name = session.GetInputNameAllocated(i, alloc);
    impl_->input_names_str.push_back(name.get());
  }

  size_t num_outputs = session.GetOutputCount();
  for (size_t i = 0; i < num_outputs; i++) {
    auto name = session.GetOutputNameAllocated(i, alloc);
    impl_->output_names_str.push_back(name.get());
  }

  for (auto& s : impl_->input_names_str) impl_->input_names.push_back(s.c_str());
  for (auto& s : impl_->output_names_str) impl_->output_names.push_back(s.c_str());

  // Detect input channels from model shape.
  auto input_shape = session.GetInputTypeInfo(0)
      .GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() >= 2) {
    impl_->input_channels = static_cast<int>(input_shape[1]);
  }

  // Try loading separate ONNX files for larger batch sizes.
  // Convention: model.onnx → model_b2.onnx, model_b4.onnx, etc.
  // Strip ".onnx" from the path and look for batch variants.
  std::string base_path = onnx_path;
  if (base_path.size() > 5 && base_path.substr(base_path.size() - 5) == ".onnx") {
    base_path = base_path.substr(0, base_path.size() - 5);
  }

  for (int si = 1; si < kNumBatchSizes; si++) {
    int bs = kBatchSizes[si];
    std::string batch_path = base_path + "_b" + std::to_string(bs) + ".onnx";

    // Check if the file exists.
    std::ifstream test_file(batch_path);
    if (!test_file.good()) continue;
    test_file.close();

    try {
      impl_->sessions[si] = std::make_unique<Ort::Session>(
          impl_->env, batch_path.c_str(), opts);
      impl_->session_valid[si] = true;
    } catch (...) {
      impl_->sessions[si].reset();
      impl_->session_valid[si] = false;
    }
  }

  // Report batch support.
  int max_batch = 1;
  for (int si = 0; si < kNumBatchSizes; si++) {
    if (impl_->session_valid[si]) max_batch = kBatchSizes[si];
  }
  supports_batch_ = (max_batch > 1);
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

  // Find the best session for this batch size.
  int session_idx = 0;  // default: batch=1
  for (int si = kNumBatchSizes - 1; si >= 0; si--) {
    if (impl_->session_valid[si] && kBatchSizes[si] >= batch_size) {
      session_idx = si;
    }
  }

  int padded_size = kBatchSizes[session_idx];

  // If no session can fit this batch, evaluate one at a time.
  if (padded_size < batch_size) {
    std::vector<NNOutput> results;
    results.reserve(batch_size);
    for (const auto& [board, moves] : batch) {
      results.push_back(Evaluate(board, moves));
    }
    return results;
  }

  // Encode input planes (zero-padded for unused slots).
  std::vector<float> input_data(padded_size * channels * sq, 0.0f);

  for (int b = 0; b < batch_size; b++) {
    auto planes = EncodeShogiPosition(batch[b].first);
    float* dst = input_data.data() + b * channels * sq;
    for (int c = 0; c < channels && c < kShogiInputPlanes; c++) {
      std::copy(planes[c].data, planes[c].data + sq, dst + c * sq);
    }
  }

  // Create input tensor with padded batch size.
  std::array<int64_t, 4> input_shape = {padded_size, channels, 9, 9};
  auto memory_info = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  // Run inference on the selected session.
  auto& session = *impl_->sessions[session_idx];
  std::vector<Ort::Value> outputs;
  try {
    outputs = session.Run(
        Ort::RunOptions{nullptr},
        impl_->input_names.data(), &input_tensor, 1,
        impl_->output_names.data(), impl_->output_names.size());
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "[NN] ONNX error (batch=%d, padded=%d): %s\n",
            batch_size, padded_size, e.what());
    // Fall back to one-at-a-time with batch=1 session.
    std::vector<NNOutput> results;
    for (const auto& [board, moves] : batch) {
      results.push_back(Evaluate(board, moves));
    }
    return results;
  }

  // Parse outputs.
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

    // Policy: extract logits for legal moves, then softmax.
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

// Stub implementation when ONNX Runtime is not available.

struct NNEvaluator::Impl {};

NNEvaluator::NNEvaluator(const std::string& /*onnx_path*/, bool /*use_gpu*/)
    : impl_(std::make_unique<Impl>()) {
  ShogiEncoderTables::Init();
}

NNEvaluator::~NNEvaluator() = default;

NNOutput NNEvaluator::Evaluate(const ShogiBoard& board,
                                const MoveList& legal_moves) {
  NNOutput result;
  result.value = 0.0f;
  result.draw = 0.1f;
  result.wdl[0] = 0.45f;
  result.wdl[1] = 0.1f;
  result.wdl[2] = 0.45f;
  float uniform = legal_moves.empty() ? 0.0f : 1.0f / legal_moves.size();
  result.policy.assign(legal_moves.size(), uniform);
  return result;
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
  std::vector<NNOutput> results;
  results.reserve(batch.size());
  for (auto& [board, moves] : batch) {
    results.push_back(Evaluate(board, moves));
  }
  return results;
}

#endif  // HAS_ONNXRUNTIME

}  // namespace jhbr2
