/*
  JHBR2 Shogi Engine — Native TensorRT Backend

  Uses TensorRT C++ API directly for NN inference.
  Supports dynamic batch sizes and FP16 inference.
*/

#ifdef USE_TENSORRT

#include "mcts/nn_tensorrt.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <vector>

#include "shogi/encoder.h"

namespace jhbr2 {

using namespace lczero;

// =====================================================================
// TensorRT logger
// =====================================================================

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      fprintf(stderr, "[TRT] %s\n", msg);
    }
  }
};

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
// CUDA helpers
// =====================================================================

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "[CUDA] Error: %s at %s:%d\n", \
            cudaGetErrorString(err), __FILE__, __LINE__); \
  } \
} while(0)

// =====================================================================
// Implementation
// =====================================================================

struct NNEvaluator::Impl {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  // Binding indices for input/output tensors.
  int input_idx = -1;
  int policy_idx = -1;
  int wdl_idx = -1;
  int mlh_idx = -1;

  // Input dimensions.
  int input_channels = 48;
  int max_batch_size = 32;

  // GPU buffers (pre-allocated for max batch size).
  void* d_input = nullptr;
  void* d_policy = nullptr;
  void* d_wdl = nullptr;
  void* d_mlh = nullptr;

  // Host buffers.
  std::vector<float> h_input;
  std::vector<float> h_policy;
  std::vector<float> h_wdl;
  std::vector<float> h_mlh;

  // Output sizes per sample.
  int policy_size = 0;
  bool dynamic_batch = false;

  cudaStream_t stream = nullptr;

  ~Impl() {
    if (d_input) cudaFree(d_input);
    if (d_policy) cudaFree(d_policy);
    if (d_wdl) cudaFree(d_wdl);
    if (d_mlh) cudaFree(d_mlh);
    if (stream) cudaStreamDestroy(stream);
  }
};

NNEvaluator::NNEvaluator(const std::string& engine_path, bool /*use_gpu*/)
    : impl_(std::make_unique<Impl>()) {

  ShogiEncoderTables::Init();

  // Load serialized engine from file.
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    fprintf(stderr, "[TRT] Cannot open engine file: %s\n", engine_path.c_str());
    fprintf(stderr, "[TRT] Convert with: trtexec --onnx=model.onnx "
            "--saveEngine=model.engine --fp16 "
            "--minShapes=input_planes:1x48x9x9 "
            "--optShapes=input_planes:16x48x9x9 "
            "--maxShapes=input_planes:32x48x9x9\n");
    return;
  }

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(file_size);
  file.read(engine_data.data(), file_size);

  // Create runtime and deserialize engine.
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(
      engine_data.data(), engine_data.size()));

  if (!impl_->engine) {
    fprintf(stderr, "[TRT] Failed to deserialize engine\n");
    return;
  }

  // Create execution context.
  impl_->context.reset(impl_->engine->createExecutionContext());

  // Find binding indices by name.
  int nb = impl_->engine->getNbIOTensors();
  for (int i = 0; i < nb; i++) {
    const char* name = impl_->engine->getIOTensorName(i);
    if (std::string(name) == "input_planes") impl_->input_idx = i;
    else if (std::string(name) == "policy") impl_->policy_idx = i;
    else if (std::string(name) == "wdl") impl_->wdl_idx = i;
    else if (std::string(name) == "mlh") impl_->mlh_idx = i;
  }

  if (impl_->input_idx < 0 || impl_->policy_idx < 0 || impl_->wdl_idx < 0) {
    fprintf(stderr, "[TRT] Missing expected tensor names (input_planes, policy, wdl)\n");
    return;
  }

  // Get input channel count from the engine.
  auto input_dims = impl_->engine->getTensorShape("input_planes");
  if (input_dims.nbDims >= 2) {
    impl_->input_channels = input_dims.d[1];
  }

  // Determine batch size from the engine's input shape.
  // For static-batch engines, this is the fixed batch size.
  // For dynamic-batch engines, use the max profile shape.
  auto engine_input_dims = impl_->engine->getTensorShape("input_planes");
  if (engine_input_dims.nbDims >= 1 && engine_input_dims.d[0] > 0) {
    // Static batch — the batch dim is fixed.
    impl_->max_batch_size = engine_input_dims.d[0];
    impl_->dynamic_batch = false;
  } else {
    // Dynamic batch — check optimization profile.
    impl_->max_batch_size = 32;  // default
    impl_->dynamic_batch = true;
    int nb_profiles = impl_->engine->getNbOptimizationProfiles();
    if (nb_profiles > 0) {
      auto max_dims = impl_->engine->getProfileShape(
          "input_planes", 0, nvinfer1::OptProfileSelector::kMAX);
      if (max_dims.nbDims >= 1) {
        impl_->max_batch_size = max_dims.d[0];
      }
    }
  }

  // Determine policy size.
  auto policy_dims = impl_->engine->getTensorShape("policy");
  impl_->policy_size = (policy_dims.nbDims >= 2) ? policy_dims.d[1] : 2187;

  fprintf(stderr, "[TRT] Engine loaded: channels=%d, max_batch=%d, policy_size=%d, dynamic=%s\n",
          impl_->input_channels, impl_->max_batch_size, impl_->policy_size,
          impl_->dynamic_batch ? "yes" : "no");

  // Create CUDA stream.
  CUDA_CHECK(cudaStreamCreate(&impl_->stream));

  // Allocate GPU buffers for max batch size.
  int B = impl_->max_batch_size;
  int C = impl_->input_channels;
  int P = impl_->policy_size;

  CUDA_CHECK(cudaMalloc(&impl_->d_input, B * C * 81 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&impl_->d_policy, B * P * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&impl_->d_wdl, B * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&impl_->d_mlh, B * 1 * sizeof(float)));

  // Allocate host buffers.
  impl_->h_input.resize(B * C * 81);
  impl_->h_policy.resize(B * P);
  impl_->h_wdl.resize(B * 3);
  impl_->h_mlh.resize(B * 1);
}

NNEvaluator::~NNEvaluator() = default;

NNOutput NNEvaluator::Evaluate(const ShogiBoard& board,
                                const MoveList& legal_moves) {
  std::vector<std::pair<ShogiBoard, MoveList>> batch;
  batch.emplace_back(board, legal_moves);
  return EvaluateBatch(batch)[0];
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {

  const int batch_size = static_cast<int>(batch.size());
  const int C = impl_->input_channels;
  const int P = impl_->policy_size;
  constexpr int sq = 81;

  if (!impl_->engine || !impl_->context) {
    // Engine not loaded — return uniform policy.
    std::vector<NNOutput> results(batch_size);
    for (int b = 0; b < batch_size; b++) {
      results[b].value = 0.0f;
      results[b].draw = 0.1f;
      results[b].wdl[0] = 0.45f;
      results[b].wdl[1] = 0.1f;
      results[b].wdl[2] = 0.45f;
      float u = batch[b].second.empty() ? 0.0f : 1.0f / batch[b].second.size();
      results[b].policy.assign(batch[b].second.size(), u);
    }
    return results;
  }

  // For dynamic-batch engines, use the actual batch size (no padding waste).
  // For static-batch engines, always use the full batch size (pad with zeros).
  int run_batch = impl_->dynamic_batch ? batch_size : impl_->max_batch_size;
  if (batch_size > run_batch) {
    // Batch too large — evaluate in chunks.
    std::vector<NNOutput> results;
    results.reserve(batch_size);
    for (int start = 0; start < batch_size; start += run_batch) {
      int end = std::min(start + run_batch, batch_size);
      std::vector<std::pair<ShogiBoard, MoveList>> chunk(
          batch.begin() + start, batch.begin() + end);
      auto chunk_results = EvaluateBatch(chunk);
      results.insert(results.end(), chunk_results.begin(), chunk_results.end());
    }
    return results;
  }

  // Encode input planes (zero-padded to run_batch).
  std::fill(impl_->h_input.begin(),
            impl_->h_input.begin() + run_batch * C * sq, 0.0f);

  for (int b = 0; b < batch_size; b++) {
    auto planes = EncodeShogiPosition(batch[b].first);
    float* dst = impl_->h_input.data() + b * C * sq;
    for (int c = 0; c < C && c < kShogiInputPlanes; c++) {
      std::copy(planes[c].data, planes[c].data + sq, dst + c * sq);
    }
  }

  // Set input shape for this batch.
  nvinfer1::Dims4 input_dims{run_batch, C, 9, 9};
  impl_->context->setInputShape("input_planes", input_dims);

  // Set tensor addresses.
  impl_->context->setTensorAddress("input_planes", impl_->d_input);
  impl_->context->setTensorAddress("policy", impl_->d_policy);
  impl_->context->setTensorAddress("wdl", impl_->d_wdl);
  if (impl_->mlh_idx >= 0) {
    impl_->context->setTensorAddress("mlh", impl_->d_mlh);
  }

  // Copy input to GPU (full run_batch, including padding).
  CUDA_CHECK(cudaMemcpyAsync(impl_->d_input, impl_->h_input.data(),
      run_batch * C * sq * sizeof(float),
      cudaMemcpyHostToDevice, impl_->stream));

  // Run inference.
  bool ok = impl_->context->enqueueV3(impl_->stream);
  if (!ok) {
    fprintf(stderr, "[TRT] Inference failed\n");
  }

  // Copy outputs to host (only need batch_size results, but copy run_batch for simplicity).
  CUDA_CHECK(cudaMemcpyAsync(impl_->h_policy.data(), impl_->d_policy,
      run_batch * P * sizeof(float),
      cudaMemcpyDeviceToHost, impl_->stream));
  CUDA_CHECK(cudaMemcpyAsync(impl_->h_wdl.data(), impl_->d_wdl,
      run_batch * 3 * sizeof(float),
      cudaMemcpyDeviceToHost, impl_->stream));

  // Wait for completion.
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

  // Parse outputs.
  std::vector<NNOutput> results(batch_size);

  for (int b = 0; b < batch_size; b++) {
    auto& result = results[b];
    const auto& board = batch[b].first;
    const auto& legal_moves = batch[b].second;

    // WDL softmax.
    float wdl[3];
    std::copy(impl_->h_wdl.data() + b * 3,
              impl_->h_wdl.data() + b * 3 + 3, wdl);
    Softmax(wdl, 3);
    result.wdl[0] = wdl[0];
    result.wdl[1] = wdl[1];
    result.wdl[2] = wdl[2];
    result.value = wdl[0] - wdl[2];
    result.draw = wdl[1];

    // Policy.
    float* logits = impl_->h_policy.data() + b * P;
    bool is_white = (board.side_to_move() == lczero::WHITE);

    std::vector<float> legal_logits(legal_moves.size());
    float max_logit = -1e10f;

    for (size_t i = 0; i < legal_moves.size(); i++) {
      Move m = legal_moves[i];
      if (is_white) m.Flip();
      int idx = ShogiMoveToNNIndex(m);
      if (idx >= 0 && idx < P) {
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

}  // namespace jhbr2

#endif  // USE_TENSORRT
