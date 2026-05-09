/*
  JHBR2 Shogi Engine — Native TensorRT Backend

  Uses TensorRT C++ API directly. Supports multiple execution slots
  per GPU: each slot owns its own IExecutionContext + CUDA stream +
  pinned host buffers + device buffers, allowing N workers to keep
  the GPU busy concurrently (dlshogi-style design).
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
#include <memory>
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
// Per-slot resources: one IExecutionContext + stream + buffers.
// =====================================================================

struct Slot {
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream = nullptr;
  void* d_input = nullptr;
  void* d_policy = nullptr;
  void* d_wdl = nullptr;
  void* d_mlh = nullptr;
  float* h_input = nullptr;
  float* h_policy = nullptr;
  float* h_wdl = nullptr;
  float* h_mlh = nullptr;

  ~Slot() {
    if (d_input)  cudaFree(d_input);
    if (d_policy) cudaFree(d_policy);
    if (d_wdl)    cudaFree(d_wdl);
    if (d_mlh)    cudaFree(d_mlh);
    if (h_input)  cudaFreeHost(h_input);
    if (h_policy) cudaFreeHost(h_policy);
    if (h_wdl)    cudaFreeHost(h_wdl);
    if (h_mlh)    cudaFreeHost(h_mlh);
    if (stream)   cudaStreamDestroy(stream);
  }
};

// =====================================================================
// Implementation
// =====================================================================

struct NNEvaluator::Impl {
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::vector<std::unique_ptr<Slot>> slots;

  int input_idx = -1;
  int policy_idx = -1;
  int wdl_idx = -1;
  int mlh_idx = -1;

  int input_channels = 48;
  int max_batch_size = 32;
  int policy_size = 0;
  bool dynamic_batch = false;

  int device_id = 0;
};


NNEvaluator::NNEvaluator(const std::string& engine_path, bool /*use_gpu*/,
                         int device_id, int /*max_batch_size*/,
                         int num_slots)
    : impl_(std::make_unique<Impl>()) {

  CUDA_CHECK(cudaSetDevice(device_id));
  impl_->device_id = device_id;
  ShogiEncoderTables::Init();

  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    fprintf(stderr, "[TRT] Cannot open engine file: %s\n", engine_path.c_str());
    return;
  }

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(file_size);
  file.read(engine_data.data(), file_size);

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(
      engine_data.data(), engine_data.size()));

  if (!impl_->engine) {
    fprintf(stderr, "[TRT] Failed to deserialize engine\n");
    return;
  }

  // Find tensor binding indices.
  int nb = impl_->engine->getNbIOTensors();
  for (int i = 0; i < nb; i++) {
    const char* name = impl_->engine->getIOTensorName(i);
    if (std::string(name) == "input_planes") impl_->input_idx = i;
    else if (std::string(name) == "policy") impl_->policy_idx = i;
    else if (std::string(name) == "wdl") impl_->wdl_idx = i;
    else if (std::string(name) == "mlh") impl_->mlh_idx = i;
  }

  if (impl_->input_idx < 0 || impl_->policy_idx < 0 || impl_->wdl_idx < 0) {
    fprintf(stderr, "[TRT] Missing expected tensor names\n");
    return;
  }

  auto input_dims = impl_->engine->getTensorShape("input_planes");
  if (input_dims.nbDims >= 2) {
    impl_->input_channels = input_dims.d[1];
  }

  auto engine_input_dims = impl_->engine->getTensorShape("input_planes");
  if (engine_input_dims.nbDims >= 1 && engine_input_dims.d[0] > 0) {
    impl_->max_batch_size = engine_input_dims.d[0];
    impl_->dynamic_batch = false;
  } else {
    impl_->max_batch_size = 32;
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

  auto policy_dims = impl_->engine->getTensorShape("policy");
  impl_->policy_size = (policy_dims.nbDims >= 2) ? policy_dims.d[1] : 2187;

  if (num_slots < 1) num_slots = 1;
  fprintf(stderr,
          "[TRT] Engine loaded gpu=%d: channels=%d max_batch=%d "
          "policy=%d dynamic=%s slots=%d\n",
          device_id, impl_->input_channels, impl_->max_batch_size,
          impl_->policy_size, impl_->dynamic_batch ? "yes" : "no", num_slots);

  // Allocate per-slot resources: one execution context + stream + buffers.
  impl_->slots.reserve(num_slots);
  const int B = impl_->max_batch_size;
  const int C = impl_->input_channels;
  const int P = impl_->policy_size;
  for (int s = 0; s < num_slots; s++) {
    auto slot = std::make_unique<Slot>();
    slot->context.reset(impl_->engine->createExecutionContext());
    if (!slot->context) {
      fprintf(stderr, "[TRT] Failed to create execution context (slot=%d)\n", s);
      continue;
    }
    CUDA_CHECK(cudaStreamCreate(&slot->stream));
    CUDA_CHECK(cudaMalloc(&slot->d_input,  static_cast<size_t>(B) * C * 81 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&slot->d_policy, static_cast<size_t>(B) * P * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&slot->d_wdl,    static_cast<size_t>(B) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&slot->d_mlh,    static_cast<size_t>(B) * 1 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&slot->h_input),
                              static_cast<size_t>(B) * C * 81 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&slot->h_policy),
                              static_cast<size_t>(B) * P * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&slot->h_wdl),
                              static_cast<size_t>(B) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&slot->h_mlh),
                              static_cast<size_t>(B) * 1 * sizeof(float)));
    impl_->slots.push_back(std::move(slot));
  }
}

NNEvaluator::~NNEvaluator() {
  if (impl_) cudaSetDevice(impl_->device_id);
}

int NNEvaluator::num_slots() const {
  return impl_ ? static_cast<int>(impl_->slots.size()) : 0;
}

NNOutput NNEvaluator::Evaluate(const ShogiBoard& board,
                                const MoveList& legal_moves) {
  std::vector<std::pair<ShogiBoard, MoveList>> batch;
  batch.emplace_back(board, legal_moves);
  return EvaluateBatchSlot(0, batch)[0];
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {
  return EvaluateBatchSlot(0, batch);
}

std::vector<NNOutput> NNEvaluator::EvaluateBatchSlot(
    int slot_id,
    const std::vector<std::pair<ShogiBoard, MoveList>>& batch) {

  const int batch_size = static_cast<int>(batch.size());
  const int C = impl_->input_channels;
  const int P = impl_->policy_size;
  constexpr int sq = 81;

  if (!impl_->engine || impl_->slots.empty()) {
    // Engine not loaded — return uniform policy (matches old fallback).
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

  if (slot_id < 0 || slot_id >= static_cast<int>(impl_->slots.size())) {
    slot_id = 0;
  }
  Slot& slot = *impl_->slots[slot_id];

  // Multi-GPU contexts: ensure CUDA calls in this thread go to our device.
  cudaSetDevice(impl_->device_id);

  // For dynamic engines: use actual batch size. For static: pad to max.
  int run_batch = impl_->dynamic_batch ? batch_size : impl_->max_batch_size;
  if (batch_size > run_batch) {
    // Batch too large — chunk and recurse on the same slot.
    std::vector<NNOutput> results;
    results.reserve(batch_size);
    for (int start = 0; start < batch_size; start += run_batch) {
      int end = std::min(start + run_batch, batch_size);
      std::vector<std::pair<ShogiBoard, MoveList>> chunk(
          batch.begin() + start, batch.begin() + end);
      auto chunk_results = EvaluateBatchSlot(slot_id, chunk);
      results.insert(results.end(), chunk_results.begin(), chunk_results.end());
    }
    return results;
  }

  std::fill(slot.h_input,
            slot.h_input + static_cast<size_t>(run_batch) * C * sq, 0.0f);

  for (int b = 0; b < batch_size; b++) {
    auto planes = EncodeShogiPosition(batch[b].first);
    float* dst = slot.h_input + b * C * sq;
    for (int c = 0; c < C && c < kShogiInputPlanes; c++) {
      std::copy(planes[c].data, planes[c].data + sq, dst + c * sq);
    }
  }

  nvinfer1::Dims4 input_dims{run_batch, C, 9, 9};
  slot.context->setInputShape("input_planes", input_dims);

  slot.context->setTensorAddress("input_planes", slot.d_input);
  slot.context->setTensorAddress("policy", slot.d_policy);
  slot.context->setTensorAddress("wdl", slot.d_wdl);
  if (impl_->mlh_idx >= 0) {
    slot.context->setTensorAddress("mlh", slot.d_mlh);
  }

  CUDA_CHECK(cudaMemcpyAsync(slot.d_input, slot.h_input,
      run_batch * C * sq * sizeof(float),
      cudaMemcpyHostToDevice, slot.stream));

  bool ok = slot.context->enqueueV3(slot.stream);
  if (!ok) {
    fprintf(stderr, "[TRT] Inference failed (slot=%d)\n", slot_id);
  }

  CUDA_CHECK(cudaMemcpyAsync(slot.h_policy, slot.d_policy,
      run_batch * P * sizeof(float),
      cudaMemcpyDeviceToHost, slot.stream));
  CUDA_CHECK(cudaMemcpyAsync(slot.h_wdl, slot.d_wdl,
      run_batch * 3 * sizeof(float),
      cudaMemcpyDeviceToHost, slot.stream));

  CUDA_CHECK(cudaStreamSynchronize(slot.stream));

  std::vector<NNOutput> results(batch_size);
  for (int b = 0; b < batch_size; b++) {
    auto& result = results[b];
    const auto& board = batch[b].first;
    const auto& legal_moves = batch[b].second;

    float wdl[3];
    std::copy(slot.h_wdl + b * 3, slot.h_wdl + b * 3 + 3, wdl);
    Softmax(wdl, 3);
    result.wdl[0] = wdl[0];
    result.wdl[1] = wdl[1];
    result.wdl[2] = wdl[2];
    result.value = wdl[0] - wdl[2];
    result.draw = wdl[1];

    float* logits = slot.h_policy + b * P;
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
