/*
  JHBR2 Shogi Engine - Neural Network Evaluator (ONNX Runtime)

  Input:  (batch, C, 9, 9) float32
  Output: policy (batch, 2187), wdl (batch, 3), mlh (batch, 1)

  Execution provider priority:
    1. CUDA
    2. CPU (fallback)
*/

#include "inference/nn_eval.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#if HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include "shogi/encoder.h"

namespace jhbr2 {

using namespace lczero;

namespace {

void Softmax(float* data, int size) {
  const float max_val = *std::max_element(data, data + size);
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    data[i] = std::exp(data[i] - max_val);
    sum += data[i];
  }
  if (sum > 0.0f) {
    for (int i = 0; i < size; i++) data[i] /= sum;
  }
}

}  // namespace

#if HAS_ONNXRUNTIME

struct NNEvaluator::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "jhbr2"};
  std::unique_ptr<Ort::Session> session;

  std::vector<std::string> input_names_str;
  std::vector<std::string> output_names_str;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;

  int input_channels = 0;
};

namespace {

Ort::SessionOptions MakeSessionOptions() {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return options;
}

void AppendCudaProvider(Ort::SessionOptions& options, int device_id) {
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(
      Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));

  try {
    const std::string device_id_string = std::to_string(device_id);
    const char* keys[] = {"device_id"};
    const char* values[] = {device_id_string.c_str()};
    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
        cuda_options, keys, values, 1));
    options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  } catch (...) {
    Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
    throw;
  }

  Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
}

}  // namespace

NNEvaluator::NNEvaluator(const std::string& onnx_path, bool use_gpu,
                         int device_id, int /*num_slots*/,
                         ModelFormat model_format)
    : impl_(std::make_unique<Impl>()) {
  if (model_format == ModelFormat::kDlshogi) {
    throw std::invalid_argument(
        "ONNX Runtime supports only the single-input JHBR2 model format");
  }

  ShogiEncoderTables::Init();

  if (use_gpu) {
    try {
      auto options = MakeSessionOptions();
      AppendCudaProvider(options, device_id);
      impl_->session = std::make_unique<Ort::Session>(
          impl_->env, onnx_path.c_str(), options);
      using_gpu_ = true;
    } catch (const std::exception& e) {
      fprintf(stderr, "[ORT] CUDA unavailable on device %d: %s\n",
              device_id, e.what());
    }
  }

  if (!impl_->session) {
    auto options = MakeSessionOptions();
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env, onnx_path.c_str(), options);
    using_gpu_ = false;
  }

  const size_t num_inputs = impl_->session->GetInputCount();
  if (num_inputs != 1) {
    throw std::runtime_error(
        "ONNX Runtime requires a JHBR2 model with exactly one input");
  }
  const size_t num_outputs = impl_->session->GetOutputCount();
  if (num_outputs < 2) {
    throw std::runtime_error(
        "ONNX Runtime requires policy and WDL model outputs");
  }

  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < num_inputs; i++) {
    auto name = impl_->session->GetInputNameAllocated(i, allocator);
    impl_->input_names_str.push_back(name.get());
  }
  for (size_t i = 0; i < num_outputs; i++) {
    auto name = impl_->session->GetOutputNameAllocated(i, allocator);
    impl_->output_names_str.push_back(name.get());
  }
  for (auto& name : impl_->input_names_str) {
    impl_->input_names.push_back(name.c_str());
  }
  for (auto& name : impl_->output_names_str) {
    impl_->output_names.push_back(name.c_str());
  }

  const auto input_shape = impl_->session->GetInputTypeInfo(0)
      .GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape.size() != 4 || input_shape[1] <= 0 ||
      input_shape[2] != 9 || input_shape[3] != 9) {
    throw std::runtime_error(
        "ONNX Runtime requires input shape (batch, channels, 9, 9)");
  }
  if (input_shape[0] >= 0) {
    throw std::runtime_error(
        "ONNX Runtime requires a dynamic model batch dimension; fixed batch=" +
        std::to_string(input_shape[0]));
  }
  impl_->input_channels = static_cast<int>(input_shape[1]);

  if (using_gpu_) {
    fprintf(stderr,
            "[ORT] backend=CUDA device=%d batch=dynamic channels=%d\n",
            device_id, impl_->input_channels);
  } else {
    fprintf(stderr, "[ORT] backend=CPU batch=dynamic channels=%d\n",
            impl_->input_channels);
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
  if (batch_size == 0) return {};

  const int channels = impl_->input_channels;
  constexpr int squares = 81;

  std::vector<float> input_data(
      static_cast<size_t>(batch_size) * channels * squares, 0.0f);
  for (int b = 0; b < batch_size; b++) {
    auto planes = EncodeShogiPosition(batch[b].first);
    float* destination =
        input_data.data() + static_cast<size_t>(b) * channels * squares;
    for (int c = 0; c < channels && c < kShogiInputPlanes; c++) {
      std::copy(planes[c].data, planes[c].data + squares,
                destination + c * squares);
    }
  }

  std::array<int64_t, 4> input_shape = {
      batch_size, channels, 9, 9};
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  std::vector<Ort::Value> outputs;
  try {
    outputs = impl_->session->Run(
        Ort::RunOptions{nullptr}, impl_->input_names.data(), &input_tensor, 1,
        impl_->output_names.data(), impl_->output_names.size());
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "[ORT] Inference failed (batch=%d): %s\n",
            batch_size, e.what());
    throw;
  }

  float* policy_data = outputs[0].GetTensorMutableData<float>();
  float* wdl_data = outputs[1].GetTensorMutableData<float>();

  const auto policy_shape =
      outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const int policy_size = static_cast<int>(policy_shape.back());

  std::vector<NNOutput> results(batch_size);
  for (int b = 0; b < batch_size; b++) {
    auto& result = results[b];
    const auto& board = batch[b].first;
    const auto& legal_moves = batch[b].second;

    float wdl[3];
    std::copy(wdl_data + b * 3, wdl_data + b * 3 + 3, wdl);
    Softmax(wdl, 3);

    result.wdl[0] = wdl[0];
    result.wdl[1] = wdl[1];
    result.wdl[2] = wdl[2];
    result.value = wdl[0] - wdl[2];
    result.draw = wdl[1];

    float* logits = policy_data + b * policy_size;
    const bool is_white = (board.side_to_move() == lczero::WHITE);

    std::vector<float> legal_logits(legal_moves.size());
    float max_logit = -1e10f;
    for (size_t i = 0; i < legal_moves.size(); i++) {
      Move move = legal_moves[i];
      if (is_white) move.Flip();
      const int index = ShogiMoveToNNIndex(move);
      if (index >= 0 && index < policy_size) {
        legal_logits[i] = logits[index];
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
      for (auto& probability : result.policy) probability /= total;
    }
  }

  return results;
}

#else  // !HAS_ONNXRUNTIME

struct NNEvaluator::Impl {};

NNEvaluator::NNEvaluator(const std::string&, bool, int, int, ModelFormat)
    : impl_(std::make_unique<Impl>()) {
  throw std::runtime_error(
      "JHBR2 was built without TensorRT or ONNX Runtime; "
      "a neural-network backend is required");
}

NNEvaluator::~NNEvaluator() = default;

NNOutput NNEvaluator::Evaluate(const ShogiBoard&, const MoveList&) {
  throw std::logic_error("neural-network backend is unavailable");
}

std::vector<NNOutput> NNEvaluator::EvaluateBatch(
    const std::vector<std::pair<ShogiBoard, MoveList>>&) {
  throw std::logic_error("neural-network backend is unavailable");
}

#endif

}  // namespace jhbr2
