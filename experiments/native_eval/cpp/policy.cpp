#include "native_eval/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace native_eval {
namespace {

bool hidden_activation_supported(const std::string& activation) {
  return activation == "identity" || activation == "silu" ||
         activation == "relu" || activation == "tanh";
}

bool output_activation_supported(const std::string& activation) {
  return activation == "identity" || activation == "tanh";
}

float activate(float value, const std::string& activation) {
  if (activation == "silu") return value / (1.0f + std::exp(-value));
  if (activation == "relu") return std::max(0.0f, value);
  if (activation == "tanh") return std::tanh(value);
  return value;
}

void check_finite(const std::vector<float>& values, const char* name) {
  if (std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); })) {
    throw std::invalid_argument(std::string(name) + " must contain only finite float32 values");
  }
}

void sized_default(std::vector<float>& values, std::size_t size,
                   float default_value, const char* name) {
  if (values.empty()) values.assign(size, default_value);
  if (values.size() != size) {
    throw std::invalid_argument(std::string(name) + " must have action_size entries");
  }
  check_finite(values, name);
}

class MlpPolicy final : public Policy {
 public:
  explicit MlpPolicy(MlpSpec spec) : spec_(std::move(spec)) {
    if (spec_.mean.empty() || spec_.stddev.size() != spec_.mean.size()) {
      throw std::invalid_argument("policy mean/std must have equal positive dimensions");
    }
    check_finite(spec_.mean, "policy mean");
    check_finite(spec_.stddev, "policy std");
    for (const float value : spec_.stddev) {
      if (value <= 0.0f) throw std::invalid_argument("policy std entries must be positive");
    }
    if (spec_.action_size == 0 || spec_.layers.empty()) {
      throw std::invalid_argument("policy action_size and number of layers must be positive");
    }
    std::size_t previous = spec_.mean.size();
    max_width_ = previous;
    for (const auto& layer : spec_.layers) {
      if (layer.inputs == 0 || layer.outputs == 0 || layer.inputs != previous) {
        throw std::invalid_argument("dense layer input/output dimensions do not connect");
      }
      if (layer.inputs > std::numeric_limits<std::size_t>::max() / layer.outputs ||
          layer.weights.size() != layer.inputs * layer.outputs ||
          layer.bias.size() != layer.outputs) {
        throw std::invalid_argument("dense layer weights/bias have the wrong shape");
      }
      if (!hidden_activation_supported(layer.activation)) {
        throw std::invalid_argument("unsupported dense layer activation: " + layer.activation);
      }
      check_finite(layer.weights, "dense layer weights");
      check_finite(layer.bias, "dense layer bias");
      previous = layer.outputs;
      max_width_ = std::max(max_width_, layer.outputs);
    }
    if (previous != spec_.action_size) {
      throw std::invalid_argument("final dense layer does not match action_size");
    }
    if (!output_activation_supported(spec_.output_activation)) {
      throw std::invalid_argument("unsupported policy output activation: " + spec_.output_activation);
    }
    if (spec_.skip_transform != "identity" && spec_.skip_transform != "atanh") {
      throw std::invalid_argument("unsupported policy skip_transform: " + spec_.skip_transform);
    }
    if (!std::isfinite(spec_.skip_clip_epsilon) || spec_.skip_clip_epsilon <= 0.0f ||
        spec_.skip_clip_epsilon >= 1.0f ||
        1.0f - spec_.skip_clip_epsilon >= 1.0f ||
        -1.0f + spec_.skip_clip_epsilon <= -1.0f) {
      throw std::invalid_argument("skip_clip_epsilon must define a nonempty float32 interval inside (-1, 1)");
    }
    if (spec_.skip_indices.empty()) spec_.skip_indices.assign(spec_.action_size, -1);
    if (spec_.skip_indices.size() != spec_.action_size) {
      throw std::invalid_argument("skip_indices must have action_size entries");
    }
    for (const int index : spec_.skip_indices) {
      if (index < -1 || (index >= 0 && static_cast<std::size_t>(index) >= spec_.mean.size())) {
        throw std::invalid_argument("skip_indices must be -1 or valid raw observation indices");
      }
    }
    sized_default(spec_.network_scales, spec_.action_size, 1.0f, "network_scales");
    sized_default(spec_.skip_scales, spec_.action_size, 1.0f, "skip_scales");
  }

  std::size_t observation_size() const override { return spec_.mean.size(); }
  std::size_t action_size() const override { return spec_.action_size; }

  void infer(const std::vector<double>& observation, std::vector<double>& action) const override {
    if (observation.size() != observation_size()) {
      throw std::invalid_argument("policy observation has the wrong shape");
    }
    // Scratch belongs to the invocation, so independent calls and evaluators
    // share neither policy state nor mutable inference buffers.
    std::vector<float> current;
    std::vector<float> next;
    current.reserve(max_width_);
    next.reserve(max_width_);
    current.resize(observation.size());
    for (std::size_t index = 0; index < observation.size(); ++index) {
      const float raw = static_cast<float>(observation[index]);
      if (!std::isfinite(raw)) {
        throw std::invalid_argument("policy observation must contain finite float32 values");
      }
      current[index] = (raw - spec_.mean[index]) / spec_.stddev[index];
    }
    for (const auto& layer : spec_.layers) {
      next.resize(layer.outputs);
      for (std::size_t output = 0; output < layer.outputs; ++output) {
        float value = 0.0f;
        const std::size_t offset = output * layer.inputs;
        for (std::size_t input = 0; input < layer.inputs; ++input) {
          value += layer.weights[offset + input] * current[input];
        }
        next[output] = activate(value + layer.bias[output], layer.activation);
      }
      current.swap(next);
    }
    action.resize(action_size());
    for (std::size_t output = 0; output < action_size(); ++output) {
      float value = spec_.network_scales[output] * current[output];
      const int index = spec_.skip_indices[output];
      if (index >= 0) {
        float skip = static_cast<float>(observation[static_cast<std::size_t>(index)]);
        if (spec_.skip_transform == "atanh") {
          skip = std::atanh(std::clamp(skip, -1.0f + spec_.skip_clip_epsilon,
                                     1.0f - spec_.skip_clip_epsilon));
        }
        value += spec_.skip_scales[output] * skip;
      }
      value = activate(value, spec_.output_activation);
      if (!std::isfinite(value)) throw std::runtime_error("policy inference produced a non-finite action");
      action[output] = static_cast<double>(value);
    }
  }

 private:
  MlpSpec spec_;
  std::size_t max_width_ = 0;
};

}  // namespace

std::unique_ptr<Policy> make_mlp_policy(MlpSpec spec) {
  return std::make_unique<MlpPolicy>(std::move(spec));
}

}  // namespace native_eval
