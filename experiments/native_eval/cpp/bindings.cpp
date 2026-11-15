#include "native_eval/runtime.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace native_eval {
namespace {

void check_keys(const py::dict& mapping, std::initializer_list<const char*> allowed,
                const char* label) {
  for (const auto& item : mapping) {
    if (!py::isinstance<py::str>(item.first)) {
      throw std::invalid_argument(std::string(label) + " keys must be strings");
    }
    const auto key = py::cast<std::string>(item.first);
    if (std::none_of(allowed.begin(), allowed.end(),
                     [&](const char* value) { return key == value; })) {
      throw std::invalid_argument(std::string(label) + " has unknown field: " + key);
    }
  }
}

py::handle required(const py::dict& mapping, const char* key, const char* label) {
  if (!mapping.contains(key)) {
    throw std::invalid_argument(std::string(label) + " is missing field: " + key);
  }
  return mapping[key];
}

py::dict as_dict(py::handle value, const char* label) {
  if (!py::isinstance<py::dict>(value)) {
    throw std::invalid_argument(std::string(label) + " must be a dictionary");
  }
  return py::reinterpret_borrow<py::dict>(value);
}

py::sequence as_sequence(py::handle value, const char* label) {
  if (py::isinstance<py::str>(value) || py::isinstance<py::bytes>(value) ||
      !py::isinstance<py::sequence>(value)) {
    throw std::invalid_argument(std::string(label) + " must be a sequence");
  }
  return py::reinterpret_borrow<py::sequence>(value);
}

std::string as_string(py::handle value, const char* label) {
  if (!py::isinstance<py::str>(value)) {
    throw std::invalid_argument(std::string(label) + " must be a string");
  }
  return py::cast<std::string>(value);
}

double as_number(py::handle value, const char* label) {
  if (PyBool_Check(value.ptr()) || !PyNumber_Check(value.ptr()) || PyComplex_Check(value.ptr())) {
    throw std::invalid_argument(std::string(label) + " must be a finite number");
  }
  const double converted = PyFloat_AsDouble(value.ptr());
  if (PyErr_Occurred()) throw py::error_already_set();
  if (!std::isfinite(converted)) {
    throw std::invalid_argument(std::string(label) + " must be a finite number");
  }
  return converted;
}

float as_float32(py::handle value, const char* label) {
  const float converted = static_cast<float>(as_number(value, label));
  if (!std::isfinite(converted)) {
    throw std::invalid_argument(std::string(label) + " must be representable as finite float32");
  }
  return converted;
}

std::uint64_t as_uint64(py::handle value, const char* label) {
  if (PyBool_Check(value.ptr()) || !PyIndex_Check(value.ptr())) {
    throw std::invalid_argument(std::string(label) + " must be a nonnegative integer");
  }
  py::object index = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
  if (!index) throw py::error_already_set();
  const auto converted = PyLong_AsUnsignedLongLong(index.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw std::invalid_argument(std::string(label) + " must be an unsigned 64-bit integer");
  }
  return static_cast<std::uint64_t>(converted);
}

std::size_t as_size(py::handle value, const char* label) {
  const auto converted = as_uint64(value, label);
  if (converted > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(label) + " is too large");
  }
  return static_cast<std::size_t>(converted);
}

int as_index(py::handle value, const char* label) {
  if (PyBool_Check(value.ptr()) || !PyIndex_Check(value.ptr())) {
    throw std::invalid_argument(std::string(label) + " must be an integer");
  }
  py::object index = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
  if (!index) throw py::error_already_set();
  const long long converted = PyLong_AsLongLong(index.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    throw std::invalid_argument(std::string(label) + " is outside the supported index range");
  }
  if (converted < std::numeric_limits<int>::min() || converted > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(label) + " is outside the supported index range");
  }
  return static_cast<int>(converted);
}

using NumericArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

NumericArray numeric_array(py::handle value, int dimensions, const char* label) {
  NumericArray array = NumericArray::ensure(value);
  if (!array) throw std::invalid_argument(std::string(label) + " must be a numeric array");
  if (array.ndim() != dimensions) {
    throw std::invalid_argument(std::string(label) + " has the wrong number of dimensions");
  }
  const double* pointer = array.data();
  for (py::ssize_t index = 0; index < array.size(); ++index) {
    if (!std::isfinite(pointer[index])) {
      throw std::invalid_argument(std::string(label) + " must contain only finite values");
    }
  }
  return array;
}

std::vector<double> numeric_vector(py::handle value, const char* label) {
  auto array = numeric_array(value, 1, label);
  if (array.size() == 0) return {};
  return {array.data(), array.data() + array.size()};
}

std::vector<float> float_vector(py::handle value, const char* label) {
  auto array = numeric_array(value, 1, label);
  std::vector<float> result(static_cast<std::size_t>(array.size()));
  for (py::ssize_t index = 0; index < array.size(); ++index) {
    result[static_cast<std::size_t>(index)] = static_cast<float>(array.data()[index]);
    if (!std::isfinite(result[static_cast<std::size_t>(index)])) {
      throw std::invalid_argument(std::string(label) + " must contain finite float32 values");
    }
  }
  return result;
}

TaskParameters parse_task_parameters(const py::dict& mapping) {
  TaskParameters result;
  for (const auto& item : mapping) {
    const auto key = as_string(item.first, "task config key");
    const auto value = item.second;
    if (py::isinstance<py::bool_>(value)) {
      result.flags.emplace(key, py::cast<bool>(value));
    } else if (py::isinstance<py::str>(value)) {
      result.strings.emplace(key, py::cast<std::string>(value));
    } else if (py::isinstance<py::sequence>(value) &&
               !py::isinstance<py::bytes>(value)) {
      auto sequence = as_sequence(value, "task config vector");
      bool all_strings = py::len(sequence) > 0;
      for (py::handle entry : sequence) all_strings = all_strings && py::isinstance<py::str>(entry);
      if (all_strings) {
        std::vector<std::string> names;
        names.reserve(static_cast<std::size_t>(py::len(sequence)));
        for (py::handle entry : sequence) names.push_back(py::cast<std::string>(entry));
        result.names.emplace(key, std::move(names));
      } else {
        result.vectors.emplace(key, numeric_vector(value, "task config vector"));
      }
    } else {
      result.numbers.emplace(key, as_number(value, "task config value"));
    }
  }
  return result;
}

struct TaskDefinition {
  std::string type;
  TaskParameters parameters;
};

TaskDefinition parse_task(const py::dict& mapping) {
  check_keys(mapping, {"type", "config"}, "task");
  TaskDefinition result;
  result.type = as_string(required(mapping, "type", "task"), "task type");
  if (result.type != "legwheel" && result.type != "pendulum") {
    throw std::invalid_argument("unsupported native task type: " + result.type);
  }
  result.parameters = parse_task_parameters(as_dict(required(mapping, "config", "task"), "task config"));
  return result;
}

MlpSpec parse_policy(const py::dict& mapping) {
  check_keys(mapping, {"type", "mean", "std", "layers", "action_size", "output_activation",
                       "skip_indices", "network_scales", "skip_scales", "skip_transform",
                       "skip_clip_epsilon"}, "policy");
  const auto type = as_string(required(mapping, "type", "policy"), "policy type");
  if (type != "mlp") throw std::invalid_argument("unsupported policy type: " + type);
  MlpSpec result;
  result.mean = float_vector(required(mapping, "mean", "policy"), "policy mean");
  result.stddev = float_vector(required(mapping, "std", "policy"), "policy std");
  result.action_size = as_size(required(mapping, "action_size", "policy"), "policy action_size");
  auto layers = as_sequence(required(mapping, "layers", "policy"), "policy layers");
  result.layers.reserve(static_cast<std::size_t>(py::len(layers)));
  for (py::handle value : layers) {
    auto mapping_layer = as_dict(value, "dense layer");
    check_keys(mapping_layer, {"weights", "bias", "activation"}, "dense layer");
    auto weights = numeric_array(required(mapping_layer, "weights", "dense layer"), 2, "dense layer weights");
    DenseLayer layer;
    layer.outputs = static_cast<std::size_t>(weights.shape(0));
    layer.inputs = static_cast<std::size_t>(weights.shape(1));
    layer.weights.resize(static_cast<std::size_t>(weights.size()));
    for (py::ssize_t index = 0; index < weights.size(); ++index) {
      const float converted = static_cast<float>(weights.data()[index]);
      if (!std::isfinite(converted)) throw std::invalid_argument("dense layer weights exceed finite float32");
      layer.weights[static_cast<std::size_t>(index)] = converted;
    }
    layer.bias = float_vector(required(mapping_layer, "bias", "dense layer"), "dense layer bias");
    if (mapping_layer.contains("activation")) {
      layer.activation = as_string(mapping_layer["activation"], "dense layer activation");
    }
    result.layers.push_back(std::move(layer));
  }
  if (mapping.contains("output_activation")) {
    result.output_activation = as_string(mapping["output_activation"], "policy output_activation");
  }
  if (mapping.contains("skip_indices")) {
    for (py::handle value : as_sequence(mapping["skip_indices"], "policy skip_indices")) {
      result.skip_indices.push_back(as_index(value, "policy skip_indices entry"));
    }
    if (result.skip_indices.size() != result.action_size) {
      throw std::invalid_argument("skip_indices must have action_size entries");
    }
  }
  if (mapping.contains("network_scales")) {
    result.network_scales = float_vector(mapping["network_scales"], "network_scales");
    if (result.network_scales.size() != result.action_size) {
      throw std::invalid_argument("network_scales must have action_size entries");
    }
  }
  if (mapping.contains("skip_scales")) {
    result.skip_scales = float_vector(mapping["skip_scales"], "skip_scales");
    if (result.skip_scales.size() != result.action_size) {
      throw std::invalid_argument("skip_scales must have action_size entries");
    }
  }
  if (mapping.contains("skip_transform")) result.skip_transform = as_string(mapping["skip_transform"], "skip_transform");
  if (mapping.contains("skip_clip_epsilon")) result.skip_clip_epsilon = as_float32(mapping["skip_clip_epsilon"], "skip_clip_epsilon");
  return result;
}

EvalOptions parse_options(const py::dict& mapping) {
  check_keys(mapping, {"num_envs", "rollout_steps", "episode_steps", "control_dt", "seed", "trace_worlds"}, "options");
  EvalOptions result;
  if (mapping.contains("num_envs")) result.num_envs = as_size(mapping["num_envs"], "num_envs");
  if (mapping.contains("rollout_steps")) result.rollout_steps = as_size(mapping["rollout_steps"], "rollout_steps");
  if (mapping.contains("episode_steps")) result.episode_steps = as_size(mapping["episode_steps"], "episode_steps");
  if (mapping.contains("control_dt")) result.control_dt = as_number(mapping["control_dt"], "control_dt");
  if (mapping.contains("seed")) result.seed = as_uint64(mapping["seed"], "seed");
  if (mapping.contains("trace_worlds")) {
    for (py::handle value : as_sequence(mapping["trace_worlds"], "trace_worlds")) {
      result.trace_worlds.push_back(as_size(value, "trace_world index"));
    }
  }
  return result;
}

std::vector<InitialState> parse_initial_states(py::handle values, int nq, int nv) {
  std::vector<InitialState> result;
  const auto sequence = as_sequence(values, "initial_states");
  result.reserve(static_cast<std::size_t>(py::len(sequence)));
  for (py::handle value : sequence) {
    auto mapping = as_dict(value, "initial state");
    check_keys(mapping, {"qpos", "qvel", "command", "extra"}, "initial state");
    InitialState state;
    if (mapping.contains("qpos")) {
      state.qpos = numeric_vector(mapping["qpos"], "initial qpos");
      if (state.qpos.size() != static_cast<std::size_t>(nq)) throw std::invalid_argument("initial qpos has the wrong shape");
    }
    if (mapping.contains("qvel")) {
      state.qvel = numeric_vector(mapping["qvel"], "initial qvel");
      if (state.qvel.size() != static_cast<std::size_t>(nv)) throw std::invalid_argument("initial qvel has the wrong shape");
    }
    if (mapping.contains("command")) state.command = numeric_vector(mapping["command"], "initial command");
    if (mapping.contains("extra")) {
      for (const auto& item : as_dict(mapping["extra"], "initial extra")) {
        state.extra.emplace(as_string(item.first, "initial extra key"),
                            numeric_vector(item.second, "initial extra vector"));
      }
    }
    result.push_back(std::move(state));
  }
  return result;
}

std::vector<std::vector<double>> parse_observations(py::handle value, std::size_t columns) {
  const auto array = numeric_array(value, 2, "observations");
  if (static_cast<std::size_t>(array.shape(1)) != columns) {
    throw std::invalid_argument("observations have the wrong shape");
  }
  std::vector<std::vector<double>> result(static_cast<std::size_t>(array.shape(0)));
  for (std::size_t row = 0; row < result.size(); ++row) {
    const double* start = array.data() + row * columns;
    result[row].assign(start, start + columns);
  }
  return result;
}

py::ssize_t python_size(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())) {
    throw std::overflow_error("result array is too large for Python");
  }
  return static_cast<py::ssize_t>(value);
}

template <typename T>
py::array_t<T> vector_array(const std::vector<T>& values) {
  py::array_t<T> result(python_size(values.size()));
  if (!values.empty()) std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(T));
  return result;
}

py::array_t<bool> boolean_array(const std::vector<unsigned char>& values) {
  py::array_t<bool> result(python_size(values.size()));
  for (std::size_t index = 0; index < values.size(); ++index) result.mutable_data()[index] = values[index] != 0;
  return result;
}

py::array_t<double> matrix_array(const std::vector<double>& values, std::size_t rows, std::size_t columns) {
  py::array_t<double> result({python_size(rows), python_size(columns)});
  if (!values.empty()) std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(double));
  return result;
}

py::array_t<double> action_array(const std::vector<std::vector<double>>& values, std::size_t columns) {
  py::array_t<double> result({python_size(values.size()), python_size(columns)});
  for (std::size_t row = 0; row < values.size(); ++row) {
    std::copy(values[row].begin(), values[row].end(), result.mutable_data() + row * columns);
  }
  return result;
}

py::dict result_dict(const EvalResult& result, const Evaluator& evaluator) {
  py::dict output;
  output["returns"] = vector_array(result.returns);
  output["lengths"] = vector_array(result.lengths);
  output["terminated"] = boolean_array(result.terminated);
  output["truncated"] = boolean_array(result.truncated);
  output["reasons"] = py::cast(result.reasons);
  output["resets"] = vector_array(result.resets);
  py::dict episode_metrics;
  episode_metrics["reward"] = vector_array(result.returns);
  for (const auto& metric : result.episode_metrics) {
    episode_metrics[py::str(metric.first)] = vector_array(metric.second);
  }
  output["episode_metrics"] = std::move(episode_metrics);
  py::list traces;
  for (const auto& trace : result.traces) {
    py::dict entry;
    entry["world"] = trace.world;
    entry["steps"] = trace.steps;
    entry["observation"] = matrix_array(trace.observation, trace.steps, evaluator.observation_size());
    entry["action"] = matrix_array(trace.action, trace.steps, evaluator.action_size());
    entry["qpos"] = matrix_array(trace.qpos, trace.steps, static_cast<std::size_t>(evaluator.nq()));
    entry["qvel"] = matrix_array(trace.qvel, trace.steps, static_cast<std::size_t>(evaluator.nv()));
    entry["ctrl"] = matrix_array(trace.ctrl, trace.steps, static_cast<std::size_t>(evaluator.nu()));
    py::array_t<double> xmat({python_size(trace.steps),
                              python_size(static_cast<std::size_t>(evaluator.nbody())),
                              py::ssize_t(3), py::ssize_t(3)});
    if (!trace.xmat.empty()) {
      std::memcpy(xmat.mutable_data(), trace.xmat.data(), trace.xmat.size() * sizeof(double));
    }
    entry["xmat"] = std::move(xmat);
    entry["reward"] = vector_array(trace.reward);
    entry["done"] = boolean_array(trace.done);
    traces.append(std::move(entry));
  }
  output["traces"] = std::move(traces);
  output["control_steps"] = result.control_steps;
  output["physics_steps"] = result.physics_steps;
  output["cancelled"] = result.cancelled;
  output["wall_seconds"] = result.wall_seconds;
  return output;
}

}  // namespace
}  // namespace native_eval

PYBIND11_MODULE(_native, module) {
  using namespace native_eval;
  module.doc() = "Native single-thread MuJoCo evaluation with native tasks and stateless MLP policies";
  py::class_<Evaluator>(module, "Evaluator")
      .def(py::init([](const std::string& xml_path, const py::dict& task,
                       const py::dict& policy, double physics_dt) {
        auto task_definition = parse_task(task);
        auto policy_spec = parse_policy(policy);
        Evaluator::TaskFactory factory = [definition = std::move(task_definition)](const mjModel* model) {
          if (definition.type == "legwheel") return make_legwheel_task(model, definition.parameters);
          return make_pendulum_task(model, definition.parameters);
        };
        py::gil_scoped_release release;
        return std::make_unique<Evaluator>(xml_path, std::move(factory),
                                           make_mlp_policy(std::move(policy_spec)), physics_dt);
      }), py::arg("xml_path"), py::arg("task"), py::arg("policy"), py::arg("physics_dt") = 0.0)
      .def("evaluate", [](Evaluator& evaluator, py::handle initial_states, const py::dict& options) {
        auto initial = parse_initial_states(initial_states, evaluator.nq(), evaluator.nv());
        auto parsed_options = parse_options(options);
        EvalResult result;
        {
          py::gil_scoped_release release;
          result = evaluator.evaluate(initial, parsed_options);
        }
        return result_dict(result, evaluator);
      }, py::arg("initial_states"), py::arg("options"))
      .def("policy_actions", [](const Evaluator& evaluator, py::handle observations) {
        auto parsed = parse_observations(observations, evaluator.observation_size());
        std::vector<std::vector<double>> actions;
        {
          py::gil_scoped_release release;
          actions = evaluator.policy_actions(parsed);
        }
        return action_array(actions, evaluator.action_size());
      }, py::arg("observations"))
      .def("cancel", &Evaluator::cancel, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("nq", &Evaluator::nq)
      .def_property_readonly("nv", &Evaluator::nv)
      .def_property_readonly("nu", &Evaluator::nu)
      .def_property_readonly("nbody", &Evaluator::nbody)
      .def_property_readonly("physics_dt", &Evaluator::physics_dt)
      .def_property_readonly("observation_size", &Evaluator::observation_size)
      .def_property_readonly("action_size", &Evaluator::action_size)
      .def_property_readonly("task_name", &Evaluator::task_name);

  module.def("policy_actions", [](const py::dict& descriptor, py::handle observations) {
    auto spec = parse_policy(descriptor);
    const std::size_t outputs = spec.action_size;
    auto parsed = parse_observations(observations, spec.mean.size());
    std::vector<std::vector<double>> actions;
    actions.reserve(parsed.size());
    {
      py::gil_scoped_release release;
      auto policy = make_mlp_policy(std::move(spec));
      for (const auto& observation : parsed) {
        actions.emplace_back();
        policy->infer(observation, actions.back());
      }
    }
    return action_array(actions, outputs);
  }, py::arg("policy"), py::arg("observations"));

  module.def("capabilities", []() {
    py::dict result;
    result["schema"] = 1;
    result["version"] = "0.1.0";
    result["tasks"] = py::cast(std::vector<std::string>{"legwheel", "pendulum"});
    result["policy_types"] = py::cast(std::vector<std::string>{"mlp"});
    result["activations"] = py::cast(std::vector<std::string>{"identity", "silu", "relu", "tanh"});
    result["output_activations"] = py::cast(std::vector<std::string>{"identity", "tanh"});
    result["skip_transforms"] = py::cast(std::vector<std::string>{"identity", "atanh"});
    result["stateless_policy"] = true;
    result["deterministic_policy"] = true;
    result["episode_metrics"] = true;
    result["native_control_threads"] = 1;
    return result;
  });
}
