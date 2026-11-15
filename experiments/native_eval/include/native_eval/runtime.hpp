#pragma once

#include <mujoco/mujoco.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace native_eval {

// These types contain no Python objects. All callbacks below are native C++.
struct TaskParameters {
  std::map<std::string, double> numbers;
  std::map<std::string, bool> flags;
  std::map<std::string, std::string> strings;
  std::map<std::string, std::vector<double>> vectors;
  std::map<std::string, std::vector<std::string>> names;
};

struct InitialState {
  std::vector<double> qpos;  // empty: use the XML default
  std::vector<double> qvel;  // empty: zero/default velocity
  std::vector<double> command;
  std::map<std::string, std::vector<double>> extra;
};

struct TaskState {
  virtual ~TaskState() = default;
};

struct StepOutcome {
  double reward = 0.0;
  bool terminated = false;
  std::string reason;
  // Per-control-step metrics. Names must be nonempty and cannot be "reward".
  std::map<std::string, double> metrics{};
};

class Task {
 public:
  virtual ~Task() = default;
  virtual std::string name() const = 0;
  virtual std::size_t observation_size() const = 0;
  virtual std::size_t action_size() const = 0;
  virtual std::unique_ptr<TaskState> make_state() const = 0;
  // The engine has reset mjData, applied qpos/qvel, and called mj_forward.
  // A task may sample its supported initial randomization when coordinates
  // were not supplied; it must forward updated coordinates before observing.
  virtual void reset(const mjModel* model, mjData* data, TaskState& state,
                     const InitialState& initial, std::uint64_t seed) const = 0;
  virtual void observe(const mjModel* model, const mjData* data,
                       const TaskState& state, std::vector<double>& out) const = 0;
  // Executes one control period including exactly substeps calls to mj_step.
  // Substep control/geometry/history stays in the task, never in Python.
  virtual StepOutcome advance(const mjModel* model, mjData* data, TaskState& state,
                              const std::vector<double>& action, int substeps) const = 0;
};

class Policy {
 public:
  virtual ~Policy() = default;
  virtual std::size_t observation_size() const = 0;
  virtual std::size_t action_size() const = 0;
  virtual void infer(const std::vector<double>& observation,
                     std::vector<double>& action) const = 0;
};

struct DenseLayer {
  std::size_t inputs = 0;
  std::size_t outputs = 0;
  std::vector<float> weights;  // row-major [outputs, inputs]
  std::vector<float> bias;
  std::string activation = "identity";
};

struct MlpSpec {
  std::vector<float> mean;
  std::vector<float> stddev;
  std::vector<DenseLayer> layers;
  std::size_t action_size = 0;
  std::string output_activation = "tanh";
  // Optional generic input skip: output_i = activation(scale_i*logit_i +
  // skip_scale_i*transform(raw_observation[skip_index_i])). -1 disables a skip.
  std::vector<int> skip_indices;
  std::vector<float> network_scales;
  std::vector<float> skip_scales;
  std::string skip_transform = "identity";
  float skip_clip_epsilon = 1e-5f;
};

std::unique_ptr<Policy> make_mlp_policy(MlpSpec spec);
std::unique_ptr<Task> make_legwheel_task(const mjModel* model, const TaskParameters& params);
std::unique_ptr<Task> make_pendulum_task(const mjModel* model, const TaskParameters& params);

struct EvalOptions {
  std::size_t num_envs = 1;
  std::size_t rollout_steps = 1;
  std::size_t episode_steps = 1;
  double control_dt = 0.01;
  std::uint64_t seed = 0;
  // Optional diagnostic world indices. Empty means no trajectory recording.
  std::vector<std::size_t> trace_worlds;
};

struct Trace {
  std::size_t world = 0;
  std::size_t steps = 0;
  std::vector<double> observation;  // pre-action, [steps, observation_size]
  std::vector<double> action;       // [steps, action_size]
  std::vector<double> qpos;         // raw post-step before automatic reset
  std::vector<double> qvel;
  std::vector<double> ctrl;         // raw final-substep actuator command
  std::vector<double> xmat;         // raw derived body rotations, [steps, nbody, 3, 3]
  std::vector<double> reward;
  std::vector<unsigned char> done;
};

struct EvalResult {
  std::vector<double> returns;
  std::vector<std::size_t> lengths;
  std::vector<unsigned char> terminated;
  std::vector<unsigned char> truncated;
  std::vector<std::string> reasons;
  std::vector<std::size_t> resets;
  std::vector<Trace> traces;
  std::size_t control_steps = 0;  // actual completed world/control transitions
  std::size_t physics_steps = 0;
  bool cancelled = false;
  double wall_seconds = 0.0;
  // Accumulated under the same first-episode mask as returns and lengths.
  std::map<std::string, std::vector<double>> episode_metrics{};
};

class Evaluator {
 public:
  using TaskFactory = std::function<std::unique_ptr<Task>(const mjModel*)>;
  Evaluator(const std::string& xml_path, TaskFactory task_factory,
            std::unique_ptr<Policy> policy, double physics_dt = 0.0);
  ~Evaluator();
  Evaluator(const Evaluator&) = delete;
  Evaluator& operator=(const Evaluator&) = delete;
  EvalResult evaluate(const std::vector<InitialState>& initial, const EvalOptions& options);
  std::vector<std::vector<double>> policy_actions(const std::vector<std::vector<double>>& obs) const;
  void cancel() noexcept;
  int nq() const;
  int nv() const;
  int nu() const;
  int nbody() const;
  double physics_dt() const;
  std::size_t observation_size() const;
  std::size_t action_size() const;
  std::string task_name() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace native_eval
