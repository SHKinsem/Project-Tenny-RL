#include "native_eval/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace native_eval {
namespace {

struct ModelDeleter {
  void operator()(mjModel* value) const noexcept { mj_deleteModel(value); }
};
struct DataDeleter {
  void operator()(mjData* value) const noexcept { mj_deleteData(value); }
};
using ModelPtr = std::unique_ptr<mjModel, ModelDeleter>;
using DataPtr = std::unique_ptr<mjData, DataDeleter>;

std::size_t checked_product(std::size_t a, std::size_t b, const char* label) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::invalid_argument(std::string(label) + " is too large");
  }
  return a * b;
}

void check_finite(const std::vector<double>& values, const char* label) {
  if (std::any_of(values.begin(), values.end(),
                  [](double value) { return !std::isfinite(value); })) {
    throw std::invalid_argument(std::string(label) + " must contain only finite values");
  }
}

std::uint64_t splitmix64(std::uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

std::uint64_t episode_seed(std::uint64_t seed, std::size_t world, std::size_t episode) {
  return splitmix64(seed + UINT64_C(0x9e3779b97f4a7c15) *
                              (static_cast<std::uint64_t>(world) + 1) +
                    UINT64_C(0xd1b54a32d192ed03) * static_cast<std::uint64_t>(episode));
}

int substep_count(double control_dt, double physics_dt) {
  if (!std::isfinite(control_dt) || control_dt <= 0.0) {
    throw std::invalid_argument("control_dt must be finite and positive");
  }
  const double ratio = control_dt / physics_dt;
  const double rounded = std::round(ratio);
  const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
                           std::max(1.0, std::abs(ratio));
  if (!std::isfinite(ratio) || rounded < 1.0 ||
      rounded > static_cast<double>(std::numeric_limits<int>::max()) ||
      std::abs(ratio - rounded) > tolerance) {
    throw std::invalid_argument("control_dt must be a positive integer multiple of physics_dt");
  }
  return static_cast<int>(rounded);
}

void check_initial(const InitialState& initial, const mjModel* model) {
  if (!initial.qpos.empty() && initial.qpos.size() != static_cast<std::size_t>(model->nq)) {
    throw std::invalid_argument("initial qpos has the wrong shape");
  }
  if (!initial.qvel.empty() && initial.qvel.size() != static_cast<std::size_t>(model->nv)) {
    throw std::invalid_argument("initial qvel has the wrong shape");
  }
  check_finite(initial.qpos, "initial qpos");
  check_finite(initial.qvel, "initial qvel");
  check_finite(initial.command, "initial command");
  for (const auto& item : initial.extra) check_finite(item.second, "initial extra");
}

struct World {
  DataPtr data;
  std::unique_ptr<TaskState> state;
  std::vector<double> observation;
  std::vector<double> action;
  std::size_t episode = 0;
  std::size_t episode_length = 0;
  bool first_episode_finished = false;
};

void reset_world(World& world, const mjModel* model, const Task& task,
                 const InitialState& initial, std::uint64_t seed) {
  mj_resetData(model, world.data.get());
  if (!initial.qpos.empty()) {
    std::copy(initial.qpos.begin(), initial.qpos.end(), world.data->qpos);
  }
  if (!initial.qvel.empty()) {
    std::copy(initial.qvel.begin(), initial.qvel.end(), world.data->qvel);
  }
  mj_forward(model, world.data.get());
  task.reset(model, world.data.get(), *world.state, initial, seed);
  world.episode_length = 0;
}

void reserve_trace(Trace& trace, std::size_t steps, const mjModel* model,
                   std::size_t observations, std::size_t actions) {
  trace.observation.reserve(checked_product(steps, observations, "observation trace"));
  trace.action.reserve(checked_product(steps, actions, "action trace"));
  trace.qpos.reserve(checked_product(steps, static_cast<std::size_t>(model->nq), "qpos trace"));
  trace.qvel.reserve(checked_product(steps, static_cast<std::size_t>(model->nv), "qvel trace"));
  trace.ctrl.reserve(checked_product(steps, static_cast<std::size_t>(model->nu), "ctrl trace"));
  trace.xmat.reserve(checked_product(
      steps, checked_product(static_cast<std::size_t>(model->nbody), 9, "body rotations"),
      "xmat trace"));
  trace.reward.reserve(steps);
  trace.done.reserve(steps);
}

void append_trace(Trace& trace, const World& world, const mjModel* model,
                  double reward, bool done) {
  trace.observation.insert(trace.observation.end(), world.observation.begin(), world.observation.end());
  trace.action.insert(trace.action.end(), world.action.begin(), world.action.end());
  if (model->nq) trace.qpos.insert(trace.qpos.end(), world.data->qpos, world.data->qpos + model->nq);
  if (model->nv) trace.qvel.insert(trace.qvel.end(), world.data->qvel, world.data->qvel + model->nv);
  if (model->nu) trace.ctrl.insert(trace.ctrl.end(), world.data->ctrl, world.data->ctrl + model->nu);
  const auto xmat_size = static_cast<std::size_t>(model->nbody) * 9;
  if (xmat_size) trace.xmat.insert(trace.xmat.end(), world.data->xmat, world.data->xmat + xmat_size);
  trace.reward.push_back(reward);
  trace.done.push_back(static_cast<unsigned char>(done));
  ++trace.steps;
}

void validate_metrics(const std::map<std::string, double>& metrics,
                      std::size_t num_envs, bool& schema_initialized,
                      EvalResult& result) {
  for (const auto& item : metrics) {
    if (item.first.empty() || item.first == "reward") {
      throw std::runtime_error("task metric names must be nonempty and cannot be reserved 'reward'");
    }
    if (!std::isfinite(item.second)) {
      throw std::runtime_error("task produced a non-finite metric: " + item.first);
    }
  }
  if (!schema_initialized) {
    for (const auto& item : metrics) {
      result.episode_metrics.emplace(item.first, std::vector<double>(num_envs, 0.0));
    }
    schema_initialized = true;
  }
  if (metrics.size() != result.episode_metrics.size()) {
    throw std::runtime_error("task metric fields must be identical for every world and control step");
  }
  auto expected = result.episode_metrics.begin();
  for (const auto& item : metrics) {
    if (item.first != expected->first) {
      throw std::runtime_error("task metric fields must be identical for every world and control step");
    }
    ++expected;
  }
}

}  // namespace

struct Evaluator::Impl {
  ModelPtr model;
  std::unique_ptr<Task> task;
  std::unique_ptr<Policy> policy;
  // A single state transition makes entering evaluate and accepting cancellation
  // race-free: 0 = idle, 1 = evaluating, 2 = cancellation requested.
  std::atomic<unsigned int> activity{0};
};

Evaluator::Evaluator(const std::string& xml_path, TaskFactory task_factory,
                     std::unique_ptr<Policy> policy, double physics_dt)
    : impl_(std::make_unique<Impl>()) {
  if (!std::isfinite(physics_dt) || physics_dt < 0.0) {
    throw std::invalid_argument("physics_dt must be zero or finite and positive");
  }
  if (!task_factory) throw std::invalid_argument("a native task factory is required");
  if (!policy) throw std::invalid_argument("a policy is required");
  char error[2048] = {};
  impl_->model.reset(mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error)));
  if (!impl_->model) {
    throw std::runtime_error("failed to load model XML '" + xml_path + "': " + error);
  }
  if (physics_dt > 0.0) impl_->model->opt.timestep = physics_dt;
  if (!std::isfinite(impl_->model->opt.timestep) || impl_->model->opt.timestep <= 0.0) {
    throw std::invalid_argument("model physics timestep must be finite and positive");
  }
  impl_->task = task_factory(impl_->model.get());
  if (!impl_->task) throw std::invalid_argument("native task factory returned no task");
  if (impl_->task->observation_size() == 0 || impl_->task->action_size() == 0) {
    throw std::invalid_argument("task observation and action dimensions must be positive");
  }
  if (policy->observation_size() != impl_->task->observation_size() ||
      policy->action_size() != impl_->task->action_size()) {
    throw std::invalid_argument("policy dimensions do not match the selected task");
  }
  impl_->policy = std::move(policy);
}

Evaluator::~Evaluator() = default;

EvalResult Evaluator::evaluate(const std::vector<InitialState>& initial,
                               const EvalOptions& options) {
  unsigned int expected = 0;
  if (!impl_->activity.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
    throw std::runtime_error("evaluate is already running on this Evaluator");
  }
  struct ActivityGuard {
    std::atomic<unsigned int>& state;
    ~ActivityGuard() { state.store(0, std::memory_order_release); }
  } guard{impl_->activity};
  const auto started = std::chrono::steady_clock::now();
  if (options.num_envs == 0 || options.rollout_steps == 0 || options.episode_steps == 0) {
    throw std::invalid_argument("num_envs, rollout_steps, and episode_steps must be positive");
  }
  if (initial.size() != 1 && initial.size() != options.num_envs) {
    throw std::invalid_argument("initial_states must have length one or num_envs");
  }
  const mjModel* model = impl_->model.get();
  const int substeps = substep_count(options.control_dt, model->opt.timestep);
  const auto total_controls = checked_product(options.num_envs, options.rollout_steps, "workload");
  checked_product(total_controls, static_cast<std::size_t>(substeps), "physics workload");
  for (const auto& entry : initial) check_initial(entry, model);

  EvalResult result;
  result.returns.assign(options.num_envs, 0.0);
  result.lengths.assign(options.num_envs, 0);
  result.terminated.assign(options.num_envs, 0);
  result.truncated.assign(options.num_envs, 0);
  result.reasons.resize(options.num_envs);
  result.resets.assign(options.num_envs, 0);
  bool metric_schema_initialized = false;
  const std::size_t no_trace = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> trace_indices(options.num_envs, no_trace);
  result.traces.reserve(options.trace_worlds.size());
  for (const auto index : options.trace_worlds) {
    if (index >= options.num_envs) throw std::invalid_argument("trace_world index is outside num_envs");
    if (trace_indices[index] != no_trace) throw std::invalid_argument("trace_worlds contains duplicates");
    trace_indices[index] = result.traces.size();
    result.traces.emplace_back();
    result.traces.back().world = index;
    reserve_trace(result.traces.back(), options.rollout_steps, model,
                  observation_size(), action_size());
  }

  std::vector<World> worlds;
  worlds.reserve(options.num_envs);
  for (std::size_t index = 0; index < options.num_envs; ++index) {
    if (impl_->activity.load(std::memory_order_acquire) == 2) break;
    World world;
    world.data.reset(mj_makeData(model));
    if (!world.data) throw std::runtime_error("MuJoCo could not allocate world data");
    world.state = impl_->task->make_state();
    if (!world.state) throw std::runtime_error("native task returned no world state");
    world.observation.reserve(observation_size());
    world.action.reserve(action_size());
    reset_world(world, model, *impl_->task, initial[initial.size() == 1 ? 0 : index],
                episode_seed(options.seed, index, 0));
    worlds.push_back(std::move(world));
  }

  for (std::size_t step = 0; step < options.rollout_steps; ++step) {
    if (impl_->activity.load(std::memory_order_acquire) == 2) break;
    for (std::size_t index = 0; index < worlds.size(); ++index) {
      if (impl_->activity.load(std::memory_order_acquire) == 2) break;
      auto& world = worlds[index];
      impl_->task->observe(model, world.data.get(), *world.state, world.observation);
      if (world.observation.size() != observation_size()) {
        throw std::runtime_error("task produced an observation with the wrong shape");
      }
      check_finite(world.observation, "task observation");
      impl_->policy->infer(world.observation, world.action);
      if (world.action.size() != action_size()) {
        throw std::runtime_error("policy produced an action with the wrong shape");
      }
      check_finite(world.action, "policy action");
      const StepOutcome outcome = impl_->task->advance(
          model, world.data.get(), *world.state, world.action, substeps);
      if (!std::isfinite(outcome.reward)) throw std::runtime_error("task produced a non-finite reward");
      validate_metrics(outcome.metrics, options.num_envs, metric_schema_initialized, result);
      ++world.episode_length;
      ++result.control_steps;
      result.physics_steps += static_cast<std::size_t>(substeps);
      const bool time_limit = world.episode_length >= options.episode_steps;
      const bool done = outcome.terminated || time_limit;
      if (!world.first_episode_finished) {
        result.returns[index] += outcome.reward;
        ++result.lengths[index];
        auto total = result.episode_metrics.begin();
        for (const auto& metric : outcome.metrics) {
          total->second[index] += metric.second;
          if (!std::isfinite(total->second[index])) {
            throw std::runtime_error("task episode metric accumulation is non-finite: " + metric.first);
          }
          ++total;
        }
        if (done) {
          result.terminated[index] = static_cast<unsigned char>(outcome.terminated);
          result.truncated[index] = static_cast<unsigned char>(!outcome.terminated && time_limit);
          result.reasons[index] = outcome.terminated ? outcome.reason : "time_limit";
          world.first_episode_finished = true;
        }
      }
      if (trace_indices[index] != no_trace) {
        append_trace(result.traces[trace_indices[index]], world, model, outcome.reward, done);
      }
      if (done) {
        ++world.episode;
        ++result.resets[index];
        reset_world(world, model, *impl_->task, initial[initial.size() == 1 ? 0 : index],
                    episode_seed(options.seed, index, world.episode));
      }
    }
  }
  result.cancelled = result.control_steps < total_controls &&
                     impl_->activity.load(std::memory_order_acquire) == 2;
  result.wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  return result;
}

std::vector<std::vector<double>> Evaluator::policy_actions(
    const std::vector<std::vector<double>>& observations) const {
  std::vector<std::vector<double>> result;
  result.reserve(observations.size());
  for (const auto& observation : observations) {
    if (observation.size() != observation_size()) {
      throw std::invalid_argument("policy observation has the wrong shape");
    }
    check_finite(observation, "policy observation");
    result.emplace_back();
    impl_->policy->infer(observation, result.back());
    if (result.back().size() != action_size()) {
      throw std::runtime_error("policy produced an action with the wrong shape");
    }
    check_finite(result.back(), "policy action");
  }
  return result;
}

void Evaluator::cancel() noexcept {
  unsigned int expected = 1;
  impl_->activity.compare_exchange_strong(expected, 2, std::memory_order_acq_rel);
}

int Evaluator::nq() const { return impl_->model->nq; }
int Evaluator::nv() const { return impl_->model->nv; }
int Evaluator::nu() const { return impl_->model->nu; }
int Evaluator::nbody() const { return impl_->model->nbody; }
double Evaluator::physics_dt() const { return impl_->model->opt.timestep; }
std::size_t Evaluator::observation_size() const { return impl_->task->observation_size(); }
std::size_t Evaluator::action_size() const { return impl_->task->action_size(); }
std::string Evaluator::task_name() const { return impl_->task->name(); }

}  // namespace native_eval
