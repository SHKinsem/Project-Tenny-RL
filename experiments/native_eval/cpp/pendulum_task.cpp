#include "native_eval/runtime.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace native_eval {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument("pendulum: " + message);
}

struct PendulumConfig {
  std::string joint = "hinge";
  std::string actuator = "motor";
  double torque_scale = 1.0;
  double velocity_cost = 0.1;
  double action_cost = 0.001;
  double target = 0.0;
  double terminate_angle = 0.0;  // Zero disables termination.
  double initial_angle_range = 0.0;
  double initial_velocity_range = 0.0;
};

PendulumConfig parse_config(const TaskParameters& params) {
  const std::set<std::string> numeric = {
      "torque_scale", "velocity_cost", "action_cost", "target", "terminate_angle",
      "initial_angle_range", "initial_velocity_range"};
  for (const auto& entry : params.numbers) {
    if (!numeric.count(entry.first)) invalid("unknown numeric setting '" + entry.first + "'");
    if (!std::isfinite(entry.second)) invalid("setting '" + entry.first + "' must be finite");
  }
  for (const auto& entry : params.strings) {
    if (entry.first != "joint" && entry.first != "actuator") {
      invalid("unknown string setting '" + entry.first + "'");
    }
    if (entry.second.empty()) invalid("setting '" + entry.first + "' must not be empty");
  }
  for (const auto& entry : params.flags) {
    if (entry.first != "terminate_angle" || entry.second) {
      invalid("unsupported boolean setting '" + entry.first + "'");
    }
    if (params.numbers.count(entry.first)) invalid("duplicate setting '" + entry.first + "'");
  }
  if (!params.vectors.empty()) invalid("unsupported vector setting '" + params.vectors.begin()->first + "'");
  if (!params.names.empty()) invalid("unsupported names setting '" + params.names.begin()->first + "'");

  PendulumConfig cfg;
  const auto number = [&params](const std::string& key, double fallback) {
    const auto it = params.numbers.find(key);
    return it == params.numbers.end() ? fallback : it->second;
  };
  const auto name = [&params](const std::string& key, const std::string& fallback) {
    const auto it = params.strings.find(key);
    return it == params.strings.end() ? fallback : it->second;
  };
  cfg.joint = name("joint", cfg.joint);
  cfg.actuator = name("actuator", cfg.actuator);
  cfg.torque_scale = number("torque_scale", cfg.torque_scale);
  cfg.velocity_cost = number("velocity_cost", cfg.velocity_cost);
  cfg.action_cost = number("action_cost", cfg.action_cost);
  cfg.target = number("target", cfg.target);
  cfg.initial_angle_range = number("initial_angle_range", cfg.initial_angle_range);
  cfg.initial_velocity_range = number("initial_velocity_range", cfg.initial_velocity_range);
  if (params.numbers.count("terminate_angle")) {
    cfg.terminate_angle = number("terminate_angle", 0.0);
    if (!(cfg.terminate_angle > 0.0 && cfg.terminate_angle <= kPi)) {
      invalid("terminate_angle must be in (0, pi], or false to disable");
    }
  }
  if (!(cfg.torque_scale > 0.0)) invalid("torque_scale must be positive");
  if (cfg.velocity_cost < 0.0) invalid("velocity_cost must be nonnegative");
  if (cfg.action_cost < 0.0) invalid("action_cost must be nonnegative");
  if (cfg.initial_angle_range < 0.0) invalid("initial_angle_range must be nonnegative");
  if (cfg.initial_velocity_range < 0.0) invalid("initial_velocity_range must be nonnegative");
  return cfg;
}

struct PendulumState final : TaskState {};

class PendulumTask final : public Task {
 public:
  PendulumTask(const mjModel* model, PendulumConfig config) : config_(std::move(config)) {
    if (!model) invalid("model is null");
    // This model assumption belongs to this task, never to the evaluation engine.
    if (model->nq != 1 || model->nv != 1 || model->nu != 1) {
      invalid("requires a model with nq=nv=nu=1");
    }
    joint_ = mj_name2id(model, mjOBJ_JOINT, config_.joint.c_str());
    if (joint_ < 0) invalid("joint '" + config_.joint + "' does not exist");
    if (model->jnt_type[joint_] != mjJNT_HINGE) invalid("configured joint must be a hinge");
    actuator_ = mj_name2id(model, mjOBJ_ACTUATOR, config_.actuator.c_str());
    if (actuator_ < 0) invalid("actuator '" + config_.actuator + "' does not exist");
    if (model->actuator_trntype[actuator_] != mjTRN_JOINT ||
        model->actuator_trnid[2 * actuator_] != joint_ ||
        model->actuator_dyntype[actuator_] != mjDYN_NONE ||
        model->actuator_gaintype[actuator_] != mjGAIN_FIXED ||
        model->actuator_biastype[actuator_] != mjBIAS_NONE) {
      invalid("configured actuator must be a torque motor on the configured joint");
    }
    qpos_ = model->jnt_qposadr[joint_];
    qvel_ = model->jnt_dofadr[joint_];
  }

  std::string name() const override { return "pendulum"; }
  std::size_t observation_size() const override { return 3; }
  std::size_t action_size() const override { return 1; }
  std::unique_ptr<TaskState> make_state() const override {
    return std::make_unique<PendulumState>();
  }

  void reset(const mjModel* model, mjData* data, TaskState&,
             const InitialState& initial, std::uint64_t seed) const override {
    if (!initial.command.empty()) invalid("command is not supported by this task");
    if (!initial.extra.empty()) invalid("extra initial-state fields are not supported by this task");
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    bool changed = false;
    if (initial.qpos.empty() && config_.initial_angle_range > 0.0) {
      data->qpos[qpos_] = model->qpos0[qpos_] + config_.initial_angle_range * unit(rng);
      changed = true;
    }
    if (initial.qvel.empty() && config_.initial_velocity_range > 0.0) {
      data->qvel[qvel_] = config_.initial_velocity_range * unit(rng);
      changed = true;
    }
    data->ctrl[actuator_] = 0.0;
    if (changed) mj_forward(model, data);
  }

  void observe(const mjModel*, const mjData* data, const TaskState&,
               std::vector<double>& out) const override {
    const double angle = data->qpos[qpos_] - config_.target;
    out = {std::cos(angle), std::sin(angle), data->qvel[qvel_]};
  }

  StepOutcome advance(const mjModel* model, mjData* data, TaskState&,
                      const std::vector<double>& action, int substeps) const override {
    if (action.size() != 1 || !std::isfinite(action[0])) invalid("action must have one finite value");
    if (substeps <= 0) invalid("substeps must be positive");
    const double command = config_.torque_scale * action[0];
    if (!std::isfinite(command)) invalid("scaled actuator command must be finite");
    data->ctrl[actuator_] = command;
    for (int step = 0; step < substeps; ++step) mj_step(model, data);
    const double angle = data->qpos[qpos_] - config_.target;
    const double velocity = data->qvel[qvel_];
    StepOutcome outcome;
    outcome.reward = std::cos(angle) - config_.velocity_cost * velocity * velocity -
                     config_.action_cost * action[0] * action[0];
    outcome.terminated = config_.terminate_angle > 0.0 &&
                         std::abs(std::remainder(angle, 2.0 * kPi)) > config_.terminate_angle;
    if (outcome.terminated) outcome.reason = "angle_limit";
    return outcome;
  }

 private:
  PendulumConfig config_;
  int joint_ = -1;
  int actuator_ = -1;
  int qpos_ = 0;
  int qvel_ = 0;
};

}  // namespace

std::unique_ptr<Task> make_pendulum_task(const mjModel* model, const TaskParameters& params) {
  return std::make_unique<PendulumTask>(model, parse_config(params));
}

}  // namespace native_eval
