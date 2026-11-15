#include "native_eval/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace native_eval {
namespace {

using Vec3 = std::array<double, 3>;
using Vec4 = std::array<double, 4>;
using Mat3 = std::array<double, 9>;

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument("legwheel: " + message);
}

template <class Values>
void reject_unknown(const Values& values, const std::set<std::string>& accepted) {
  for (const auto& item : values) {
    if (!accepted.count(item.first)) invalid("unknown or incorrectly typed config key: " + item.first);
  }
}

double number(const TaskParameters& p, const std::string& key, double fallback) {
  auto found = p.numbers.find(key);
  const double value = found == p.numbers.end() ? fallback : found->second;
  if (!std::isfinite(value)) invalid("nonfinite config value: " + key);
  return value;
}

bool flag(const TaskParameters& p, const std::string& key, bool fallback) {
  auto found = p.flags.find(key);
  return found == p.flags.end() ? fallback : found->second;
}

std::string string_value(const TaskParameters& p, const std::string& key,
                         const std::string& fallback = "") {
  auto found = p.strings.find(key);
  const auto value = found == p.strings.end() ? fallback : found->second;
  if (value.empty()) invalid("missing or empty config string: " + key);
  return value;
}

template <std::size_t N>
std::array<double, N> vector_value(const TaskParameters& p, const std::string& key) {
  auto found = p.vectors.find(key);
  if (found == p.vectors.end() || found->second.size() != N)
    invalid(key + " must have " + std::to_string(N) + " numeric entries");
  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = found->second[i];
    if (!std::isfinite(out[i])) invalid("nonfinite config vector: " + key);
  }
  return out;
}

template <std::size_t N>
std::array<int, N> named_ids(const mjModel* model, const TaskParameters& p,
                           const std::string& key, mjtObj type) {
  const auto found = p.names.find(key);
  if (found == p.names.end() || found->second.size() != N)
    invalid(key + " must have " + std::to_string(N) + " model names");
  std::array<int, N> ids{};
  std::set<int> unique;
  for (std::size_t i = 0; i < N; ++i) {
    ids[i] = mj_name2id(model, type, found->second[i].c_str());
    if (ids[i] < 0) invalid("missing " + key + " name: " + found->second[i]);
    if (!unique.insert(ids[i]).second) invalid("duplicate model name in " + key);
  }
  return ids;
}

int named_id(const mjModel* model, const TaskParameters& p,
             const std::string& key, mjtObj type) {
  const auto name = string_value(p, key);
  const int id = mj_name2id(model, type, name.c_str());
  if (id < 0) invalid("missing model name for " + key + ": " + name);
  return id;
}

double dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

Vec3 matvec(const Mat3& m, const Vec3& v) {
  return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
          m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
          m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

Mat3 rotation(const Vec3& axis, double angle) {
  const double c = std::cos(angle), s = std::sin(angle);
  const double x = axis[0], y = axis[1], z = axis[2];
  return {c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s,
          y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s,
          z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)};
}

// Match array min/max reductions, including NaN propagation. Invalid physics
// must not turn into apparently finite diagnostic values.
double diagnostic_max(double a, double b) {
  return std::isnan(a) || std::isnan(b) ? std::numeric_limits<double>::quiet_NaN() : std::max(a, b);
}

double diagnostic_min(double a, double b) {
  return std::isnan(a) || std::isnan(b) ? std::numeric_limits<double>::quiet_NaN() : std::min(a, b);
}

struct LegGeometry {
  double hip_ref, knee_ref;
  std::array<double, 5> poly{};
  Vec3 knee_offset{}, knee_anchor{}, knee_axis{}, wheel_center{};
  Mat3 knee_rotation{};

  explicit LegGeometry(const std::array<double, 28>& flat)
      : hip_ref(flat[0]), knee_ref(flat[1]) {
    std::copy_n(flat.begin() + 2, 5, poly.begin());
    std::copy_n(flat.begin() + 7, 3, knee_offset.begin());
    std::copy_n(flat.begin() + 10, 9, knee_rotation.begin());
    std::copy_n(flat.begin() + 19, 3, knee_anchor.begin());
    std::copy_n(flat.begin() + 22, 3, knee_axis.begin());
    std::copy_n(flat.begin() + 25, 3, wheel_center.begin());
    if (std::abs(dot(knee_axis, knee_axis) - 1.0) > 1e-8)
      invalid("geometry knee axis must be normalized");
  }

  double coupled_knee(double qh) const {
    const double x = qh - hip_ref;
    return knee_ref + poly[0] + x * (poly[1] + x * (poly[2] + x * (poly[3] + x * poly[4])));
  }

  double slope(double qh) const {
    const double x = qh - hip_ref;
    return poly[1] + x * (2 * poly[2] + x * (3 * poly[3] + x * 4 * poly[4]));
  }

  std::array<double, 3> evaluate(double qh, double qk) const {
    Vec3 v{}, inner{}, position{};
    for (int i = 0; i < 3; ++i) v[i] = wheel_center[i] - knee_anchor[i];
    const auto rotated = matvec(rotation(knee_axis, qk - knee_ref), v);
    for (int i = 0; i < 3; ++i) inner[i] = knee_anchor[i] + rotated[i];
    const auto outer = matvec(knee_rotation, inner);
    for (int i = 0; i < 3; ++i) position[i] = knee_offset[i] + outer[i];
    const auto derivative = matvec(knee_rotation, cross(knee_axis, rotated));
    const double length = std::sqrt(dot(position, position));
    const double partial = dot(position, derivative) / (length + 1e-15);
    return {length, slope(qh) * partial, partial};
  }
};

struct LegwheelState final : TaskState {
  Vec4 command{}, last_sent{}, last_action{}, last_last_action{};
  Vec4 last_actual{}, last_last_actual{};
  double position_start = 0, sinyaw_start = 0, sin_balance_zero = 0;
};

class LegwheelTask final : public Task {
 public:
  LegwheelTask(const mjModel* model, const TaskParameters& p)
      : legs_{LegGeometry(vector_value<28>(p, "leg_geometry_left")),
              LegGeometry(vector_value<28>(p, "leg_geometry_right"))},
        jmax_(vector_value<2>(p, "jmax")),
        force_scales_(vector_value<2>(p, "force_scales")),
        torque_scales_(vector_value<4>(p, "torque_scales")),
        hip_range_(vector_value<2>(p, "hip_range")),
        command_scales_(vector_value<4>(p, "command_scales")),
        thigh_range_(vector_value<2>(p, "thigh_range")) {
    reject_unknown(p.numbers, {"reference_dt", "wheel_radius", "sinyaw_scale",
      "cmd_lin_vel_x_zero_margin", "cmd_ang_vel_zero_margin", "sigma", "sigma_wheel",
      "height_scale", "vel_scale", "roll_scale", "pitch_scale", "zero_keep_scale",
      "energy_scale", "action_rate_scale", "pitch_velocity_penalty_scale",
      "roll_thigh_coupling", "equality_position_limit", "length_velocity_limit",
      "minimum_jacobian_ratio", "pitch_min", "pitch_max", "roll_limit", "noise_level",
      "command_probability", "action_delay_steps", "action_repeat", "obs_history_len"});
    reject_unknown(p.flags, {"sinpitch_zero", "sinyaw", "wheel_sinpos_start", "enable_lin_vel_x",
      "enable_ang_vel_yaw", "enable_height_value", "enable_roll_value", "height_feasible_projection",
      "push_enabled", "domain_randomization", "last_act_obs", "hold_error_clip",
      "obs_normalization", "termination_penalty", "do_deconflict"});
    reject_unknown(p.strings, {"root_body", "gyro_sensor", "local_linvel_sensor", "action_mode",
                               "geometry_schema", "observation_version"});
    reject_unknown(p.vectors, {"leg_geometry_left", "leg_geometry_right", "jmax", "force_scales",
                               "torque_scales", "hip_range", "command_scales", "thigh_range"});
    reject_unknown(p.names, {"hip_joints", "knee_joints", "wheel_joints", "wheel_bodies", "actuators"});
    for (const auto& key : {"noise_level", "command_probability", "action_delay_steps"})
      if (number(p, key, 0) != 0) invalid(std::string(key) + " is unsupported");
    for (const auto& key : {"action_repeat", "obs_history_len"})
      if (number(p, key, 1) != 1) invalid(std::string(key) + " must equal one");
    for (const auto& key : {"push_enabled", "domain_randomization", "last_act_obs", "hold_error_clip",
                            "obs_normalization", "termination_penalty", "do_deconflict"})
      if (flag(p, key, false)) invalid(std::string(key) + " is unsupported");
    if (string_value(p, "geometry_schema", "wheel-center-axial-v1") != "wheel-center-axial-v1")
      invalid("unsupported geometry_schema");
    if (string_value(p, "observation_version", "torque24-mean-sent-v1") != "torque24-mean-sent-v1")
      invalid("unsupported observation_version");
    mode_ = string_value(p, "action_mode");
    if (mode_ != "virtual_leg_force" && mode_ != "direct_torque" && mode_ != "matched_envelope")
      invalid("unsupported action_mode: " + mode_);

    reference_dt_ = number(p, "reference_dt", .01);
    wheel_radius_ = number(p, "wheel_radius", .0195);
    sinyaw_scale_ = number(p, "sinyaw_scale", .5);
    linear_margin_ = number(p, "cmd_lin_vel_x_zero_margin", .000195);
    angular_margin_ = number(p, "cmd_ang_vel_zero_margin", .01);
    sigma_ = number(p, "sigma", .5);
    sigma_wheel_ = number(p, "sigma_wheel", .5);
    height_scale_ = number(p, "height_scale", 2);
    vel_scale_ = number(p, "vel_scale", 2);
    roll_scale_ = number(p, "roll_scale", 4);
    pitch_scale_ = number(p, "pitch_scale", 0);
    zero_keep_scale_ = number(p, "zero_keep_scale", 10);
    energy_scale_ = number(p, "energy_scale", -1);
    action_rate_scale_ = number(p, "action_rate_scale", -.5);
    pitch_velocity_scale_ = number(p, "pitch_velocity_penalty_scale", -.5);
    roll_coupling_ = number(p, "roll_thigh_coupling", .433);
    equality_limit_ = number(p, "equality_position_limit", .01);
    length_velocity_limit_ = number(p, "length_velocity_limit", .05);
    minimum_ratio_ = number(p, "minimum_jacobian_ratio", .05);
    pitch_min_ = number(p, "pitch_min", -.6);
    pitch_max_ = number(p, "pitch_max", .85);
    roll_limit_ = number(p, "roll_limit", .6);
    balance_zero_ = flag(p, "sinpitch_zero", true);
    yaw_hold_ = flag(p, "sinyaw", true);
    position_hold_ = flag(p, "wheel_sinpos_start", true);
    enable_linear_ = flag(p, "enable_lin_vel_x", true);
    enable_angular_ = flag(p, "enable_ang_vel_yaw", true);
    enable_height_ = flag(p, "enable_height_value", true);
    enable_roll_ = flag(p, "enable_roll_value", true);
    height_projection_ = flag(p, "height_feasible_projection", true);
    if (!(reference_dt_ > 0 && wheel_radius_ > 0 && sinyaw_scale_ > 0 && sigma_ > 0 &&
          sigma_wheel_ > 0 && equality_limit_ > 0 && length_velocity_limit_ > 0 &&
          minimum_ratio_ > 0 && minimum_ratio_ <= 1 && roll_limit_ > 0 && roll_limit_ <= 1 &&
          linear_margin_ >= 0 && angular_margin_ >= 0 && roll_coupling_ >= 0 &&
          pitch_min_ < pitch_max_ && pitch_min_ >= -1 && pitch_max_ <= 1))
      invalid("invalid reward/validity scales or limits");
    if (!(hip_range_[0] < hip_range_[1] && thigh_range_[0] < thigh_range_[1]))
      invalid("invalid hip/thigh interval");
    for (double value : jmax_) if (!(value > 0)) invalid("jmax must be positive");
    for (double value : force_scales_) if (!(value > 0)) invalid("force_scales must be positive");
    for (double value : torque_scales_) if (!(value > 0)) invalid("torque_scales must be positive");
    for (double value : command_scales_) if (!(value > 0)) invalid("command_scales must be positive");

    const auto hips = named_ids<2>(model, p, "hip_joints", mjOBJ_JOINT);
    const auto knees = named_ids<2>(model, p, "knee_joints", mjOBJ_JOINT);
    const auto wheels = named_ids<2>(model, p, "wheel_joints", mjOBJ_JOINT);
    wheel_body_ = named_ids<2>(model, p, "wheel_bodies", mjOBJ_BODY);
    actuator_ = named_ids<4>(model, p, "actuators", mjOBJ_ACTUATOR);
    root_body_ = named_id(model, p, "root_body", mjOBJ_BODY);
    const int gyro = named_id(model, p, "gyro_sensor", mjOBJ_SENSOR);
    const int local_velocity = named_id(model, p, "local_linvel_sensor", mjOBJ_SENSOR);
    if (model->sensor_dim[gyro] != 3 || model->sensor_dim[local_velocity] != 3 ||
        model->sensor_type[gyro] != mjSENS_GYRO ||
        model->sensor_type[local_velocity] != mjSENS_VELOCIMETER)
      invalid("expected a three-component gyro and local velocimeter sensor");
    gyro_ = model->sensor_adr[gyro];
    local_velocity_ = model->sensor_adr[local_velocity];
    std::set<int> all_joints;
    for (int side = 0; side < 2; ++side) {
      for (const int joint : {hips[side], knees[side], wheels[side]}) {
        if (model->jnt_type[joint] != mjJNT_HINGE || !all_joints.insert(joint).second)
          invalid("hip/knee/wheel joints must be six distinct hinges");
      }
      hip_q_[side] = model->jnt_qposadr[hips[side]];
      knee_q_[side] = model->jnt_qposadr[knees[side]];
      wheel_q_[side] = model->jnt_qposadr[wheels[side]];
      hip_v_[side] = model->jnt_dofadr[hips[side]];
      knee_v_[side] = model->jnt_dofadr[knees[side]];
      wheel_v_[side] = model->jnt_dofadr[wheels[side]];
      validate_geometry(model, side, hips[side], knees[side], wheels[side]);
    }
    if (model->nu != 4) invalid("adapter supports exactly four named motors");
    const std::array<int, 4> driven_joints{hips[0], hips[1], wheels[0], wheels[1]};
    actual_v_ = {hip_v_[0], hip_v_[1], wheel_v_[0], wheel_v_[1]};
    for (int index = 0; index < 4; ++index) {
      const int act = actuator_[index];
      if (model->actuator_trntype[act] != mjTRN_JOINT ||
          model->actuator_trnid[2 * act] != driven_joints[index] ||
          model->actuator_dyntype[act] != mjDYN_NONE ||
          model->actuator_gaintype[act] != mjGAIN_FIXED ||
          model->actuator_biastype[act] != mjBIAS_NONE ||
          model->actuator_gainprm[mjNGAIN * act] != 1 ||
          model->actuator_gear[6 * act] != 1)
        invalid("actuators must be stateless unit-gain, unit-gear joint motors");
      for (int k = 0; k < mjNBIAS; ++k)
        if (model->actuator_biasprm[mjNBIAS * act + k] != 0)
          invalid("motor bias parameters must be zero");
      for (int k = 1; k < 6; ++k)
        if (model->actuator_gear[6 * act + k] != 0)
          invalid("motor gear must have only its first component set");
      if ((model->actuator_ctrllimited[act] &&
           (model->actuator_ctrlrange[2 * act] > -torque_scales_[index] ||
            model->actuator_ctrlrange[2 * act + 1] < torque_scales_[index])) ||
          (model->actuator_forcelimited[act] &&
           (model->actuator_forcerange[2 * act] > -torque_scales_[index] ||
            model->actuator_forcerange[2 * act + 1] < torque_scales_[index])))
        invalid("motor ctrl/force range is narrower than configured torque scales");
    }
  }

  std::string name() const override { return "legwheel"; }
  std::size_t observation_size() const override { return 24; }
  std::size_t action_size() const override { return 4; }
  std::unique_ptr<TaskState> make_state() const override {
    return std::make_unique<LegwheelState>();
  }

  void reset(const mjModel*, mjData* data, TaskState& state,
             const InitialState& initial, std::uint64_t) const override {
    auto& s = dynamic_cast<LegwheelState&>(state);
    s = LegwheelState{};
    if (!initial.command.empty()) {
      if (initial.command.size() != 4) invalid("initial command must have four values");
      for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(initial.command[i])) invalid("initial command must be finite");
        s.command[i] = initial.command[i];
      }
    }
    for (const auto& item : initial.extra)
      if (item.first != "position_start" && item.first != "sinyaw_start" && item.first != "sin_balance_zero")
        invalid("unsupported initial extra: " + item.first);
    s.position_start = anchor(initial, "position_start", hold_position(data));
    s.sinyaw_start = anchor(initial, "sinyaw_start", sinyaw(data));
    s.sin_balance_zero = anchor(initial, "sin_balance_zero", balance(data));
  }

  void observe(const mjModel*, const mjData* data, const TaskState& state,
               std::vector<double>& out) const override {
    const auto& s = dynamic_cast<const LegwheelState&>(state);
    out.resize(24);
    std::copy(s.command.begin(), s.command.end(), out.begin());
    std::copy_n(data->sensordata + gyro_, 3, out.begin() + 4);
    out[7] = sinyaw(data) - (yaw_hold_ ? s.sinyaw_start : 0);
    out[8] = sinpitch(data) - s.sin_balance_zero;
    out[9] = sinroll(data) - std::sin(s.command[3]);
    const double ds = hold_position(data) - (position_hold_ ? s.position_start : 0);
    out[10] = data->qpos[hip_q_[0]];
    out[11] = ds;
    out[12] = data->qpos[hip_q_[1]];
    out[13] = ds;
    out[14] = data->qvel[hip_v_[0]];
    out[15] = data->qvel[wheel_v_[0]];
    out[16] = data->qvel[hip_v_[1]];
    out[17] = data->qvel[wheel_v_[1]];
    out[18] = data->qvel[wheel_v_[0]] * wheel_radius_;
    out[19] = data->qvel[wheel_v_[1]] * wheel_radius_;
    std::copy(s.last_sent.begin(), s.last_sent.end(), out.begin() + 20);
  }

  StepOutcome advance(const mjModel* model, mjData* data, TaskState& state,
                      const std::vector<double>& action, int substeps) const override {
    if (action.size() != 4) invalid("action must have four values");
    if (substeps < 1) invalid("physics substep count must be positive");
    auto& s = dynamic_cast<LegwheelState&>(state);
    bool latched = false;
    std::string reason;
    StepOutcome outcome;
    auto& metrics = outcome.metrics;
    std::array<double, 8> diagnostics{0, 0, 0, 0, 0, 0, 0,
                                      std::numeric_limits<double>::infinity()};
    Vec4 clipped{}, sent_sum{};
    for (int i = 0; i < 4; ++i) {
      if (!std::isfinite(action[i])) { latched = true; reason = "nonfinite_action"; }
      clipped[i] = std::clamp(action[i], -1.0, 1.0);
    }
    const double ratio = reference_dt_ / model->opt.timestep;
    double effort_sum = 0, rate_sum = 0;
    for (int step = 0; step < substeps; ++step) {
      const auto geometry = geometry_state(model, data);
      if (!geometry.reason.empty()) {
        latched = true;
        if (reason.empty()) reason = geometry.reason;
      }
      double saturated = 0;
      for (int i = 0; i < 4; ++i) {
        double gain = torque_scales_[i];
        if (i < 2 && mode_ != "direct_torque") {
          const double j = std::isfinite(geometry.jac[i]) ? geometry.jac[i] : 0;
          gain = (mode_ == "matched_envelope" ? std::abs(j) : j) * force_scales_[i];
        }
        const double raw = latched ? 0.0 : gain * clipped[i];
        const double torque = latched ? 0.0 : std::clamp(raw, -torque_scales_[i], torque_scales_[i]);
        saturated += std::abs(raw) > torque_scales_[i] ? .25 : 0;
        data->ctrl[actuator_[i]] = torque;
        sent_sum[i] += torque;
      }
      // As in the native production adapter, derived xmat/sensors remain at
      // MuJoCo's mj_step stage; an extra mj_forward here changes the observation.
      mj_step(model, data);
      for (int i = 0; i < 4; ++i) {
        const double actual = data->qfrc_actuator[actual_v_[i]] / torque_scales_[i];
        if (!std::isfinite(actual)) {
          latched = true;
          if (reason.empty()) reason = "nonfinite_actual_torque";
        }
        effort_sum += actual * actual;
        rate_sum += std::abs(actual - s.last_actual[i]) * ratio +
                    std::abs(actual + s.last_last_actual[i] - 2 * s.last_actual[i]) * ratio * ratio;
        s.last_last_actual[i] = s.last_actual[i];
        s.last_actual[i] = actual;
      }
      // Production diagnostics observe pre-substep geometry and the latch after
      // actual torque has been checked. Final post-step guards are separate.
      const std::array<double, 8> step_diagnostics{
        saturated, geometry.deadpoint_fraction, geometry.outside ? 1.0 : 0.0,
        latched ? 1.0 : 0.0, geometry.equality_position_max,
        geometry.equality_velocity_max, geometry.length_velocity_max,
        geometry.minimum_jacobian_ratio};
      for (int i = 0; i < 8; ++i)
        diagnostics[i] = i == 7 ? diagnostic_min(diagnostics[i], step_diagnostics[i])
                                : diagnostic_max(diagnostics[i], step_diagnostics[i]);
    }
    Vec4 current{};
    for (int i = 0; i < 4; ++i) current[i] = sent_sum[i] / substeps / torque_scales_[i];
    const double effort = effort_sum / (substeps * 4.0);
    const double action_rate = rate_sum / (substeps * 4.0);
    constexpr std::array<const char*, 8> diagnostic_names{
      "saturation_fraction", "deadpoint_fraction", "out_of_range", "invalid_state",
      "equality_position_max_rad", "equality_velocity_max_rad_s",
      "length_velocity_residual_max_m_s", "min_jacobian_ratio"};
    for (int i = 0; i < 8; ++i)
      metrics[std::string("hybrid/") + diagnostic_names[i]] = diagnostics[i];
    metrics["hybrid/actual_effort"] = effort;
    const double reward = get_reward(data, s, effort, action_rate, metrics) *
                          (substeps * model->opt.timestep / reference_dt_);
    s.last_last_action = s.last_action;
    s.last_action = current;
    s.last_sent = current;
    if (std::abs(s.command[0]) > linear_margin_) s.position_start = hold_position(data);
    if (std::abs(s.command[1]) > angular_margin_) s.sinyaw_start = sinyaw(data);
    s.sin_balance_zero = balance(data);

    const auto after = geometry_state(model, data);
    if (reason.empty()) reason = after.reason;
    const double pitch = sinpitch(data), roll = sinroll(data);
    if (reason.empty() && (pitch < pitch_min_ || pitch > pitch_max_)) reason = "pitch";
    if (reason.empty() && std::abs(roll) > roll_limit_) reason = "roll";
    if (!std::isfinite(reward)) throw std::runtime_error("legwheel: nonfinite reward");
    outcome.reward = reward;
    outcome.terminated = latched || !reason.empty();
    outcome.reason = std::move(reason);
    return outcome;
  }

 private:
  struct GeometryState {
    std::array<double, 2> jac{};
    double deadpoint_fraction = 0;
    bool outside = false;
    double equality_position_max = 0;
    double equality_velocity_max = 0;
    double length_velocity_max = 0;
    double minimum_jacobian_ratio = std::numeric_limits<double>::infinity();
    std::string reason;
  };

  static double anchor(const InitialState& initial, const std::string& key, double fallback) {
    auto found = initial.extra.find(key);
    if (found == initial.extra.end()) return fallback;
    if (found->second.size() != 1 || !std::isfinite(found->second[0]))
      invalid("initial extra " + key + " must contain one finite value");
    return found->second[0];
  }

  void validate_geometry(const mjModel* model, int side, int hip, int knee, int wheel) const {
    const auto& leg = legs_[side];
    const int thigh_body = model->jnt_bodyid[hip], calf_body = model->jnt_bodyid[knee];
    if (model->body_parentid[calf_body] != thigh_body ||
        model->jnt_bodyid[wheel] != wheel_body_[side] ||
        model->body_parentid[wheel_body_[side]] != calf_body)
      invalid("geometry requires hip/calf/wheel bodies in a direct chain");
    auto close = [](double actual, double expected) { return std::abs(actual - expected) <= 1e-9; };
    if (!close(model->qpos0[model->jnt_qposadr[hip]], leg.hip_ref) ||
        !close(model->qpos0[model->jnt_qposadr[knee]], leg.knee_ref))
      invalid("geometry references disagree with model");
    mjtNum knee_matrix[9];
    mju_quat2Mat(knee_matrix, model->body_quat + 4 * calf_body);
    for (int k = 0; k < 3; ++k) {
      if (!close(model->jnt_pos[3 * hip + k], 0) || !close(model->jnt_pos[3 * wheel + k], 0) ||
          !close(model->body_pos[3 * calf_body + k], leg.knee_offset[k]) ||
          !close(model->jnt_pos[3 * knee + k], leg.knee_anchor[k]) ||
          !close(model->jnt_axis[3 * knee + k], leg.knee_axis[k]) ||
          !close(model->body_pos[3 * wheel_body_[side] + k], leg.wheel_center[k]))
        invalid("geometry coordinates disagree with model");
    }
    for (int k = 0; k < 9; ++k)
      if (!close(knee_matrix[k], leg.knee_rotation[k])) invalid("geometry rotation disagrees with model");
    int matching_equalities = 0;
    for (int eq = 0; eq < model->neq; ++eq) {
      if (model->eq_type[eq] != mjEQ_JOINT || model->eq_obj1id[eq] != knee || model->eq_obj2id[eq] != hip)
        continue;
      ++matching_equalities;
      if (!model->eq_active0[eq]) invalid("knee coupling equality is inactive");
      for (int k = 0; k < 5; ++k)
        if (!close(model->eq_data[mjNEQDATA * eq + k], leg.poly[k]))
          invalid("geometry polynomial disagrees with model equality");
    }
    if (matching_equalities != 1) invalid("expected one active polynomial knee/hip equality per leg");
  }

  GeometryState geometry_state(const mjModel* model, const mjData* data) const {
    GeometryState out;
    for (int i = 0; i < model->nq; ++i)
      if (!std::isfinite(data->qpos[i])) { out.reason = "nonfinite_state"; break; }
    for (int i = 0; i < model->nv; ++i)
      if (!std::isfinite(data->qvel[i])) { out.reason = "nonfinite_state"; break; }
    for (int side = 0; side < 2; ++side) {
      const auto& leg = legs_[side];
      const double qh = data->qpos[hip_q_[side]], qk = data->qpos[knee_q_[side]];
      const auto result = leg.evaluate(qh, qk);
      const double rpos = qk - leg.coupled_knee(qh);
      const double rvel = data->qvel[knee_v_[side]] - leg.slope(qh) * data->qvel[hip_v_[side]];
      const double rlength = result[2] * rvel;
      out.jac[side] = result[1];
      const double jacobian_ratio = std::abs(result[1]) / jmax_[side];
      out.deadpoint_fraction += jacobian_ratio < minimum_ratio_ ? .5 : 0;
      out.outside = out.outside || qh < hip_range_[0] || qh > hip_range_[1];
      out.equality_position_max = diagnostic_max(out.equality_position_max, std::abs(rpos));
      out.equality_velocity_max = diagnostic_max(out.equality_velocity_max, std::abs(rvel));
      out.length_velocity_max = diagnostic_max(out.length_velocity_max, std::abs(rlength));
      out.minimum_jacobian_ratio = diagnostic_min(out.minimum_jacobian_ratio, jacobian_ratio);
      if (!out.reason.empty()) continue;
      if (!std::isfinite(result[0]) || !std::isfinite(result[1]) || !std::isfinite(rlength))
        out.reason = "nonfinite_geometry";
      else if (qh < hip_range_[0] || qh > hip_range_[1]) out.reason = "hip_range";
      else if (std::abs(rpos) > equality_limit_) out.reason = "equality_position";
      else if (std::abs(rlength) > length_velocity_limit_) out.reason = "equality_velocity";
      else if (std::abs(result[1]) / jmax_[side] < minimum_ratio_) out.reason = "deadpoint";
    }
    return out;
  }

  double hold_position(const mjData* data) const {
    return .5 * (data->qpos[wheel_q_[0]] + data->qpos[wheel_q_[1]]);
  }
  double sinpitch(const mjData* data) const { return -data->xmat[9 * root_body_ + 6]; }
  double sinroll(const mjData* data) const {
    const double pitch = sinpitch(data);
    return data->xmat[9 * root_body_ + 7] / std::sqrt(1 - pitch * pitch);
  }
  double sinyaw(const mjData* data) const {
    const auto* r = data->xmat + 9 * root_body_;
    return std::sin(std::atan2(r[3], r[0]) * sinyaw_scale_) / sinyaw_scale_;
  }
  double balance(const mjData* data) const {
    if (!balance_zero_) return 0;
    Vec3 direction{}, forward{};
    const auto* r = data->xmat + 9 * root_body_;
    for (int k = 0; k < 3; ++k) {
      direction[k] = .5 * (data->xpos[3 * wheel_body_[0] + k] + data->xpos[3 * wheel_body_[1] + k]) -
                     data->subtree_com[3 * root_body_ + k];
      forward[k] = r[3 * k];
    }
    const double cosine = dot(direction, forward) /
                          (std::sqrt(dot(direction, direction)) * std::sqrt(dot(forward, forward)) + 1e-8);
    constexpr double half_pi = 1.57079632679489661923;
    return std::sin(half_pi - std::acos(std::clamp(cosine, -1.0, 1.0)));
  }

  double get_reward(const mjData* data, const LegwheelState& s, double effort, double rate,
                    std::map<std::string, double>& metrics) const {
    const double tracking_linear = std::exp(-std::abs(data->sensordata[local_velocity_] - s.command[0]) /
                                            (sigma_ * command_scales_[0]));
    const double tracking_angular = std::exp(-std::abs(data->sensordata[gyro_ + 2] - s.command[1]) /
                                             (sigma_ * command_scales_[1]));
    const double tracking_position = std::abs(s.command[0]) <= linear_margin_
      ? std::exp(-std::abs(hold_position(data) - s.position_start) / sigma_wheel_) : 0;
    const double tracking_yaw = std::abs(s.command[1]) <= angular_margin_
      ? std::exp(-std::abs(sinyaw(data) - s.sinyaw_start) / (sigma_ * command_scales_[3])) : 0;
    const double pitch = std::exp(-std::abs(sinpitch(data) - s.sin_balance_zero) / (sigma_ * command_scales_[3]));
    const double roll = std::exp(-std::abs(sinroll(data) - s.command[3]) / (sigma_ * command_scales_[3]));
    double height_ref = s.command[2];
    if (height_projection_) {
      const double delta = roll_coupling_ * std::abs(s.command[3]);
      // NumPy/JAX clip returns the upper bound when lower > upper.
      height_ref = std::min(std::max(height_ref, thigh_range_[0] + delta), thigh_range_[1] - delta);
    }
    const double height_error = .5 * ((data->qpos[hip_q_[0]] - height_ref) + (data->qpos[hip_q_[1]] - height_ref));
    const double height = std::exp(-std::abs(height_error) / (sigma_ * command_scales_[2]));
    metrics["reward/tracking_lin_vel"] = tracking_linear;
    metrics["reward/tracking_wheels_sinposition"] = tracking_position;
    metrics["reward/tracking_sinyaw"] = tracking_yaw;
    metrics["reward/tracking_ang_vel"] = tracking_angular;
    metrics["reward/pitch_cost"] = pitch;
    metrics["reward/roll_cost"] = roll;
    metrics["reward/tracking_height_jnt"] = height;
    metrics["reward/energy_cost"] = effort;
    metrics["reward/action_rate"] = rate;
    metrics["reward/pitch_velocity_penalty"] = std::abs(data->sensordata[gyro_ + 1]);
    metrics["reward/termination_penalty"] = 0;
    const double linear_error = (data->sensordata[local_velocity_] - s.command[0]) / .2;
    const double angular_error = (data->sensordata[gyro_ + 2] - s.command[1]) / 1.5;
    metrics["track/err2_lin_vel"] = linear_error * linear_error;
    metrics["track/err2_ang_vel"] = angular_error * angular_error;
    metrics["track/active_steps_sinpos"] = std::abs(s.command[0]) <= linear_margin_ ? 1.0 : 0.0;
    metrics["track/active_steps_sinyaw"] = std::abs(s.command[1]) <= angular_margin_ ? 1.0 : 0.0;
    const double velocity_reward = ((enable_linear_ ? tracking_linear : 0) +
                                    (enable_angular_ ? tracking_angular : 0)) * vel_scale_;
    const double posture = pitch * pitch_scale_ + (enable_roll_ ? roll * roll_scale_ : 0);
    const double zero_keep = ((position_hold_ ? tracking_position : 0) +
                              (yaw_hold_ ? tracking_yaw : 0)) * zero_keep_scale_;
    const double height_reward = (enable_height_ ? height : 0) * height_scale_;
    return velocity_reward + posture + zero_keep + height_reward + effort * energy_scale_ +
           rate * action_rate_scale_ + std::abs(data->sensordata[gyro_ + 1]) * pitch_velocity_scale_;
  }

  std::array<LegGeometry, 2> legs_;
  std::array<double, 2> jmax_, force_scales_;
  Vec4 torque_scales_;
  std::array<double, 2> hip_range_;
  Vec4 command_scales_;
  std::array<double, 2> thigh_range_;
  std::array<int, 2> hip_q_{}, knee_q_{}, wheel_q_{}, hip_v_{}, knee_v_{}, wheel_v_{}, wheel_body_{};
  std::array<int, 4> actuator_{}, actual_v_{};
  int root_body_ = -1, gyro_ = -1, local_velocity_ = -1;
  std::string mode_;
  double reference_dt_, wheel_radius_, sinyaw_scale_, linear_margin_, angular_margin_, sigma_, sigma_wheel_;
  double height_scale_, vel_scale_, roll_scale_, pitch_scale_, zero_keep_scale_, energy_scale_, action_rate_scale_;
  double pitch_velocity_scale_, roll_coupling_, equality_limit_, length_velocity_limit_, minimum_ratio_;
  double pitch_min_, pitch_max_, roll_limit_;
  bool balance_zero_, yaw_hold_, position_hold_, enable_linear_, enable_angular_, enable_height_, enable_roll_;
  bool height_projection_;
};

}  // namespace

std::unique_ptr<Task> make_legwheel_task(const mjModel* model, const TaskParameters& params) {
  if (!model) invalid("model must not be null");
  return std::make_unique<LegwheelTask>(model, params);
}

}  // namespace native_eval
