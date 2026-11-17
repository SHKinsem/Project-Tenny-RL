"""Fail-closed patches for an isolated, physically aligned capability experiment.

No production source is written.  The caller passes the four original source
texts, archives the returned copies, and performs native build/runtime checks.
"""
import hashlib

from experiments.torque_limit_ablation.native_patch import patch_legwheel_source


LEG_ANCHOR_RATIO = 0.5 * (1.0 - 2.0**-20)
FROZEN_SOURCE_SHA256 = {
    'cpp/legwheel_task.cpp': '03c864594f315847ca48b466c933fc6e3f803ef879a2a4d238454970eb1ee906',
    'cpp/policy.cpp': '3e301263667034f3e743e80f83edc2441ea9a8aa4837486f4f10409103d351d2',
    'cpp/bindings.cpp': '1d7eafb46059c1d01a4b81f878a00930192af58f1c2ec73f0a65849c12c13bff',
    'include/native_eval/runtime.hpp': '31e594bce75e81d11b47d2a67253f5c51b03e74a9c9327369f5d2b12f81244e0',
}
_NORMALIZED_TASK_SHA256 = '7e6105cb2c977616ecf5077a571e0f7046d9c6c7519a9a4242cea8123c3c353f'
_MARKER = '// aligned-limit-ablation:'


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Aligned native patch anchor must occur exactly once: '+old.strip())
    return source.replace(old, new, 1)


def _patch_task(source):
    source = patch_legwheel_source(source)
    if hashlib.sha256(source.encode()).hexdigest() != _NORMALIZED_TASK_SHA256:
        raise ValueError('The inherited normalization patch differs from its reviewed output')
    edits = (
        (
            '        force_scales_(vector_value<2>(p, "force_scales")),\n',
            '        force_scales_(vector_value<2>(p, "force_scales")),\n'
            '        policy_leg_force_scales_(p.vectors.count("policy_leg_force_scales")\n'
            '            ? vector_value<2>(p, "policy_leg_force_scales") : force_scales_),\n'
            '        has_policy_leg_force_scales_(p.vectors.count("policy_leg_force_scales") != 0),\n',
        ),
        (
            '    reject_unknown(p.vectors, {"leg_geometry_left", "leg_geometry_right", "jmax", "force_scales",\n',
            '    reject_unknown(p.vectors, {"leg_geometry_left", "leg_geometry_right", "jmax", "force_scales",\n'
            '                               "policy_leg_force_scales",\n',
        ),
        (
            '      invalid("unsupported action_mode: " + mode_);\n',
            '      invalid("unsupported action_mode: " + mode_);\n'
            '    '+_MARKER+' policy force units and execution capacity are separate.\n'
            '    if (has_policy_leg_force_scales_ && mode_ != "virtual_leg_force")\n'
            '      invalid("policy_leg_force_scales requires virtual_leg_force mode");\n'
            '    for (double value : policy_leg_force_scales_)\n'
            '      if (!(std::isfinite(value) && value > 0))\n'
            '        invalid("policy_leg_force_scales must be positive and finite");\n',
        ),
        (
            '        const double raw = latched ? 0.0 : gain * clipped[i];\n',
            '        double raw = latched ? 0.0 : gain * clipped[i];\n'
            '        if (has_policy_leg_force_scales_ && i < 2) {\n'
            '          const double force = std::clamp(clipped[i] * policy_leg_force_scales_[i],\n'
            '                                          -force_scales_[i], force_scales_[i]);\n'
            '          const double j = std::isfinite(geometry.jac[i]) ? geometry.jac[i] : 0;\n'
            '          raw = latched ? 0.0 : j * force;\n'
            '        }\n',
        ),
        (
            '  std::array<double, 2> jmax_, force_scales_;\n',
            '  std::array<double, 2> jmax_, force_scales_, policy_leg_force_scales_;\n'
            '  bool has_policy_leg_force_scales_ = false;\n',
        ),
    )
    for old, new in edits:
        source = _replace_once(source, old, new)
    return source


def _patch_header(source):
    return _replace_once(
        source,
        '  float skip_clip_epsilon = 1e-5f;\n',
        '  float skip_clip_epsilon = 1e-5f;\n'
        '  '+_MARKER+' frozen raw-mean anchor, absent for legacy policies.\n'
        '  std::shared_ptr<MlpSpec> leg_anchor;\n'
        '  float leg_anchor_ratio = 0.4999995231628418f;\n',
    )


def _patch_bindings(source):
    edits = (
        (
            'MlpSpec parse_policy(const py::dict& mapping) {\n',
            'MlpSpec parse_policy(const py::dict& mapping, bool allow_leg_anchor = true) {\n'
            '  '+_MARKER+' reject nesting before recursively parsing an anchor.\n'
            '  if (!allow_leg_anchor && mapping.contains("leg_anchor"))\n'
            '    throw std::invalid_argument("leg_anchor must not contain another leg_anchor");\n',
        ),
        (
            '                       "skip_clip_epsilon"}, "policy");\n',
            '                       "skip_clip_epsilon", "leg_anchor", "leg_anchor_ratio"}, "policy");\n',
        ),
        (
            '  if (mapping.contains("skip_clip_epsilon")) result.skip_clip_epsilon = as_float32(mapping["skip_clip_epsilon"], "skip_clip_epsilon");\n',
            '  if (mapping.contains("skip_clip_epsilon")) result.skip_clip_epsilon = as_float32(mapping["skip_clip_epsilon"], "skip_clip_epsilon");\n'
            '  if (mapping.contains("leg_anchor")) {\n'
            '    result.leg_anchor = std::make_shared<MlpSpec>(\n'
            '        parse_policy(as_dict(mapping["leg_anchor"], "leg_anchor"), false));\n'
            '  }\n'
            '  if (mapping.contains("leg_anchor_ratio")) {\n'
            '    if (!result.leg_anchor)\n'
            '      throw std::invalid_argument("leg_anchor_ratio requires leg_anchor");\n'
            '    result.leg_anchor_ratio = as_float32(mapping["leg_anchor_ratio"], "leg_anchor_ratio");\n'
            '  }\n',
        ),
    )
    for old, new in edits:
        source = _replace_once(source, old, new)
    return source


def _patch_policy(source):
    edits = (
        (
            '    sized_default(spec_.skip_scales, spec_.action_size, 1.0f, "skip_scales");\n',
            '    sized_default(spec_.skip_scales, spec_.action_size, 1.0f, "skip_scales");\n'
            '    '+_MARKER+' same observation units and an immutable raw leg-mean head.\n'
            '    if (spec_.leg_anchor) {\n'
            '      if (spec_.mean.size() != 24 || spec_.action_size != 4 ||\n'
            '          spec_.output_activation != "tanh" ||\n'
            '          spec_.layers.back().activation != "identity")\n'
            '        throw std::invalid_argument("leg_anchor requires a 24-observation, 4-action raw mean head with tanh output");\n'
            '      if (!(std::isfinite(spec_.leg_anchor_ratio) && spec_.leg_anchor_ratio > 0.0f &&\n'
            '            spec_.leg_anchor_ratio <= 0.5f))\n'
            '        throw std::invalid_argument("leg_anchor_ratio must be finite and in (0, 0.5]");\n'
            '      const auto& anchor_spec = *spec_.leg_anchor;\n'
            '      if (anchor_spec.leg_anchor)\n'
            '        throw std::invalid_argument("leg_anchor must not contain another leg_anchor");\n'
            '      if (anchor_spec.mean != spec_.mean || anchor_spec.stddev != spec_.stddev ||\n'
            '          anchor_spec.action_size != 4 || anchor_spec.output_activation != "identity" ||\n'
            '          anchor_spec.layers.empty() || anchor_spec.layers.back().activation != "identity")\n'
            '        throw std::invalid_argument("leg_anchor must share mean/std and expose four raw identity outputs");\n'
            '      anchor_ = std::make_unique<MlpPolicy>(anchor_spec);\n'
            '      for (std::size_t leg = 0; leg < 2; ++leg) {\n'
            '        if (spec_.skip_indices[leg] != -1 || spec_.network_scales[leg] != 1.0f ||\n'
            '            anchor_->spec_.skip_indices[leg] != -1 ||\n'
            '            anchor_->spec_.network_scales[leg] != 1.0f)\n'
            '          throw std::invalid_argument("aligned leg means require no input skip and unit network scale");\n'
            '      }\n'
            '    }\n',
        ),
        (
            '    action.resize(action_size());\n',
            '    std::vector<double> anchor_mean;\n'
            '    if (anchor_) anchor_->infer(observation, anchor_mean);\n'
            '    action.resize(action_size());\n',
        ),
        (
            '      float value = spec_.network_scales[output] * current[output];\n',
            '      float value = spec_.network_scales[output] * current[output];\n'
            '      if (anchor_ && output < 2) {\n'
            '        const float anchor_mu = static_cast<float>(anchor_mean[output]);\n'
            '        value = std::atanh(spec_.leg_anchor_ratio * std::tanh(anchor_mu)) +\n'
            '                (value - anchor_mu);\n'
            '      }\n',
        ),
        (
            '  MlpSpec spec_;\n',
            '  MlpSpec spec_;\n'
            '  std::unique_ptr<MlpPolicy> anchor_;\n',
        ),
    )
    for old, new in edits:
        source = _replace_once(source, old, new)
    return source


def patch_sources(sources: dict[str, str]) -> dict[str, str]:
    """Patch four reviewed originals; preserve other mapping entries unchanged.

    Required keys are the repository-relative native source paths in
    ``FROZEN_SOURCE_SHA256``.  CRLF input is accepted and changed sources use LF.
    Already patched or otherwise unknown required source text is rejected.
    """
    if not isinstance(sources, dict):
        raise TypeError('Native sources must be a path-to-text dictionary')
    normalized = {}
    for path, expected in FROZEN_SOURCE_SHA256.items():
        if path not in sources:
            raise ValueError('Missing frozen native source: '+path)
        if not isinstance(sources[path], str):
            raise TypeError('Native source must be text: '+path)
        text = sources[path].replace('\r\n', '\n')
        if _MARKER in text or 'policy_leg_force_scales' in text or 'leg_anchor_ratio' in text:
            raise ValueError('Aligned native patch was already applied: '+path)
        actual = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if actual != expected:
            raise ValueError('Native source differs from its frozen reviewed original: '+path+' '+actual)
        normalized[path] = text
    result = dict(sources)
    result['cpp/legwheel_task.cpp'] = _patch_task(normalized['cpp/legwheel_task.cpp'])
    result['include/native_eval/runtime.hpp'] = _patch_header(normalized['include/native_eval/runtime.hpp'])
    result['cpp/bindings.cpp'] = _patch_bindings(normalized['cpp/bindings.cpp'])
    result['cpp/policy.cpp'] = _patch_policy(normalized['cpp/policy.cpp'])
    return result
