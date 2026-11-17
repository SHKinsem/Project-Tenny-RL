"""Patch an isolated native-task copy to separate limits from normalization.

The production evaluator is never written here.  The caller supplies the exact
reviewed source, records the returned source, and builds it in its own directory.
An omitted normalization vector follows the original torque scales, preserving
the evaluator's default behavior.  Actual equivalence needs runtime validation.
"""
import hashlib


FROZEN_SOURCE_SHA256 = '03c864594f315847ca48b466c933fc6e3f803ef879a2a4d238454970eb1ee906'
_MARKER = '// torque-limit-ablation: fixed torque normalization'


def patch_legwheel_source(source: str) -> str:
    """Return the reviewed normalization-only patch, rejecting source drift.

    The digest covers UTF-8 source after CRLF-to-LF normalization.  Execution
    limits, actuator-range checks, clipping, geometry, guards and timing stay on
    ``torque_scales_``.  Only the observation/effort/rate denominator is split.
    """
    if not isinstance(source, str):
        raise TypeError('Native legwheel source must be text')
    source = source.replace('\r\n', '\n')
    if _MARKER in source or 'torque_normalization_scales' in source or 'norm_scales_' in source:
        raise ValueError('Native legwheel normalization patch was already applied')
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if digest != FROZEN_SOURCE_SHA256:
        raise ValueError('Native legwheel source differs from the frozen reviewed source: '+digest)

    edits = (
        (
            '        torque_scales_(vector_value<4>(p, "torque_scales")),\n',
            '        torque_scales_(vector_value<4>(p, "torque_scales")),\n'
            '        norm_scales_(p.vectors.count("torque_normalization_scales")\n'
            '                       ? vector_value<4>(p, "torque_normalization_scales")\n'
            '                       : torque_scales_),\n',
        ),
        (
            '                               "torque_scales", "hip_range", "command_scales", "thigh_range"});\n',
            '                               "torque_scales", "torque_normalization_scales",\n'
            '                               "hip_range", "command_scales", "thigh_range"});\n',
        ),
        (
            '    for (double value : torque_scales_) if (!(value > 0)) invalid("torque_scales must be positive");\n',
            '    for (double value : torque_scales_) if (!(value > 0)) invalid("torque_scales must be positive");\n'
            '    '+_MARKER+'\n'
            '    for (double value : norm_scales_)\n'
            '      if (!(std::isfinite(value) && value > 0))\n'
            '        invalid("torque_normalization_scales must be positive and finite");\n',
        ),
        (
            '        const double actual = data->qfrc_actuator[actual_v_[i]] / torque_scales_[i];\n',
            '        const double actual = data->qfrc_actuator[actual_v_[i]] / norm_scales_[i];\n',
        ),
        (
            '    for (int i = 0; i < 4; ++i) current[i] = sent_sum[i] / substeps / torque_scales_[i];\n',
            '    for (int i = 0; i < 4; ++i) current[i] = sent_sum[i] / substeps / norm_scales_[i];\n',
        ),
        (
            '  Vec4 torque_scales_;\n',
            '  Vec4 torque_scales_, norm_scales_;\n',
        ),
    )
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError('Frozen native patch anchor must occur exactly once: '+before.strip())
        source = source.replace(before, after, 1)
    return source
