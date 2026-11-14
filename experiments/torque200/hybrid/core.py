"""Backend-neutral reduced leg kinematics and action contract.

No simulator imports. ``trig`` can be math or jax.numpy; runtime calculations
remain JIT/vmap-compatible. Geometry is read from the source model, never copied
from a remembered coefficient list. This is NOT a contact-force controller.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/legwheel_robot2_stl_nolinkage6_terrain.xml"
MODES = ("virtual_leg_force", "matched_envelope", "direct_torque")
SCHEMA = "wheel-center-axial-v1"
OBS_VERSION = "torque24-mean-sent-v1"
TORQUE_LIMITS = (1.0, 1.0, 0.0295, 0.0295)
# Software experiment envelope, NOT mechanical stops. Contains qpos0=0 with a
# small margin. The same limits and termination apply to all three NEW modes.
HIP_RANGE = (-0.5, 0.05)
# Provisional numerical validity thresholds, NOT hardware safety ratings.
MAX_EQ_POSITION = 0.01  # rad
MAX_LENGTH_VELOCITY = 0.05  # m/s
MIN_JACOBIAN_RATIO = 0.05
ACTION_ORDER = ("left_leg", "right_leg", "left_wheel", "right_wheel")


def vec(text, default="0 0 0"):
    return tuple(float(x) for x in (text if text is not None else default).split())


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def matvec(m, v):
    return tuple(dot(row, v) for row in m)


def matmul(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def rotation(axis, angle, trig=math):
    """Rodrigues rotation about a unit axis (active rotation)."""
    c, s = trig.cos(angle), trig.sin(angle)
    x, y, z = axis
    return ((c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s),
            (y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s),
            (z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)))


def body_rotation(body):
    # The supported XML explicitly uses fixed/extrinsic XYZ, not moving xyz.
    if any(k in body.attrib for k in ("quat", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("Unsupported body orientation; extend and recalibrate explicitly")
    x, y, z = vec(body.get("euler"))
    return matmul(rotation((0, 0, 1), z), matmul(rotation((0, 1, 0), y), rotation((1, 0, 0), x)))


@dataclass(frozen=True)
class LegGeometry:
    side: str
    hip_ref: float
    knee_ref: float
    poly: tuple
    knee_offset: tuple
    knee_rotation: tuple
    knee_anchor: tuple
    knee_axis: tuple
    wheel_center: tuple

    def coupled_knee(self, qh):
        x = qh - self.hip_ref
        a0, a1, a2, a3, a4 = self.poly
        return self.knee_ref + a0 + x*(a1+x*(a2+x*(a3+x*a4)))

    def slope(self, qh):
        x = qh - self.hip_ref
        _, a1, a2, a3, a4 = self.poly
        return a1 + x*(2*a2+x*(3*a3+x*4*a4))

    def evaluate(self, qh, qk=None, trig=math):
        """Return length, constrained J_l, partial dl/dqk at actual qk.

        H coincides with the hip joint anchor. Its relative distance to C is
        invariant under the hip's rigid rotation: partial dl/dqh = 0. The
        independent-coordinate derivative is therefore g'(qh) * dl/dqk.
        Do NOT overwrite qk to project a soft equality during simulation.
        """
        if qk is None:
            qk = self.coupled_knee(qh)
        v = tuple(c-a for c, a in zip(self.wheel_center, self.knee_anchor))
        rotated = matvec(rotation(self.knee_axis, qk-self.knee_ref, trig), v)
        inner = tuple(a+b for a, b in zip(self.knee_anchor, rotated))
        p = tuple(a+b for a, b in zip(self.knee_offset, matvec(self.knee_rotation, inner)))
        dp = matvec(self.knee_rotation, cross(self.knee_axis, rotated))
        length = trig.sqrt(dot(p, p))
        # This model never reaches zero length in the calibrated envelope.
        # Keep the denominator finite even if a corrupted state is diagnosed.
        dl_dk = dot(p, dp) / (length + 1e-15)
        return length, self.slope(qh)*dl_dk, dl_dk


def read_geometry(xml=XML):
    root = ET.parse(xml).getroot()
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle") != "radian" or compiler.get("eulerseq") != "XYZ":
        raise ValueError("Only the checked radian / extrinsic XYZ model is supported")
    # Visual includes are supported; hidden kinematic/default definitions are not.
    for inc in root.findall(".//include"):
        included = ET.parse(Path(xml).parent/inc.attrib["file"]).getroot()
        if any(x.tag not in ("asset", "visual") for x in included):
            raise ValueError("Only visual/asset includes are supported by the reduced FK")
    for node in root.findall(".//default//joint"):
        if any(k in node.attrib for k in ("pos", "axis", "ref", "type")):
            raise ValueError("Inherited joint geometry requires explicit parser support")
    for node in root.findall(".//default//site"):
        if "pos" in node.attrib:
            raise ValueError("Inherited site position requires explicit parser support")
    joints = {x.get("name"): x for x in root.findall(".//worldbody//joint")}
    legs = []
    for side in ("left", "right"):
        thigh = root.find(f".//body[@name='thigh_{side}']")
        calf = thigh.find(f"body[@name='calf_{side}']")
        wheel = calf.find(f"body[@name='wheel_{side}']")
        if any(len(b.findall("joint")) != 1 or b.find("freejoint") is not None for b in (thigh, calf, wheel)):
            raise ValueError("Expected one hinge per leg body")
        hj, kj, wj = (joints[f"{part}_{side}_joint"] for part in ("thigh", "calf", "wheel"))
        hs = thigh.find(f"site[@name='thigh_{side}_site']")
        ws = wheel.find(f"site[@name='wheel_{side}_site']")
        # Fail closed if a later CAD/model edit invalidates the simplified FK.
        if any(vec(node.get("pos")) != (0.0, 0.0, 0.0) for node in (hj, hs, wj, ws)):
            raise ValueError("Hip / wheel sites must coincide with their joint origins")
        if any(j.get("type", "hinge") != "hinge" for j in (hj, kj, wj)):
            raise ValueError("Expected hinge joints")
        eqs = [e for e in root.findall("equality/joint")
               if e.get("joint1") == kj.get("name") and e.get("joint2") == hj.get("name")]
        if len(eqs) != 1 or eqs[0].get("active", "true") != "true":
            raise ValueError("Expected one active knee = polynomial(hip) equality")
        poly = vec(eqs[0].get("polycoef"), "0 1 0 0 0")
        if len(poly) != 5:
            raise ValueError("Expected five equality coefficients")
        axis = vec(kj.get("axis"), "0 0 1")
        norm = math.sqrt(dot(axis, axis))
        if norm == 0:
            raise ValueError("Invalid knee axis")
        legs.append(LegGeometry(side, float(hj.get("ref", 0)), float(kj.get("ref", 0)),
                                poly, vec(calf.get("pos")), body_rotation(calf),
                                vec(kj.get("pos")), tuple(x/norm for x in axis), vec(wheel.get("pos"))))
    return tuple(legs)


def map_action(action, jacobian, force_scales, mode, limits=TORQUE_LIMITS):
    """Scalar Python reference; runtime uses the identical array expression.

    No division by Jacobian, no mg/2 bias and no wheel reaction added to hip.
    """
    if (mode not in MODES or len(action) != 4 or len(jacobian) != 2
            or len(force_scales) != 2 or len(limits) != 4):
        raise ValueError("Invalid mode or dimensions")
    if any(not math.isfinite(float(v)) for v in (*action, *jacobian, *force_scales, *limits)):
        raise ValueError("Nonfinite control input")
    if any(v <= 0 for v in (*force_scales, *limits)):
        raise ValueError("Scales and limits must be positive")
    a = [max(-1.0, min(1.0, float(v))) for v in action]
    gains = list(limits[:2]) if mode == "direct_torque" else [
        (abs(j) if mode == "matched_envelope" else j)*f for j, f in zip(jacobian, force_scales)]
    tau = [g*a[i] for i, g in enumerate((*gains, *limits[2:]))]
    return tuple(max(-lim, min(lim, t)) for t, lim in zip(tau, limits))


def scan_geometry(legs, hip_range=HIP_RANGE, samples=4097):
    """Dense + locally refined numerical audit, NOT a certified analytic bound.

    Enlarge the sampled J maximum by 1% and leave a 10% torque margin. Runtime
    still clips torque and terminates outside the common experimental envelope.
    """
    lo, hi = hip_range
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi) or samples < 3:
        raise ValueError("Invalid scan range")
    result = []
    for leg in legs:
        qs = [lo+(hi-lo)*i/(samples-1) for i in range(samples)]
        rows = [leg.evaluate(q) for q in qs]
        imax = max(range(samples), key=lambda i: abs(rows[i][1]))
        a, b = qs[max(0, imax-1)], qs[min(samples-1, imax+1)]
        refined = [leg.evaluate(a+(b-a)*i/1000)[1] for i in range(1001)]
        js = [r[1] for r in rows] + refined
        if min(js) <= 0 <= max(js):
            raise ValueError("Jacobian changes sign or reaches zero in the proposed envelope")
        jmax = max(abs(j) for j in js)*1.01
        min_ratio = min(abs(j) for j in js)/jmax
        if min_ratio < 0.05:
            raise ValueError("Proposed envelope includes the diagnostic dead-point band")
        errors = []
        for q in qs[::64]:
            for eps in (1e-6, 1e-5, 1e-4):
                fd = (leg.evaluate(q+eps)[0]-leg.evaluate(q-eps)[0])/(2*eps)
                errors.append(abs(fd-leg.evaluate(q)[1]))
        result.append(dict(side=leg.side, length_min_m=min(r[0] for r in rows),
                           length_max_m=max(r[0] for r in rows), jacobian_min=min(js),
                           jacobian_max=max(js), jmax_bound=jmax, minimum_ratio=min_ratio,
                           force_scale_n=0.9*TORQUE_LIMITS[len(result)]/jmax,
                           finite_difference_max_error=max(errors), samples=samples,
                           geometry=asdict(leg)))
    return result


def source_fingerprint():
    """Bind calibration/checkpoints to exact control code and model assets."""
    paths = sorted([*ROOT.joinpath("assets").rglob("*"),
                    *ROOT.joinpath("hybrid").glob("*.py"),
                    *ROOT.joinpath("legwheel").rglob("*.py"),
                    ROOT/"verify_artifacts.py", ROOT/"hybrid-wsl.sh"])
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode()+b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def metadata(mode, scan):
    if mode not in MODES:
        raise ValueError(mode)
    return dict(schema=SCHEMA, action_mode=mode, action_order=list(ACTION_ORDER),
                observation_version=OBS_VERSION, actor_dim=24, critic_dim=51,
                ctrl_dt=0.005, sim_dt=0.0025, force_bias_n=[0.0, 0.0],
                force_scales_n=[s["force_scale_n"] for s in scan],
                torque_limits_nm=list(TORQUE_LIMITS), hip_range=list(HIP_RANGE),
                validity=dict(equality_position_rad=MAX_EQ_POSITION,
                              length_velocity_m_s=MAX_LENGTH_VELOCITY,
                              minimum_jacobian_ratio=MIN_JACOBIAN_RATIO),
                source_fingerprint=source_fingerprint())


def require_metadata(actual, expected):
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f"Incompatible checkpoint/calibration metadata: {key}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")
    tmp.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_tree(path):
    """Content/name hash of a completed checkpoint; reject links/empty trees."""
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Expected a real checkpoint directory")
    files = sorted(root.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Checkpoint links are not supported")
    files = [p for p in files if p.is_file()]
    if not files:
        raise ValueError("Empty checkpoint")
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(root).as_posix().encode()+b"\0")
        digest.update(bytes.fromhex(digest_file(p)))
    return digest.hexdigest()


def validate_positions(legs, hip_positions, knee_positions, jmax):
    """Host controller state validity. Velocities need a separate caller check."""
    if len(hip_positions) != 2 or len(knee_positions) != 2 or len(jmax) != 2:
        raise ValueError("Expected left/right hip and knee angles")
    if any(not math.isfinite(float(q)) for q in (*hip_positions, *knee_positions)):
        raise ValueError("Nonfinite joint position")
    if any(q < HIP_RANGE[0] or q > HIP_RANGE[1] for q in hip_positions):
        raise ValueError("Outside calibrated experimental range; do not extrapolate")
    if any(abs(knee_positions[i]-leg.coupled_knee(hip_positions[i])) > MAX_EQ_POSITION
           for i, leg in enumerate(legs)):
        raise ValueError("Knee equality residual exceeds diagnostic limit")
    jac = [leg.evaluate(hip_positions[i], knee_positions[i])[1] for i, leg in enumerate(legs)]
    if any(not math.isfinite(j) or abs(j)/m < MIN_JACOBIAN_RATIO for j, m in zip(jac, jmax)):
        raise ValueError("Degenerate or nonfinite Jacobian")
    return jac
