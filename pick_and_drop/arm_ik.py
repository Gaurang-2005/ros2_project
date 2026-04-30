"""
arm_ik.py  —  Inverse Kinematics for pick_and_drop arm
=======================================================
Arm geometry from SDF joint poses:
  base_joint  : at base_link + (0,0,0.1)      — rotates Z
  joint1      : at link1    + (0,0,1.677)     — rotates X
  joint2      : at arm1     + (0, 2.057, 2.083) — rotates X
  joint3      : at arm2     + (0,-2.041, 2.094) — rotates X
  joint4      : at arm3     + (0,-2.105,-0.303) — rotates X

Because joints 1-4 all rotate around X, the arm moves in the
local Y-Z plane. base_joint rotates that plane around Z.

IK strategy:
  1. base_joint  = atan2(y, x)          — aim the plane at target
  2. r           = sqrt(x²+y²)          — radial distance
  3. z_rel       = z - base_height      — height above joint1
  4. Solve joints 1-4 as planar 4-DOF in (r, z_rel) using
     geometric 2-link approach on the dominant segments,
     with joint3/4 used to control wrist angle.

Link lengths (Y-Z resultant from SDF offsets):
  L1 = |joint2 - joint1| = sqrt(2.057² + 2.083²) ≈ 2.921 m
  L2 = |joint3 - joint2| = sqrt(2.041² + 2.094²) ≈ 2.922 m
  L3 = |joint4 - joint3| = sqrt(2.105² + 0.303²) ≈ 2.127 m
"""

import math
from typing import Optional, Tuple

# ── Link lengths from SDF (metres) ────────────────────────────────────────────
BASE_HEIGHT  = 0.1 + 1.677   # base_joint Z + joint1 Z offset = 1.777 m
L1           = math.sqrt(2.057**2 + 2.083**2)   # 2.921 m  (arm1 segment)
L2           = math.sqrt(2.041**2 + 2.094**2)   # 2.922 m  (arm2 segment)
L3           = math.sqrt(2.105**2 + 0.303**2)   # 2.127 m  (arm3 + wrist)

# ── Joint limits from SDF ─────────────────────────────────────────────────────
JOINT_LIMITS = {
    "base_joint"   : (-3.14,  3.14),
    "joint1"       : (-1.57,  1.57),
    "joint2"       : (-1.57,  1.57),
    "joint3"       : (-1.57,  1.57),
    "joint4"       : (-1.57,  1.57),
    "finger1_joint": ( 0.0,   0.80),
    "finger2_joint": ( 0.0,   0.80),
}

# ── Hardcoded fallback poses (joint angles in radians) ────────────────────────
# Tuned visually — safe starting points you can refine in Gazebo.
FALLBACK_POSES = {
    "home": {
        "base_joint": 0.0, "joint1": 0.0, "joint2": 0.0,
        "joint3": 0.0, "joint4": 0.0,
        "finger1_joint": 0.0, "finger2_joint": 0.0,
    },
    # Arm pointing forward, elbow up, ready to descend
    "pre_pick": {
        "base_joint": 0.0,  "joint1":  0.60, "joint2": -0.80,
        "joint3": 0.50,     "joint4":  0.20,
        "finger1_joint": 0.0, "finger2_joint": 0.0,
    },
    # Arm lowered to pick height
    "pick": {
        "base_joint": 0.0,  "joint1":  0.90, "joint2": -1.10,
        "joint3": 0.60,     "joint4":  0.30,
        "finger1_joint": 0.0, "finger2_joint": 0.0,
    },
    # Gripper closed
    "grip": {
        "base_joint": 0.0,  "joint1":  0.90, "joint2": -1.10,
        "joint3": 0.60,     "joint4":  0.30,
        "finger1_joint": 0.70, "finger2_joint": 0.70,
    },
    # Arm lifted with object
    "lift": {
        "base_joint": 0.0,  "joint1":  0.50, "joint2": -0.70,
        "joint3": 0.40,     "joint4":  0.20,
        "finger1_joint": 0.70, "finger2_joint": 0.70,
    },
    # Rotated to drop zone (base_joint overridden at runtime by DROP_BASE_ANGLE)
    "pre_place": {
        "base_joint": 1.57, "joint1":  0.60, "joint2": -0.80,
        "joint3": 0.50,     "joint4":  0.20,
        "finger1_joint": 0.70, "finger2_joint": 0.70,
    },
    # Lower to place
    "place": {
        "base_joint": 1.57, "joint1":  0.90, "joint2": -1.10,
        "joint3": 0.60,     "joint4":  0.30,
        "finger1_joint": 0.70, "finger2_joint": 0.70,
    },
    # Release
    "release": {
        "base_joint": 1.57, "joint1":  0.90, "joint2": -1.10,
        "joint3": 0.60,     "joint4":  0.30,
        "finger1_joint": 0.0, "finger2_joint": 0.0,
    },
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _in_limits(name: str, angle: float) -> bool:
    lo, hi = JOINT_LIMITS[name]
    return lo <= angle <= hi


def solve_ik(
    x: float, y: float, z: float,
    wrist_angle: float = 0.0,
    fingers_closed: bool = False,
) -> Optional[dict]:
    """
    Compute joint angles to place the wrist at (x, y, z) in world frame.

    Parameters
    ----------
    x, y, z       : Target position in metres (world frame, robot at origin)
    wrist_angle   : Desired angle of the wrist link w.r.t. horizontal (rad).
                    0 = horizontal, positive = tilted down.
    fingers_closed: True → fingers at 0.70 rad, False → 0.0 rad

    Returns
    -------
    dict of joint angles, or None if unreachable.
    """

    # ── 1. Base rotation ──────────────────────────────────────────────────────
    base = math.atan2(y, x)
    if not _in_limits("base_joint", base):
        return None

    # ── 2. Radial & vertical components ───────────────────────────────────────
    r     = math.sqrt(x**2 + y**2)          # horizontal distance from base axis
    z_rel = z - BASE_HEIGHT                  # height relative to joint1

    # Account for wrist (L3 contributes toward target)
    # Wrist tip to elbow: back-project by L3 along desired wrist direction
    wx = r   - L3 * math.cos(wrist_angle)
    wz = z_rel - L3 * math.sin(-wrist_angle)

    reach = math.sqrt(wx**2 + wz**2)

    # ── 3. Reachability check ─────────────────────────────────────────────────
    if reach > (L1 + L2) * 0.99:
        return None   # too far
    if reach < abs(L1 - L2) * 1.01:
        return None   # too close (singularity)

    # ── 4. Two-link planar IK for joint1 & joint2 ─────────────────────────────
    cos_j2 = (wx**2 + wz**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_j2 = _clamp(cos_j2, -1.0, 1.0)
    # Elbow-up solution (negative sin)
    j2_raw = -math.acos(cos_j2)

    k1 = L1 + L2 * math.cos(j2_raw)
    k2 = L2 * math.sin(j2_raw)

    # Angle from horizontal to reach point
    gamma = math.atan2(wz, wx)
    alpha = math.atan2(k2, k1)
    j1_raw = gamma - alpha

    # ── 5. Wrist joints to maintain desired end angle ─────────────────────────
    # Total arm angle so far: j1 + j2
    arm_angle = j1_raw + j2_raw
    # joint3 corrects for wrist_angle (keep end-effector at desired pitch)
    j3_raw = wrist_angle - arm_angle
    # joint4 keeps the gripper level
    j4_raw = 0.0

    # ── 6. Validate all limits ────────────────────────────────────────────────
    checks = [
        ("joint1", j1_raw),
        ("joint2", j2_raw),
        ("joint3", j3_raw),
        ("joint4", j4_raw),
    ]
    for name, val in checks:
        if not _in_limits(name, val):
            return None

    finger_val = 0.70 if fingers_closed else 0.0

    return {
        "base_joint"   : base,
        "joint1"       : j1_raw,
        "joint2"       : j2_raw,
        "joint3"       : j3_raw,
        "joint4"       : j4_raw,
        "finger1_joint": finger_val,
        "finger2_joint": finger_val,
    }


def get_pose_for_target(
    x: float, y: float, z: float,
    phase: str,                    # "pre_pick" | "pick" | "grip" | etc.
    fingers_closed: bool = False,
) -> Tuple[dict, bool]:
    """
    Try IK first; fall back to hardcoded pose if IK fails.

    Returns (pose_dict, used_ik: bool)
    """
    wrist_angles = {
        "pre_pick" : -0.30,   # slightly angled down
        "pick"     : -0.50,   # more angled down to reach object
        "grip"     :  -0.50,
    }
    w = wrist_angles.get(phase, 0.0)

    ik_result = solve_ik(x, y, z, wrist_angle=w, fingers_closed=fingers_closed)

    if ik_result is not None:
        return ik_result, True
    else:
        fallback = dict(FALLBACK_POSES.get(phase, FALLBACK_POSES["home"]))
        # Point base at the target even in fallback
        fallback["base_joint"] = _clamp(
            math.atan2(y, x), *JOINT_LIMITS["base_joint"]
        )
        return fallback, False