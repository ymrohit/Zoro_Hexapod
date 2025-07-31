"""
hex_controller.py
A background IK + gait controller for the Freenove Big Hexapod.

- Inverse kinematics for 3-DOF legs
- Body pose (x,y,z, roll, pitch, yaw)
- Tripod & Ripple gaits with smooth foot trajectories
- Per-servo direction and auto-zero from stance
- Angle safety clamps and simple low-pass filtering
"""

import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------- Geometry (updated from your photos) ----------
# Link lengths (mm)
L1 = 30.0    # coxa length
L2 = 100.0   # femur length
L3 = 125.0   # tibia length

# Body footprint (mm) — square ~20×20 cm
W = 200.0    # hip-to-hip width  (left↔right)
L = 200.0    # hip-to-hip length (front↔rear)

# Leg base locations in BODY frame (mm). +X forward, +Y left, +Z up.
LEG_BASE = {
    1: (+W/2, -L/2, 0.0),
    2: (+W/2,    0.0, 0.0),
    3: (+W/2, +L/2, 0.0),
    4: (-W/2, +L/2, 0.0),
    5: (-W/2,    0.0, 0.0),
    6: (-W/2, -L/2, 0.0),
}

# Servo mapping (your existing map)
LEG_MAP = {
    1: [15, 14, 13],
    2: [12, 11, 10],
    3: [9,  8,  31],
    4: [22, 23, 27],
    5: [19, 20, 21],
    6: [16, 17, 18],
}

# Per-servo direction (+1/-1). Defaults then updated below with reasonable guesses.
SERVO_DIR: Dict[int, int] = {sid: +1 for sids in LEG_MAP.values() for sid in sids}
# Zero offset so that adding ZERO_OFF to the IK angle centers around 90° at the stance pose.
ZERO_OFF: Dict[int, float] = {sid: 0.0 for sids in LEG_MAP.values() for sid in sids}

# Reasonable default directions (right legs vs left legs, tibias mirrored)
# Right side legs (1,2,3): coxa +1, femur +1, tibia -1
# Left  side legs (4,5,6): coxa -1, femur -1, tibia +1
SERVO_DIR.update({
    # Leg 1 (coxa,femur,tibia)
    15:+1, 14:+1, 13:-1,
    # Leg 2
    12:+1, 11:+1, 10:-1,
    # Leg 3
    9:+1,  8:+1,  31:-1,
    # Leg 4 (mirrored side)
    22:-1, 23:-1, 27:+1,
    # Leg 5
    19:-1, 20:-1, 21:+1,
    # Leg 6
    16:-1, 17:-1, 18:+1,
})

# ---------- Helpers ----------
def clamp(a, x, b): 
    return max(a, min(b, x))

def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return ((1,0,0),(0,ca,-sa),(0,sa,ca))

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return ((ca,0,sa),(0,1,0),(-sa,0,ca))

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return ((ca,-sa,0),(sa,ca,0),(0,0,1))

def mat_vec(M, v):
    return (M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2])

def add(a,b): 
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def sub(a,b): 
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def smoothstep(s): 
    # cubic ease-in-out 0..1
    return s*s*(3 - 2*s)

def multiply(A,B):
    return tuple(tuple(sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)) for i in range(3))

def multiply3(A,B,C):
    return multiply(A, multiply(B,C))

def transpose(A):
    return ((A[0][0],A[1][0],A[2][0]),
            (A[0][1],A[1][1],A[2][1]),
            (A[0][2],A[1][2],A[2][2]))

# ---------- IK for a single leg ----------
def leg_ik(x, y, z):
    """
    Input: desired foot position (mm) in the leg's base frame.
    Output: (q1, q2, q3) in degrees: coxa yaw, femur pitch, tibia pitch.
    Returns None if unreachable.
    """
    q1 = math.atan2(y, x)
    r = math.hypot(x, y) - L1
    # planar 2-link
    D = (r*r + z*z - L2*L2 - L3*L3) / (2*L2*L3)
    if D < -1.0 or D > 1.0:
        return None
    q3 = -math.acos(clamp(-1.0, D, 1.0))  # knee down preference
    q2 = math.atan2(z, r) - math.atan2(L3*math.sin(q3), L2 + L3*math.cos(q3))
    return (math.degrees(q1), math.degrees(q2), math.degrees(q3))

# ---------- Gait engine ----------
@dataclass
class GaitParams:
    name: str = "tripod"    # "tripod" or "ripple"
    freq_hz: float = 1.6
    duty: float = 0.66
    step_len: float = 45.0    # mm
    step_height: float = 20.0 # mm

GAIT_PHASES = {
    "tripod": {1:0.0, 3:0.0, 5:0.0, 2:math.pi, 4:math.pi, 6:math.pi},
    "ripple": {1:0.0, 2:math.pi/3, 3:2*math.pi/3, 4:math.pi, 5:4*math.pi/3, 6:5*math.pi/3},
}

@dataclass
class Pose:
    x: float = 0.0      # mm
    y: float = 0.0      # mm
    z: float = -120.0   # body height (negative down)
    roll: float = 0.0   # deg
    pitch: float = 0.0  # deg
    yaw: float = 0.0    # deg

@dataclass
class VelocityCmd:
    vx: float = 0.0     # mm/s  +X forward
    vy: float = 0.0     # mm/s  +Y left
    wz: float = 0.0     # rad/s yaw rate

@dataclass
class HexState:
    pose: Pose = field(default_factory=Pose)
    vel: VelocityCmd = field(default_factory=VelocityCmd)
    gait: GaitParams = field(default_factory=GaitParams)
    walking: bool = False

class HexController:
    def __init__(self, servo_iface, write_hz: float = 50.0):
        """
        servo_iface must implement set_servo_angle(channel:int, angle_degrees:float)
        """
        self.servo = servo_iface
        self.state = HexState()
        self._thread = None
        self._stop = threading.Event()
        self.write_hz = write_hz

        # nominal stance foot positions (BODY frame); tuned for L2=100, L3=125
        self.stance = {
            1: (+100,  -85, -120),
            2: (+110,    0, -120),
            3: (+100,  +85, -120),
            4: (-100,  +85, -120),
            5: (-110,    0, -120),
            6: (-100,  -85, -120),
        }

        # Compute ZERO_OFF from stance so initial command is neutral (≈90°) for all joints
        _auto_zero_from_stance(self.stance, LEG_BASE)

        # start with joints at 90 to avoid a jump on first tick
        self.prev_deg: Dict[int, float] = {sid: 90.0 for sids in LEG_MAP.values() for sid in sids}

        # Safety angle margins per joint type
        self.safe_limits = {
            0: (10.0, 170.0),  # coxa
            1: (15.0, 165.0),  # femur
            2: (15.0, 165.0),  # tibia
        }

    # ---------- Public API ----------
    def set_pose(self, pose: Pose): 
        self.state.pose = pose

    def set_velocity(self, vel: VelocityCmd):
        self.state.vel = vel
        self.state.walking = (abs(vel.vx)+abs(vel.vy)+abs(vel.wz) > 1e-6)

    def set_gait(self, gait: GaitParams): 
        self.state.gait = gait

    def stop(self): 
        self.state.walking = False
        self.state.vel = VelocityCmd()

    def start(self):
        if self._thread and self._thread.is_alive(): 
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self):
        self._stop.set()
        if self._thread: 
            self._thread.join(timeout=1.0)

    # ---------- Core loop ----------
    def _loop(self):
        dt = 1.0 / self.write_hz
        t0 = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter() - t0
            try:
                self._tick(t, dt)
            except Exception as e:
                # If something goes wrong, try to avoid thrashing servos
                # (In production, consider logging this exception.)
                pass
            time.sleep(dt)

    def _tick(self, t: float, dt: float):
        # 1) desired feet in BODY frame (apply gait)
        feet_body = {i: self._foot_target(i, t) for i in range(1,7)}

        # 2) transform each foot to leg base frame (apply body pose)
        pose = self.state.pose
        Rx = rot_x(math.radians(pose.roll))
        Ry = rot_y(math.radians(pose.pitch))
        Rz = rot_z(math.radians(pose.yaw))
        R = multiply3(Rz, Ry, Rx)  # Z * Y * X
        Rt = transpose(R)
        T = (pose.x, pose.y, pose.z)

        for i in range(1,7):
            bx, by, bz = LEG_BASE[i]
            hip_world = add(T, mat_vec(R, (bx,by,bz)))
            v = sub(feet_body[i], hip_world)
            p_leg = mat_vec(Rt, v)
            # 3) IK
            res = leg_ik(*p_leg)
            if res is None:
                # unreachable this tick: keep previous angles for this leg
                continue
            q1, q2, q3 = res
            # 4) send to servos with dir/offset/clamp and smoothing
            for j, (sid, q) in enumerate(zip(LEG_MAP[i], (q1,q2,q3))):
                raw = 90.0 + SERVO_DIR[sid]*(q + ZERO_OFF[sid])
                lo, hi = self.safe_limits[j]
                raw = clamp(lo, raw, hi)
                prev = self.prev_deg.get(sid, raw)
                # 1st order low-pass to reduce jerk
                alpha = 0.65
                cmd = alpha*prev + (1.0-alpha)*raw
                self.servo.set_servo_angle(sid, cmd)
                self.prev_deg[sid] = cmd

    # Elliptical foot path with duty factor
    def _foot_target(self, leg: int, t: float):
        base = self.stance[leg]
        g = self.state.gait

        if not self.state.walking:
            # allow body translation/rotation without stepping
            return base

        phases = GAIT_PHASES.get(g.name, GAIT_PHASES["tripod"])
        u = (2*math.pi*g.freq_hz*t + phases[leg]) % (2*math.pi)
        duty = clamp(0.1, g.duty, 0.95)

        # Convert commanded body velocity into stance frame foot motion
        vx, vy, wz = self.state.vel.vx, self.state.vel.vy, self.state.vel.wz
        # simple scaling: ~100 mm/s maps to full step length
        Sx = clamp(-g.step_len, (vx/100.0) * g.step_len, g.step_len)
        Sy = clamp(-g.step_len, (vy/100.0) * g.step_len, g.step_len)

        if u < 2*math.pi*duty:
            s = u/(2*math.pi*duty)  # 0..1 stance
            s = smoothstep(s)
            x = base[0] - (Sx*(s - 0.5))
            y = base[1] - (Sy*(s - 0.5))
            z = base[2]
        else:
            s = (u - 2*math.pi*duty)/(2*math.pi*(1.0 - duty))  # 0..1 swing
            s = smoothstep(s)
            x = base[0] + (Sx*(0.5 - (s - 0.5)))
            y = base[1] + (Sy*(0.5 - (s - 0.5)))
            z = base[2] + g.step_height*math.sin(math.pi*s)

        # tiny yaw-based correction for aesthetics (optional)
        if abs(self.state.vel.wz) > 1e-4:
            r = 0.15 * g.step_len
            ang = self.state.vel.wz / (2*math.pi)
            x += -r*math.sin(ang)
            y +=  r*math.cos(ang)

        return (x, y, z)


def _auto_zero_from_stance(stance, leg_base):
    """
    Choose ZERO_OFF so that the configured stance produces ~90° commands on all servos.
    This prevents the first tick from yanking joints to extremes.
    """
    for i in range(1, 7):
        bx, by, bz = leg_base[i]
        px = stance[i][0] - bx
        py = stance[i][1] - by
        pz = stance[i][2] - bz
        res = leg_ik(px, py, pz)
        if res is None:
            continue
        q1, q2, q3 = res  # degrees
        for sid, q in zip(LEG_MAP[i], (q1, q2, q3)):
            ZERO_OFF[sid] = -q  # so 90 + dir*(q + ZERO_OFF) == 90