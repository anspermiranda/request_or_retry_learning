#!/usr/bin/env python3
"""
twin.py: a copy of my table in MuJoCo, with the pick-and-place task written as a reinforcement-learning environment.

Everything lives in the frame of the paper marker on my table (metres; x to the right, y away from the phone, z up),
so whatever read_demos.py measured in my videos drops straight in: the coaster, where the can starts, its path.

The task as an RL problem. One episode is one attempt:
  state   s : gripper position (3), can position (3), gripper closed? (1), finger opening (1)
  action  a : move the gripper by up to 2.5 cm along x, y and z, and open or close it. 10 decisions a second.
  reward  r : 1 at the end if the can stands upright on the coaster, the gripper has let go and nothing else on the
              table was touched; 0 otherwise. Nothing in between, so the reward is sparse.
  episode   : ends when the job is done, when the can falls over, or after 24 s.

The arm is the Franka Panda from MuJoCo Menagerie. Each move goes through inverse kinematics (mink) and the joints
follow with the Panda's own position controllers, 50 times a second.

Two shortcuts, both because simulated fingertip pads wedge into a cylinder: a weld holds the can while both fingers
touch it, and the fingers ignore the can while they open. Reaching, collisions, carrying, setting down and standing
upright are all physics.
"""
import os

import mujoco
import numpy as np

MENAGERIE = os.path.join("mujoco_menagerie", "franka_emika_panda", "panda.xml")
ROBOT_BASE = (-0.25, -0.50, 0.0)          # where I sat, so the arm comes in from the same side as my hand
ROBOT_YAW = np.pi / 2                     # facing up the table (+y)
CAN_RADIUS, CAN_HEIGHT, CAN_MASS = 0.0315, 0.175, 0.25   # a full can (a very light one makes contacts too soft)
COASTER_THICKNESS = 0.005
GRASP_Z = 0.141                           # fingertips 14.1 cm above the can's base: the pads close on the top of the
                                          # straight body and the palm clears the lid
V_MAX = 0.25                              # the gripper never moves faster than 25 cm/s
APPROACH_UP = 0.10
CONTROL_DT, SIM_DT = 0.02, 0.002          # 50 Hz control, 500 Hz physics
STEP_DT = 0.1                             # one decision every 0.1 s (5 control steps)
DMAX = V_MAX * STEP_DT                    # so a decision moves the gripper 2.5 cm at most
MAX_S = 24.0                              # an attempt is stopped after 24 s
OPEN, CLOSED = 255.0, 0.0
DOWN = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])   # gripper pointing straight down
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])    # the Panda's usual ready pose

# The other things on my table, measured on the bird's-eye view: centre (x, y), half sizes, yaw (degrees), colour
OBSTACLES = {
    "laptop_base": ((-0.210, 0.212), (0.185, 0.133, 0.012), 0, (0.25, 0.25, 0.28, 1)),
    "laptop_screen": ((-0.210, 0.355), (0.185, 0.006, 0.115), 0, (0.12, 0.12, 0.14, 1)),
    "book": ((0.115, 0.222), (0.075, 0.110, 0.013), -25, (0.85, 0.35, 0.20, 1)),
    "desk_hub": ((-0.590, -0.100), (0.065, 0.065, 0.040), 0, (0.08, 0.08, 0.08, 1)),
    "glasses_case": ((-0.480, -0.130), (0.030, 0.110, 0.030), 20, (0.05, 0.05, 0.05, 1)),
    "charger": ((-0.425, 0.085), (0.020, 0.020, 0.020), 0, (0.95, 0.95, 0.95, 1)),
    "watch": ((0.180, -0.020), (0.030, 0.040, 0.010), 0, (0.70, 0.70, 0.72, 1)),
}
CAN_RGBA = (0.45, 0.78, 0.90, 1)          # light blue, like my can: nothing else on the table has this colour

# Cameras. "overview" and "closeup" film the videos. "eye" is the robot's own camera for finding the can: high up on the far
# side of the table, looking back at the robot, so the arm in its ready pose never hides the can.
OVERVIEW_CAM = ((0.55, -1.15, 1.05), (-0.22, 0.0, 0.12))
CLOSEUP_CAM = ((0.30, -0.95, 0.80), (-0.25, -0.12, 0.05))
EYE_CAM = ((-0.20, 0.80, 1.20), (-0.20, -0.20, 0.05))


def look_at(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return [*x, *y]


def objects_under(x, y, margin=0.0):
    """Names of the objects whose top is under the point (x, y)."""
    names = []
    for name, ((cx, cy), half, yaw, _) in OBSTACLES.items():
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        lx, ly = c * (x - cx) + s * (y - cy), -s * (x - cx) + c * (y - cy)
        if abs(lx) < half[0] + margin and abs(ly) < half[1] + margin:
            names.append(name)
    return names


def support_height(x, y, margin=0.0):
    """Height of whatever the can stands on at (x, y): the table (0) or the top of an object. Two of my videos start
    with the can on the laptop's palm rest."""
    return max([0.0] + [2 * OBSTACLES[n][1][2] for n in objects_under(x, y, margin)])


def build_scene(coaster, camera=None):
    """The table, the coaster, the other objects, the can and the Panda, all where they are in my videos.
    camera: my phone's pose in one clip (from read_demos.py), to add a 'phone' camera."""
    spec = mujoco.MjSpec()
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC          # better friction model for holding things
    spec.option.impratio = 10
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    wb = spec.worldbody
    wb.add_light(pos=[-0.2, -0.3, 1.6], dir=[0.1, 0.2, -1], diffuse=[0.75, 0.75, 0.75], castshadow=True)
    wb.add_light(pos=[0.4, 0.6, 1.2], dir=[-0.3, -0.4, -1], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    # Collision bits: the world (table, coaster, objects) collides with everything (affinity 7), the can has type 2,
    # the finger pads type 4. While the can is held, its affinity drops to 3 so the pads stop pushing into it.
    wb.add_geom(name="table", type=box, size=[0.80, 0.65, 0.02], pos=[-0.20, 0.0, -0.02], rgba=[0.80, 0.62, 0.42, 1],
                conaffinity=7)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                rgba=[0.55, 0.56, 0.58, 1], contype=0, conaffinity=0)
    wb.add_geom(name="paper", type=box, size=[0.105, 0.1485, 0.0005], pos=[0.033, -0.077, 0.0005],
                rgba=[0.97, 0.97, 0.97, 1], contype=0, conaffinity=0)
    wb.add_geom(name="marker", type=box, size=[0.0478, 0.0478, 0.0006], pos=[0, 0, 0.0007], rgba=[0.02, 0.02, 0.02, 1],
                contype=0, conaffinity=0)
    wb.add_geom(name="coaster", type=cyl, size=[coaster["diameter"] / 2, COASTER_THICKNESS / 2],
                pos=[*coaster["center"], COASTER_THICKNESS / 2], rgba=[0.06, 0.06, 0.06, 1], conaffinity=7)
    for name, ((x, y), half, yaw, rgba) in OBSTACLES.items():
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, [0, 0, np.radians(yaw)], "xyz")
        wb.add_geom(name=name, type=box, size=list(half), pos=[x, y, half[2]], quat=q, rgba=list(rgba), conaffinity=7)
    can = wb.add_body(name="can", pos=[0.3, 0.3, CAN_HEIGHT / 2 + 0.001])   # each episode puts it where it should be
    can.add_freejoint(name="can_joint")
    can.add_geom(name="can_geom", type=cyl, size=[CAN_RADIUS, CAN_HEIGHT / 2], mass=CAN_MASS, condim=4,
                 friction=[1.0, 0.02, 0.001], rgba=list(CAN_RGBA), contype=2, conaffinity=7)
    can.add_geom(type=cyl, size=[CAN_RADIUS * 0.94, 0.006], pos=[0, 0, CAN_HEIGHT / 2 - 0.006], rgba=[0.80, 0.80, 0.82, 1],
                 contype=0, conaffinity=0, mass=0)
    wb.add_camera(name="overview", pos=list(OVERVIEW_CAM[0]), xyaxes=look_at(*OVERVIEW_CAM), fovy=50)
    wb.add_camera(name="closeup", pos=list(CLOSEUP_CAM[0]), xyaxes=look_at(*CLOSEUP_CAM), fovy=50)
    wb.add_camera(name="eye", pos=list(EYE_CAM[0]), xyaxes=look_at(*EYE_CAM), fovy=60)
    if camera is not None:   # my phone, exactly where it was (from the calibration and the marker)
        import cv2
        R = cv2.Rodrigues(np.array(camera["rvec"], float))[0]
        C = -R.T @ np.array(camera["tvec"], float)
        wb.add_camera(name="phone", pos=C.tolist(), xyaxes=[*R.T[:, 0], *(-R.T[:, 1])],
                      fovy=float(np.degrees(2 * np.arctan(camera["image_h"] / (2 * camera["fy"])))))
    panda = mujoco.MjSpec.from_file(MENAGERIE)
    panda.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034])
    for b in panda.bodies:                      # the real Panda compensates its own weight; without this the
        b.gravcomp = 1.0                        # simulated arm sags about 2 cm and the fingers land on the can's rim
        if "finger" in b.name:
            for g in b.geoms:
                if g.contype:
                    g.contype, g.conaffinity = 4, 4
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, [0, 0, ROBOT_YAW], "xyz")
    wb.add_frame(pos=list(ROBOT_BASE), quat=q).attach_body(panda.body("link0"), "panda_", "")
    # grasp assist: a weld between hand and can, switched on only while both fingers press on the can
    spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1="panda_hand", name2="can", active=False, name="hold")
    return spec.compile()


def set_base(x, y):
    """Stand the robot somewhere else on the table (a different embodiment of the same task), facing the coaster."""
    global ROBOT_BASE, ROBOT_YAW
    ROBOT_BASE = (x, y, 0.0)
    ROBOT_YAW = float(np.arctan2(-0.19 - y, -0.16 - x))


# ---------------------------------------------------------------- the arm
class Robot:
    """The Panda and the can for one episode: inverse kinematics, the grasp weld, and contact checks."""

    def __init__(self, model, can_xy):
        import mink
        self.mink, self.model = mink, model
        self.data = mujoco.MjData(model)
        self.can_qadr = model.jnt_qposadr[model.joint("can_joint").id]
        self.arm_qadr = np.array([model.jnt_qposadr[model.joint(f"panda_joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = np.array([model.jnt_qposadr[model.joint(f"panda_finger_joint{i}").id] for i in (1, 2)])
        self.data.qpos[self.arm_qadr] = HOME
        self.data.qpos[self.finger_qadr] = 0.04
        self.data.ctrl[:7], self.data.ctrl[7] = HOME, OPEN
        z = support_height(can_xy[0], can_xy[1]) + CAN_HEIGHT / 2 + 0.001
        self.data.qpos[self.can_qadr:self.can_qadr + 7] = [can_xy[0], can_xy[1], z, 1, 0, 0, 0]
        mujoco.mj_forward(model, self.data)
        self.cfg = mink.Configuration(model)
        self.cfg.update(self.data.qpos)
        self.task = mink.FrameTask("panda_tcp", "site", position_cost=1.0, orientation_cost=0.5)
        self.posture = mink.PostureTask(model, cost=1e-2)
        self.posture.set_target_from_configuration(self.cfg)
        self.limits = [mink.ConfigurationLimit(model)]
        self.arm_ctrl = np.arange(7)
        mujoco.mj_forward(model, self.data)
        self.robot_geoms = {g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith("panda_")}
        self.hold = model.equality("hold").id
        self.hand, self.can_body = model.body("panda_hand").id, model.body("can").id
        self.fingers = {model.body("panda_left_finger").id, model.body("panda_right_finger").id}
        self.can_geom = model.geom("can_geom").id
        self.holding, self.releasing, self.hold_ctrl = False, False, CLOSED
        self._fingers_touch_can(True)
        self.obstacles = {model.geom(n).id: n for n in OBSTACLES}
        # what the can starts on or right next to (within 1 cm): touching those while lifting it off doesn't count
        self.start_xy = np.array(can_xy[:2], float)
        self.nearby = {model.geom(n).id for n in objects_under(can_xy[0], can_xy[1], margin=CAN_RADIUS + 0.01)}

    def _fingers_touch_can(self, on):
        self.model.geom_conaffinity[self.can_geom] = 7 if on else 3
        self.releasing = False

    def fingers_on_can(self):
        touching = set()
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if self.can_body in (b1, b2):
                touching.add(b2 if b1 == self.can_body else b1)
        return self.fingers <= touching            # both fingers must touch it: a real grasp, not a nudge

    def grab(self):
        """Switch on the weld, keeping the can exactly where it is relative to the hand right now."""
        d, m = self.data, self.model
        ph, qh = d.xpos[self.hand], d.xquat[self.hand]
        pc, qc = d.xpos[self.can_body], d.xquat[self.can_body]
        qh_inv = np.zeros(4)
        mujoco.mju_negQuat(qh_inv, qh)
        rel_p = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_p, pc - ph, qh_inv)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, qh_inv, qc)
        m.eq_data[self.hold, 0:3] = 0.0
        m.eq_data[self.hold, 3:6] = rel_p
        m.eq_data[self.hold, 6:10] = rel_q
        m.eq_data[self.hold, 10] = 1.0
        d.eq_active[self.hold] = 1
        self.holding = True
        self._fingers_touch_can(False)             # the weld holds it now: the fingers stop squeezing
        self.hold_ctrl = float(np.clip(self.width() / 0.000314, 0, 255))

    def tcp(self):
        return self.data.site("panda_tcp").xpos.copy()

    def width(self):
        return float(self.data.qpos[self.finger_qadr].sum())

    def can(self):
        return self.data.qpos[self.can_qadr:self.can_qadr + 7].copy()

    def pose(self):
        """Joint angles, finger joints and the can's pose: enough to draw this moment again later."""
        return np.r_[self.data.qpos[np.r_[self.arm_qadr, self.finger_qadr]], self.can()]

    def step(self, target_xyz, grip):
        """One control step (20 ms) towards target_xyz with the gripper pointing down. Returns the names of the
        objects the arm touched, or the can touched while being carried."""
        mink = self.mink
        yaw = np.arctan2(target_xyz[1] - ROBOT_BASE[1], target_xyz[0] - ROBOT_BASE[0]) - ROBOT_YAW
        c, s = np.cos(yaw + ROBOT_YAW - np.pi / 2), np.sin(yaw + ROBOT_YAW - np.pi / 2)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ DOWN
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R), target_xyz))
        vel = mink.solve_ik(self.cfg, [self.task, self.posture], CONTROL_DT, solver="daqp", damping=1e-3, limits=self.limits)
        self.cfg.integrate_inplace(vel, CONTROL_DT)
        self.data.ctrl[self.arm_ctrl] = self.cfg.q[self.arm_qadr]
        self.data.ctrl[7] = self.hold_ctrl if (self.holding and grip == CLOSED) else grip
        if grip == OPEN and self.holding:                  # let go
            self.data.eq_active[self.hold] = 0
            self.holding = False
            self.releasing = True                          # fingers ignore the can until the hand is clear of it
        elif self.releasing:
            if self.tcp()[2] > self.can()[2] + CAN_HEIGHT / 2 + 0.01 or grip == CLOSED:
                self._fingers_touch_can(True)
        elif grip == CLOSED and not self.holding and self.fingers_on_can():
            self.grab()
        hits = set()
        for _ in range(int(round(CONTROL_DT / SIM_DT))):
            mujoco.mj_step(self.model, self.data)
            for c in self.data.contact[:self.data.ncon]:
                for ob, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if ob not in self.obstacles:
                        continue
                    if other in self.robot_geoms:                      # the arm touching anything counts
                        hits.add(self.obstacles[ob])
                    elif other == self.can_geom and self.holding:      # so does the can, while it's being carried,
                        lifting_off = np.linalg.norm(self.data.qpos[self.can_qadr:self.can_qadr + 2] - self.start_xy) < 0.03
                        if not (ob in self.nearby and lifting_off):     # except against what it stood on or next to,
                            hits.add(self.obstacles[ob])               # while it's still being lifted off
        return hits


# ---------------------------------------------------------------- the task as an environment
def finished(robot, coaster):
    """The episode is over: can on the coaster, let go and the hand above it, or the can has fallen."""
    can = robot.can()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, can[3:])
    if R[8] < np.cos(np.radians(50)) or can[2] < 0:
        return True
    on = np.linalg.norm(can[:2] - np.array(coaster["center"])) < coaster["diameter"] / 2
    resting = can[2] < CAN_HEIGHT / 2 + COASTER_THICKNESS + 0.01
    return on and resting and not robot.holding and robot.tcp()[2] > can[2] + CAN_HEIGHT / 2 + 0.06


def nice(name):
    return {"laptop_base": "laptop", "laptop_screen": "laptop screen", "desk_hub": "desk hub",
            "glasses_case": "glasses case"}.get(name, name)


def judge(robot, coaster, hits):
    """The reward: 1 only if the can stands upright on the coaster, the gripper has let go and nothing was hit."""
    q = robot.can()
    pos, quat = q[:3], q[3:]
    up = np.zeros(9)
    mujoco.mju_quat2Mat(up, quat)
    tilt = np.degrees(np.arccos(np.clip(up[8], -1, 1)))
    off = float(np.linalg.norm(pos[:2] - np.array(coaster["center"])))
    if tilt > 20:
        why = "tipped over"
    elif robot.holding:
        why = "never let go"
    elif pos[2] > CAN_HEIGHT / 2 + COASTER_THICKNESS + 0.01:
        why = "still held or stuck"
    elif off > coaster["diameter"] / 2:
        why = "missed the coaster"
    elif hits:
        why = "touched the " + " and the ".join(sorted(nice(h) for h in hits))
    else:
        why = "success"
    return why == "success", why, off, tilt


class TableEnv:
    """reset(can_xy) -> state; step(move, grip) -> state, reward, done, info. Same idea as a Gym environment."""

    def __init__(self, model, coaster, max_s=MAX_S):
        self.model, self.coaster, self.max_steps = model, coaster, int(round(max_s / STEP_DT))

    def reset(self, can_xy):
        self.robot = Robot(self.model, can_xy)
        self.grip, self.hits, self.t = OPEN, set(), 0
        return self.state()

    def state(self):
        r = self.robot
        return np.r_[r.tcp(), r.can()[:3], float(self.grip == CLOSED), r.width()]

    def step(self, move, grip, viewer=None):
        move = np.clip(np.asarray(move, float), -DMAX, DMAX)
        target, self.grip = self.robot.tcp() + move, grip
        for _ in range(int(round(STEP_DT / CONTROL_DT))):
            self.hits |= self.robot.step(target, grip)
            if viewer is not None and viewer.is_running():
                import time
                viewer.sync()
                time.sleep(CONTROL_DT)
        self.t += 1
        done = finished(self.robot, self.coaster) or self.t >= self.max_steps
        info = {}
        if done:
            ok, why, off, tilt = judge(self.robot, self.coaster, self.hits)
            info = {"success": ok, "why": why, "offset_cm": round(100 * off, 1), "tilt_deg": round(float(tilt), 1),
                    "hits": sorted(self.hits)}
        return self.state(), float(info.get("success", False)), done, info


def rollout(env, controller, can_xy, noise=0.0, seed=0, keep_q=False, watch=False, record_executed=False):
    """One episode: the controller (my retargeted video, or a learned policy) gets the state and answers with
    (move, gripper) ten times a second. noise > 0 nudges every executed move. By default the action stored is the one
    the controller asked for (demos that show how to recover from a nudge); record_executed=True stores the move that
    was really made, which is what practice (RL) should learn from. watch=True plays it in the MuJoCo viewer."""
    rng = np.random.default_rng(seed)
    s = env.reset(can_xy)
    viewer = None
    if watch:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.robot.data)
    S, A, Q = [s], [], [env.robot.pose()] if keep_q else []
    done, info = False, {}
    while not done:
        d, g = controller(s)
        d = np.clip(np.asarray(d, float), -DMAX, DMAX)
        d_run = np.clip(d + rng.normal(0, noise, 3), -DMAX, DMAX) if noise > 0 else d
        A.append(np.r_[d_run if record_executed else d, float(g == CLOSED)])
        s, reward, done, info = env.step(d_run, g, viewer)
        S.append(s)
        if keep_q:
            Q.append(env.robot.pose())
    if viewer is not None:
        import time
        print("   done - close the viewer window to go on", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        viewer.close()
    return {"states": np.array(S), "act": np.array(A), "q": np.array(Q) if keep_q else None,
            "success": bool(info["success"]), "reward": float(info["success"]), "why": info["why"],
            "offset_cm": info["offset_cm"], "tilt_deg": info["tilt_deg"], "hits": info["hits"],
            "seconds": round((len(S) - 1) * STEP_DT, 2)}


# ---------------------------------------------------------------- robot demos on disk (sim/<clip>.npz)
def save_runs(path, runs):
    arrays = {"n_runs": len(runs)}
    for i, r in enumerate(runs):
        arrays.update({f"states_{i}": r["states"], f"act_{i}": r["act"], f"success_{i}": r["success"],
                       f"seconds_{i}": r["seconds"], f"why_{i}": r["why"]})
        if r["q"] is not None:
            arrays[f"q_{i}"] = r["q"]
    np.savez_compressed(path, **arrays)


def load_runs(path):
    d = np.load(path)
    return [{"states": d[f"states_{i}"], "act": d[f"act_{i}"], "success": bool(d[f"success_{i}"]),
             "seconds": float(d[f"seconds_{i}"]), "why": str(d[f"why_{i}"]) if f"why_{i}" in d.files else "",
             "q": d[f"q_{i}"] if f"q_{i}" in d.files else None}
            for i in range(int(d["n_runs"]))]
