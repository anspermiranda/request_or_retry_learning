#!/usr/bin/env python3
"""
camera.py: pictures of the twin.

1. The robot's own eye. The simulator knows exactly where the can is, but a real robot has to find it. find_can()
   renders the 'eye' camera (colour and depth), keeps the light-blue pixels, turns them into 3D points and fits a
   circle with the can's radius: where the can is, from one picture, like SAM 2 found it in my videos. with_eye()
   gives a controller a new picture at every decision instead of the simulator's answer.
2. Drawing: a recorded moment of an episode, paths drawn into the 3D scene, captions, and my video next to the robot.
"""
import os

import cv2
import mujoco
import numpy as np

import twin

YELLOW = (1.0, 0.82, 0.0, 1.0)      # my path, from the video
GREEN = (0.15, 0.85, 0.30, 1.0)     # the robot's path


# ---------------------------------------------------------------- the robot's eye: find the can
def find_can(model, data, renderer, cam="eye", min_pixels=30):
    """Where is the can? One colour + depth picture from the robot's camera. Returns (centre xyz, pixels) or
    (None, 0). The camera sees the side of the can facing it: 3D points on a circle of known radius, so a circle fit
    gives the centre on the table, and the lowest visible points give the bottom of the can."""
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[..., 0] >= 88) & (hsv[..., 0] <= 108) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 40)
    v, u = np.nonzero(mask)
    if len(u) < min_pixels:
        return None, 0
    h, w = mask.shape
    cid = model.camera(cam).id
    f = 0.5 * h / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    d = depth[v, u]
    pts = np.stack([(u + 0.5 - w / 2) * d / f, -(v + 0.5 - h / 2) * d / f, -d], 1)   # camera frame (looks along -z)
    pts = pts @ data.cam_xmat[cid].reshape(3, 3).T + data.cam_xpos[cid]                # table frame
    xy = pts[:, :2]
    away = xy.mean(0) - data.cam_xpos[cid][:2]          # start a little behind the visible side, then fit
    c = xy.mean(0) + away / (np.linalg.norm(away) + 1e-9) * twin.CAN_RADIUS * 2 / np.pi
    for _ in range(10):
        diff = xy - c
        n = np.linalg.norm(diff, axis=1) + 1e-9
        step = np.linalg.lstsq(-diff / n[:, None], -(n - twin.CAN_RADIUS), rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-5:
            break
    bottom = np.percentile(pts[:, 2], 2)
    return np.array([c[0], c[1], bottom + twin.CAN_HEIGHT / 2]), len(u)


def with_eye(controller, env, renderer, errors=None):
    """Wrap a controller so it sees the can only through the robot's camera: a new picture at every decision (10 a
    second). Its own gripper position and finger opening come from its joints, as on a real arm. If the can is out
    of sight, it keeps its last estimate. errors: a list that collects how far off each estimate was (metres)."""
    last = [None]

    def act(s):
        found, _ = find_can(env.model, env.robot.data, renderer)
        if found is not None:
            last[0] = found
        if last[0] is None:                       # hasn't seen the can yet: stay still
            return np.zeros(3), twin.OPEN
        if errors is not None:
            errors.append(float(np.linalg.norm(last[0] - s[3:6])))
        seen = np.array(s, float).copy()
        seen[3:6] = last[0]
        return controller(seen)
    return act


# ---------------------------------------------------------------- drawing
def add_path(scene, pts, rgba, radius=0.006, every=1):
    """Draw a 3D path into a MuJoCo scene as a chain of thin capsules."""
    pts = np.asarray(pts, float)[::every]
    for a, b in zip(pts[:-1], pts[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        if np.linalg.norm(b - a) < 1e-4:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        scene.ngeom += 1


def render_pose(model, renderer, data, q, camera="overview", paths=()):
    """Draw one recorded moment (joint angles + can pose from Robot.pose()), with optional paths [(points, rgba)]."""
    arm = [model.jnt_qposadr[model.joint(f"panda_joint{i}").id] for i in range(1, 8)]
    arm += [model.jnt_qposadr[model.joint(f"panda_finger_joint{i}").id] for i in (1, 2)]
    can = model.jnt_qposadr[model.joint("can_joint").id]
    data.qpos[arm], data.qpos[can:can + 7] = q[:9], q[9:16]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    for pts, rgba in paths:
        add_path(renderer.scene, pts, rgba)
    return cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)


def caption(img, text, org=(20, 40), scale=0.9, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def my_path(demo):
    """The can's centre along my video (read_demos.py stores the can's base)."""
    return np.array(demo["frames"]["can_xyz"], float) + [0, 0, twin.CAN_HEIGHT / 2]


def phase_marks(run):
    """Robot time (s) of: start, grasp, lift, let go, just after, end."""
    closed = run["act"][:, 3] > 0.5
    lifted = run["states"][:-1, 5] > twin.CAN_HEIGHT / 2 + 0.01
    g = int(np.argmax(closed)) if closed.any() else 0
    lft = int(np.argmax(lifted)) if lifted.any() else g
    rel = g + int(np.argmax(~closed[g:])) if (~closed[g:]).any() else len(closed) - 1
    return np.maximum.accumulate(np.array([0, g, max(g, lft), rel, rel + 1, len(closed)], float)) * twin.STEP_DT


def read_video(clip, size=(960, 540)):
    """My video for the left side: the annotated one from read_demos.py (the can's mask, my hand's 21 points, the
    can's path) if it's there, otherwise the original. Same frames either way."""
    annotated = os.path.join("out", clip, "annotated.mp4")
    use_annotated = os.path.exists(annotated)
    cap = cv2.VideoCapture(annotated if use_annotated else os.path.join("data", "demos", clip + ".MOV"))
    frames = []
    ok, f = cap.read()
    while ok:
        if use_annotated:
            f = f[:720, :1280]          # the camera view, without the bird's-eye panel and the timeline under it
        frames.append(cv2.resize(f, size))
        ok, f = cap.read()
    return frames


def side_by_side_frames(clip, demo, model, run, left_text, right_text, show_my_path=True, cam="closeup"):
    """Yield frames: my video on the left, the robot's episode on the right, matched phase by phase (reach, grasp,
    carry, let go), with my path in yellow and the robot's can path in green."""
    renderer = mujoco.Renderer(model, height=540, width=960)
    data = mujoco.MjData(model)
    frames = read_video(clip)
    ev = demo["events"]
    n_real = len(frames)
    real_marks = None
    if frames:
        real_marks = np.maximum.accumulate(np.array([0, ev["grasp_frame"] or 0, ev["lift_frame"] or 0,
                                                     ev["set_down_frame"] or n_real - 1, ev["release_frame"] or n_real - 1,
                                                     n_real - 1], float))
    sim_marks = phase_marks(run)
    can_path = np.array([q[9:12] for q in run["q"]])
    for k, q in enumerate(run["q"]):
        paths = [(my_path(demo), YELLOW)] if show_my_path else []
        paths.append((can_path[:k + 1], GREEN))
        sim = render_pose(model, renderer, data, q, camera=cam, paths=paths)
        caption(sim, right_text, scale=0.7)
        if frames:
            real = frames[int(np.clip(np.interp(k * twin.STEP_DT, sim_marks, real_marks), 0, n_real - 1))].copy()
        else:
            real = np.full_like(sim, 40)
            caption(real, f"{clip}.MOV not found in data/demos", (20, 280), 0.8)
        caption(real, left_text, scale=0.7)
        yield np.hstack([real, sim])
    renderer.close()


def write_video(frames, path, fps=30, repeat=3):
    """Each robot decision lasts 0.1 s, so every frame is written `repeat` times for a 30 fps video."""
    writer = None
    for f in frames:
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
        for _ in range(repeat):
            writer.write(f)
    if writer is not None:
        writer.release()
