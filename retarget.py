#!/usr/bin/env python3
"""
retarget.py: turn each of my videos into robot demonstrations, and find out which of my videos the robot can copy.

From a video, read_demos.py measured two things: where the can went (its 3D path on the table) and when my hand
grasped and released it. The robot copies those: the path tells it where to move, the grasp and release tell it when to
close and open the gripper. It doesn't copy my fingers or wrist, because a two-finger gripper can't use them. It grasps
from the top, as I did.

One thing it doesn't copy exactly is the height. My hand passed just over the desk hub and the glasses case; the robot
can't feel them, so wherever my path goes over an object it carries the can at least 2 cm above it.

Each copy runs in the physics twin and gets the same reward as any other episode (twin.py). Copy 0 is clean: if it
fails, my motion doesn't fit the robot's body at that spot (the transfer map). Copies 1-3 get small random pushes, so
the demos also show how to get back on track (as in DART).

    python retarget.py                          # every usable video in data/splits.json, about 3 minutes
    python retarget.py --clips IMG_8682         # just one
    python retarget.py --view --clips IMG_8682  # watch it live in the MuJoCo viewer
    python retarget.py --video                  # also save my video | the robot side by side (sim/<clip>.mp4)
    python retarget.py --base 0.35 -0.10        # the same robot standing at the right side of the table (sim_base/)
Writes sim/<clip>.npz (the robot demos) and sim/replay_summary.csv (the transfer map).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import twin


SAFETY = 0.02          # the can's bottom passes at least 2 cm above anything on the table


def safe_heights(carry_xy, start_xy, lead=0.04):
    """The lowest the can's bottom may go at each point of the carry: SAFETY above any object it passes over, from
    `lead` metres before the object to `lead` metres after it. Objects right next to the start only count once the can
    is 3 cm away (it is lifted straight off first, so the robot doesn't climb over things it is moving away from)."""
    near_start = set(twin.objects_under(start_xy[0], start_xy[1], margin=twin.CAN_RADIUS + 0.01))
    need = np.zeros(len(carry_xy))
    for i, p in enumerate(carry_xy):
        lifted_off = np.linalg.norm(p - start_xy) >= 0.03
        for name in twin.objects_under(p[0], p[1], margin=twin.CAN_RADIUS + SAFETY):
            if lifted_off or name not in near_start:
                need[i] = max(need[i], 2 * twin.OBSTACLES[name][1][2] + SAFETY)
    along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(carry_xy, axis=0), axis=1))]
    return np.array([need[np.abs(along - a) <= lead].max() for a in along])


class Expert:
    """My video as a controller. It reacts to the state at every decision, so it also works from a nudged position,
    from another start spot (start=...) and inside the world model's imagination."""

    def __init__(self, demo, slow=1.5, start=None):
        fps, xyz, ev = demo["fps"], np.array(demo["frames"]["can_xyz"], float), demo["events"]
        s0, end = np.array(demo["can"]["start"], float), np.array(demo["can"]["end"], float)
        lift, down = ev.get("lift_frame"), ev.get("set_down_frame")
        # the part of my video between lifting the can and setting it down: that's what the robot carries along
        carry = xyz[lift:down + 1] if lift is not None and down is not None and down > lift + 2 else np.array([s0, end])
        carry = np.vstack([s0, carry, [end[0], end[1], 0.0]])
        carry[:, 2] = np.maximum(carry[:, 2], 0.0)
        carry[:, 2] += np.sin(np.linspace(0, np.pi, len(carry))) * max(0.0, 0.04 - carry[:, 2].max())  # lift >= 4 cm
        if start is not None:            # adapt my path to another start spot: the change fades out along the carry
            w = np.linspace(1, 0, len(carry))[:, None]
            carry[:, :2] += w * (np.asarray(start, float)[:2] - s0[:2])
            s0 = np.array([start[0], start[1], 0.0])
        safe = safe_heights(carry[:, :2], s0[:2])              # the safety margin over objects
        safe[-1] = 0.0                                          # (the last point is the coaster itself)
        carry[:, 2] = np.maximum(carry[:, 2], safe)
        n = max(2, int(round((len(carry) - 1) / fps * slow / twin.STEP_DT)))   # my timing, slowed down
        u = np.linspace(0, len(carry) - 1, n)
        path = np.array([np.interp(u, np.arange(len(carry)), carry[:, i]) for i in range(3)]).T
        dense = [path[0]]                                       # never faster than the robot's speed limit
        for p in path[1:]:
            k = int(np.ceil(np.linalg.norm(p - dense[-1]) / (0.9 * twin.DMAX)))
            dense += [dense[-1] + (p - dense[-1]) * (j + 1) / k for j in range(k)] if k > 1 else [p]
        dense[0] = dense[0] + [0, 0, 0.02]                       # first lift straight up
        self.path = np.array(dense)
        self.end = np.array([end[0], end[1], twin.COASTER_THICKNESS + 0.004])   # 4 mm above the coaster, then let go
        self.start = s0
        self.k, self.wait, self.done, self.offset, self.regrasp = 0, 0, False, None, False

    def act(self, s):
        tcp, can, closed, width = s[:3], s[3:6], s[6] > 0.5, s[7]
        base = can - [0, 0, twin.CAN_HEIGHT / 2]
        zero = np.zeros(3)
        dmax = twin.DMAX

        def toward(p, vmax=dmax):
            d = np.asarray(p, float) - tcp
            n = np.linalg.norm(d)
            return d if n <= vmax else d * vmax / n

        if closed:
            if width > 0.07:                                    # fingers still closing
                self.wait += 1
                return (zero, twin.CLOSED) if self.wait < 8 else (np.array([0, 0, dmax]), twin.OPEN)
            if width < 0.02 or np.linalg.norm(can[:2] - tcp[:2]) > 0.02:   # closed on nothing: open, try again
                return np.array([0, 0, dmax]), twin.OPEN
            self.wait = 0
            if self.offset is None:                             # just grasped: remember how it sits in the hand,
                self.offset = tcp - base                        # and carry on from the nearest point of my path
                self.k = int(np.argmin(np.linalg.norm(self.path[:, :2] - base[:2], axis=1))) if self.regrasp else 0
                self.regrasp = True
            if self.k < len(self.path):                         # carry it along my path
                if np.linalg.norm(base - self.path[self.k]) < 0.02:
                    self.k += 1
                return toward(self.path[min(self.k, len(self.path) - 1)] + self.offset), twin.CLOSED
            if np.linalg.norm(base - self.end) < 0.006:         # in place: let go
                self.done = True
                return zero, twin.OPEN
            return toward(self.end + self.offset, 0.01), twin.CLOSED
        self.offset = None
        self.wait = 0
        if self.done:                                           # let go: wait for the fingers, then move away
            if width < 0.075:
                return zero, twin.OPEN
            return toward([tcp[0], tcp[1], base[2] + twin.CAN_HEIGHT + 0.12]), twin.OPEN
        if width < 0.075:                                       # fingers still opening
            return zero, twin.OPEN
        grasp = base + [0, 0, twin.GRASP_Z]
        above = grasp + [0, 0, twin.APPROACH_UP]
        off = np.linalg.norm(tcp[:2] - grasp[:2])
        if off > 0.008:                                         # get above the can, staying high
            if tcp[2] < above[2] - 0.03 and off > 0.03:
                return toward([tcp[0], tcp[1], above[2]]), twin.OPEN
            return toward(above if off > 0.03 else [grasp[0], grasp[1], tcp[2]]), twin.OPEN
        if tcp[2] - grasp[2] > 0.005:                           # come down around it, gently
            return toward(grasp, 0.008), twin.OPEN
        return zero, twin.CLOSED                                # close


def convert(demo, env, slow=1.5, copies=4, noise=0.008, seed=0, watch=False):
    """My video -> robot demos: copy 0 clean, the others with random pushes of `noise` metres per decision."""
    runs = []
    for c in range(copies):
        runs.append(twin.rollout(env, Expert(demo, slow).act, demo["can"]["start"][:2], noise=0.0 if c == 0 else noise,
                                 seed=seed + c, keep_q=(c == 0), watch=watch and c == 0))
    return runs


def camera_info(demo):
    calib = json.load(open("calibration.json"))
    return {"rvec": demo["camera"]["rvec"], "tvec": demo["camera"]["tvec"], "fy": calib["fy"],
            "image_h": calib["image_height"]}


def main():
    p = argparse.ArgumentParser(description="Copy my videos onto the simulated Panda.")
    p.add_argument("--clips", nargs="*", help="clip names (default: every usable clip in data/splits.json)")
    p.add_argument("--out", default="out", help="folder with read_demos.py results (default out)")
    p.add_argument("--slow", type=float, default=1.5, help="the robot moves this many times slower than me")
    p.add_argument("--copies", type=int, default=4, help="robot copies per video: 1 clean + pushed ones")
    p.add_argument("--noise", type=float, default=0.008, help="size of the pushes, metres per decision")
    p.add_argument("--view", action="store_true", help="watch each clean copy live in the MuJoCo viewer")
    p.add_argument("--video", action="store_true", help="save my video | the robot, side by side, per clip")
    p.add_argument("--base", nargs=2, type=float, metavar=("X", "Y"), help="stand the robot elsewhere (writes sim_base/)")
    args = p.parse_args()
    if not os.path.exists(twin.MENAGERIE):
        sys.exit("Panda model not found. Run:\n  git clone --depth 1 --filter=blob:none --sparse "
                 "https://github.com/google-deepmind/mujoco_menagerie.git\n"
                 "  cd mujoco_menagerie && git sparse-checkout set franka_emika_panda")
    if args.clips:
        clips = [os.path.splitext(c)[0] for c in args.clips]
    else:
        s = json.load(open("data/splits.json"))
        clips = sorted(s["start"] + s["exam"] + s["pile"])
    folder = "sim"
    if args.base:
        twin.set_base(*args.base)
        folder = "sim_base"
    os.makedirs(folder, exist_ok=True)
    rows, env = [], None
    for k, clip in enumerate(clips, 1):
        demo = json.load(open(os.path.join(args.out, clip, "demo.json")))
        if env is None:
            env = twin.TableEnv(twin.build_scene(demo["coaster"], camera=camera_info(demo)), demo["coaster"])
        seed = int("".join(ch for ch in clip if ch.isdigit()) or k)   # the same pushes whichever clips you pick
        runs = convert(demo, env, args.slow, args.copies, args.noise, seed=seed, watch=args.view)
        twin.save_runs(os.path.join(folder, clip + ".npz"), runs)
        r = runs[0]
        pushed_ok = sum(x["success"] for x in runs[1:])
        print(f"[{k}/{len(clips)}] {clip}: {r['why']} | can {r['offset_cm']} cm from the coaster centre, tilt "
              f"{r['tilt_deg']} deg | {r['seconds']:.1f} s | pushed copies OK: {pushed_ok}/{len(runs) - 1}", flush=True)
        rows.append({"clip": clip, "success": r["success"], "result": r["why"], "offset_cm": r["offset_cm"],
                     "tilt_deg": r["tilt_deg"], "seconds": r["seconds"], "pushed_copies_ok": pushed_ok})
        if args.video:
            import camera
            frames = camera.side_by_side_frames(clip, demo, env.model, r, "my video (phone)",
                                                "the Panda copying it (MuJoCo)")
            camera.write_video(frames, os.path.join(folder, clip + ".mp4"))
    summary = os.path.join(folder, "replay_summary.csv" if not args.clips else "replay_summary_selected.csv")
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nThe robot copied {sum(r['success'] for r in rows)}/{len(rows)} of my videos. Details in {summary}")
    if args.video:
        print(f"Side-by-side videos are in {folder}/, e.g.  xdg-open {folder}/{rows[0]['clip']}.mp4")
    sys.stdout.flush()
    if args.view:
        os._exit(0)          # skip the window library's shutdown, which can crash on Wayland


if __name__ == "__main__":
    main()
