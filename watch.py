#!/usr/bin/env python3
"""
watch.py: watch the robot. My video on the left, the robot in the twin on the right. Yellow is the can's path in my
video, green is the can's path when the robot does it.

    python watch.py IMG_8694              # the final robot (results/policy_ladder.pkl), with the can where it
                                          # starts in this video
    python watch.py IMG_8682 --copy       # the robot copying my video directly (retargeting, no learning)
    python watch.py IMG_8694 --eye        # the robot sees the can only through its own camera
    python watch.py --exam                # the 7 exam spots one after the other (videos it never saw)
    python watch.py --train               # the 7 spots it started learning from
    python watch.py IMG_8694 --viewer     # first play it live in the MuJoCo window, with both paths drawn
    python watch.py IMG_8694 --save       # also save results/watch_IMG_8694.mp4
    python watch.py IMG_8694 --policy results/policy_all_videos.pkl
Keys in the window: space = pause, q = quit. Add --no-window on a machine without a screen (then it only saves).
"""
import argparse
import json
import os
import pickle

import cv2
import mujoco
import numpy as np

import camera
import twin
from retarget import Expert, camera_info


def make_controller(args, demo, policy, env, renderer, seed):
    if args.copy:
        controller, who = Expert(demo, 1.5).act, "the Panda copying my video (retargeting)"
    else:
        controller, who = policy.controller(seed), "the trained robot"
    if args.eye:
        controller = camera.with_eye(controller, env, renderer)
        who += ", seeing the can with its own camera"
    return controller, who


def live(env, controller, spot, mine):
    """Play the episode in the MuJoCo viewer, where you can turn and zoom the view. My path is drawn in yellow and
    the can's path in green as it goes. Close the window to go on."""
    import time
    import mujoco.viewer
    s = env.reset(spot)
    trail = [s[3:6].copy()]
    with mujoco.viewer.launch_passive(env.model, env.robot.data) as viewer:
        done, info = False, {}
        while not done and viewer.is_running():
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                camera.add_path(viewer.user_scn, mine[::3], camera.YELLOW)
                camera.add_path(viewer.user_scn, np.array(trail), camera.GREEN)
            d, g = controller(s)
            s, _, done, info = env.step(d, g, viewer)
            trail.append(s[3:6].copy())
        print(f"   MuJoCo window: {info.get('why', 'stopped')}. Close the window to go on.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def episode(clip, args, env, splits, policy, renderer, seed):
    demo = json.load(open(os.path.join("out", clip, "demo.json")))
    spot = np.array(demo["can"]["start"][:2])
    if args.viewer:
        controller, _ = make_controller(args, demo, policy, env, renderer, seed)
        live(env, controller, spot, camera.my_path(demo))
    controller, who = make_controller(args, demo, policy, env, renderer, seed)   # the same episode again, recorded
    run = twin.rollout(env, controller, spot, keep_q=True, seed=seed)
    group = next((g for g in ("start", "exam", "pile") if clip in splits.get(g, [])), "")
    note = {"start": "it learned from this video", "exam": "an exam spot: it never saw this video",
            "pile": "this video was in the pile"}.get(group, "")
    left = f"my video {clip}" + (f" ({note})" if note and not args.copy else "")
    right = f"{who}: {'success' if run['success'] else run['why']}"
    print(f"{clip}: {run['why']} | {run['seconds']:.1f} s" + (f" | {note}" if note else ""))
    return camera.side_by_side_frames(clip, demo, env.model, run, left, right)


def main():
    p = argparse.ArgumentParser(description="My video next to the robot doing the same task.")
    p.add_argument("clips", nargs="*")
    p.add_argument("--exam", action="store_true", help="all exam clips")
    p.add_argument("--train", action="store_true", help="the 7 start clips it learned from first")
    p.add_argument("--copy", action="store_true", help="show the direct copy of my video instead of the trained robot")
    p.add_argument("--eye", action="store_true", help="the robot sees the can only through its own camera")
    p.add_argument("--policy", default="results/policy_ladder.pkl")
    p.add_argument("--viewer", action="store_true", help="also play it live in the MuJoCo viewer")
    p.add_argument("--save", action="store_true", help="save results/watch_<clip>.mp4")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    splits = json.load(open("data/splits.json"))
    clips = list(args.clips) + (list(splits["exam"]) if args.exam else []) + (list(splits["start"]) if args.train else [])
    if not clips:
        p.error("name a clip (e.g. IMG_8694) or use --exam or --train")
    first = json.load(open(os.path.join("out", splits["start"][0], "demo.json")))
    model = twin.build_scene(first["coaster"], camera=camera_info(first))
    env = twin.TableEnv(model, first["coaster"])
    policy = None if args.copy else pickle.load(open(args.policy, "rb"))
    renderer = mujoco.Renderer(model, 480, 640) if args.eye else None
    os.makedirs("results", exist_ok=True)
    for clip in clips:
        clip = os.path.splitext(clip)[0]
        frames = episode(clip, args, env, splits, policy, renderer, args.seed)
        if frames is None:
            continue
        path = os.path.join("results", f"watch_{clip}{'_copy' if args.copy else ''}{'_eye' if args.eye else ''}.mp4")
        shown = []                                            # small copies for the window, full size to the file
        writer = None
        for f in frames:
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (f.shape[1], f.shape[0]))
                for _ in range(3):
                    writer.write(f)
            if not args.no_window:
                shown.append(cv2.resize(f, (1600, 450)))
        if writer is not None:
            writer.release()
            print(f"   saved {path}")
        if args.no_window:
            continue
        frames = shown
        k, paused = 0, False
        while k < len(frames):
            cv2.imshow("my video | the robot   (space = pause, q = quit)", frames[k])
            key = cv2.waitKey(100) & 0xFF                     # one robot decision = 0.1 s
            if key == ord("q"):
                cv2.destroyAllWindows()
                if args.viewer:
                    os._exit(0)
                return
            if key == ord(" "):
                paused = not paused
            if not paused:
                k += 1
        cv2.waitKey(800)
    if not args.no_window:
        cv2.destroyAllWindows()
    if args.viewer:
        os._exit(0)          # skip the window library's shutdown, which can crash on Wayland after the viewer closes


if __name__ == "__main__":
    main()
