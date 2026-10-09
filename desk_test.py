#!/usr/bin/env python3
"""
desk_test.py: does the robot work anywhere on the desk, or only where my videos started?

The exam uses the start spots of 7 videos the robot never saw. This test goes further: 100 spots drawn at random over
the area where my videos start, skipping the coaster and anything on or touching another object. One try at each.
Nothing here is used for training. Three versions:
  1. my robot (imagine -> retry -> request, run 1), told where the can is by the simulator
  2. my robot, finding the can with its own camera at every decision (camera.py), as a real robot would
  3. the policy trained on all 33 videos at once, told where the can is
The coaster never moves in my videos, so its position is known in all three.

    python desk_test.py            # about 10 minutes, after experiment.py
Writes results/desk_test.json, results/desk_map.png (version 2) and results/desk_map_compare.png (all three).
A green dot = the can ended upright on the coaster with nothing touched; red = it didn't.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle

import cv2
import numpy as np

import agent as ag
import twin
from experiment import World, label, table_image

GREEN, RED, WHITE = (60, 180, 60), (40, 40, 230), (255, 255, 255)


def random_spots(world, n, seed):
    """n spots in the box where my videos start, at least 12 cm from the coaster's centre and 4 cm clear of every
    object."""
    starts = np.array([world.start_of(c) for c in world.trainable + world.exam_clips])
    lo, hi = starts.min(0), starts.max(0)
    rng = np.random.default_rng(seed)
    spots = []
    while len(spots) < n:
        p = lo + rng.random(2) * (hi - lo)
        if np.linalg.norm(p - world.coaster_xy) >= 0.12 and twin.support_height(p[0], p[1], margin=0.04) == 0.0:
            spots.append(p)
    return spots


def videos_learned_from(run, world, max_fix):
    """The videos the ladder robot ended up with: its start videos plus the ones it asked for. Newer runs of
    experiment.py save this list; for older ones it is replayed from the log, exactly as agent.request() chose them
    (the weakest spots first, and for each 'request' the nearest video left in the pile)."""
    if run.get("owned"):
        return list(run["owned"])
    owned = list(run["start"])
    pile = [c for c in world.trainable if c not in run["start"]]
    for h in run["history"]:
        if "decisions" not in h:
            continue
        rates = h["practice_rates"]
        for i in [i for i in np.argsort(rates, kind="stable") if rates[i] < 1.0][:max_fix]:
            if h["decisions"][i] == "request":
                free = [c for c in pile if c not in owned]
                owned.append(min(free, key=lambda c: ag.dist(world.start_of(c), world.probe_spots[i])))
    return owned


def with_camera(world, policy, spots, seeds):
    """The same episodes, but the robot finds the can with its own camera at every decision. Runs in this process
    (one renderer), after the worker pool is closed."""
    import mujoco
    import camera
    model = twin.build_scene(world.coaster)
    renderer = mujoco.Renderer(model, 480, 640)
    env = twin.TableEnv(model, world.coaster)
    eps, errors = [], []
    for k, (s, sd) in enumerate(zip(spots, seeds)):
        eps.append(twin.rollout(env, camera.with_eye(policy.controller(sd), env, renderer, errors), s, seed=sd))
        if (k + 1) % 20 == 0:
            print(f"   with its own camera: {k + 1}/{len(spots)}", flush=True)
    renderer.close()
    return eps, 100 * float(np.mean(errors)) if errors else None


def panel(world, spots, ok, learned_from, title, note):
    """The desk from above: the random spots in green or red, where its videos start (white rings), exam spots (X)."""
    img, px = table_image("out")
    if img is None:
        return None
    for c in learned_from:
        cv2.circle(img, px(*world.start_of(c)), 13, WHITE, 2)
    for e in world.exam_spots:
        cv2.drawMarker(img, px(*e), WHITE, cv2.MARKER_TILTED_CROSS, 22, 3)
    for s, good in zip(spots, ok):
        cv2.circle(img, px(*s), 8, GREEN if good else RED, -1)
        cv2.circle(img, px(*s), 8, (0, 0, 0), 1)
    label(img, title, (20, 38), 0.75)
    label(img, note, (20, 70), 0.55)
    label(img, "green = can on the coaster, red = failed, white ring = a video it learned from starts here, "
               "X = exam spot", (20, img.shape[0] - 20), 0.5)
    return img


def summary(eps):
    whys = {}
    for e in eps:
        if not e["success"]:
            whys[e["why"]] = whys.get(e["why"], 0) + 1
    return float(np.mean([e["success"] for e in eps])), dict(sorted(whys.items(), key=lambda x: -x[1]))


def main():
    p = argparse.ArgumentParser(description="The final robot at random spots all over the desk.")
    p.add_argument("--spots", type=int, default=100)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    p.add_argument("--no-camera", action="store_true", help="skip version 2 (the slowest)")
    a = p.parse_args()
    world = World()
    spots = random_spots(world, a.spots, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    seeds = [int(rng.integers(1 << 30)) for _ in spots]
    run1 = json.load(open("results/ladder_run1.json"))
    settings = json.load(open("results/summary.json"))["settings"]
    owned = videos_learned_from(run1, world, settings.get("max_fix", 4))
    if len(owned) != run1["history"][-1]["videos"]:          # can't happen unless the log is from other code
        print("   (couldn't replay which videos it asked for; showing its start videos only)")
        owned = list(run1["start"])
    asked = [c for c in owned if c not in run1["start"]]
    ladder_note = (f"learned from {len(owned)} videos: its {len(run1['start'])} start videos"
                   + (f" + {', '.join(asked)}, which it asked for" if asked else ""))
    ladder = pickle.load(open("results/policy_ladder.pkl", "rb"))
    every = pickle.load(open("results/policy_all_videos.pkl", "rb"))
    print(f"{len(spots)} random spots on the desk, one try each", flush=True)
    with mp.get_context("fork").Pool(a.workers, initializer=ag.init_worker, initargs=(world.coaster,)) as pool:
        told = {name: pool.map(ag.run_task, [("policy", pol, s, sd, 0.0, False, False) for s, sd in zip(spots, seeds)])
                for name, pol in (("ladder", ladder), ("all_videos", every))}
    runs = [("ladder", "my robot, told where the can is", told["ladder"], owned, ladder_note, None),
            ("all_videos", "all 33 videos at once, told where the can is", told["all_videos"], world.trainable,
             f"learned from all {len(world.trainable)} videos", None)]
    if not a.no_camera:
        eps, err = with_camera(world, ladder, spots, seeds)
        runs.insert(1, ("ladder_camera", "my robot, finding the can with its own camera", eps, owned, ladder_note, err))
    report = {"spots": [[float(x), float(y)] for x, y in spots], "results": {}}
    panels = {}
    print()
    for key, name, eps, learned_from, note, err in runs:
        rate, whys = summary(eps)
        near = [min(np.linalg.norm(s - world.start_of(c)) for c in learned_from) for s in spots]
        report["results"][key] = {"name": name, "success": rate, "failures": whys,
                                  "ok": [e["success"] for e in eps], "learned_from": list(learned_from),
                                  "cm_to_nearest_video_it_learned_from": [round(100 * d, 1) for d in near]}
        if err is not None:
            report["results"][key]["camera_error_cm"] = err
        print(f"{name}: {100 * rate:.0f}% of {len(spots)}" + (f" (camera {err:.1f} cm off on average)" if err else "")
              + " | failures: " + (", ".join(f"{w} {n}" for w, n in whys.items()) or "none"), flush=True)
        panels[key] = panel(world, spots, [e["success"] for e in eps], learned_from,
                            f"{name}: {100 * rate:.0f}% of {len(spots)} random spots", note)
    json.dump(report, open("results/desk_test.json", "w"), indent=1)
    if all(x is not None for x in panels.values()):
        cv2.imwrite("results/desk_map.png", panels["ladder_camera" if "ladder_camera" in panels else "ladder"])
        cv2.imwrite("results/desk_map_compare.png", np.hstack(list(panels.values())))
        print("Saved results/desk_map.png, results/desk_map_compare.png and results/desk_test.json")
    else:
        print("Saved results/desk_test.json (no bird's-eye picture in out/ to draw the map on)")


if __name__ == "__main__":
    main()
