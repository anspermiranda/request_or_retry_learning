#!/usr/bin/env python3
"""
evaluate.py: a closer look at the final robot (the ladder, run 1: my 7 videos, then 5 rounds of getting help).

  1. The exam, spot by spot: successes out of 5 and what went wrong when it failed.
  2. The same exam, but the robot finds the can with its own camera instead of being told where it is (camera.py).
  3. The 7 spots it learned from: does it still do them, and how closely does it follow my path there? This is the
     only place my path is compared. At exam spots any path to the coaster is fine, as long as nothing is touched.
  4. Every strategy, exam spot by spot (from the experiment's logs), with the 'all videos' upper bound.

    python evaluate.py              # about 5 minutes, after experiment.py
Writes results/evaluation.md and results/evaluation.json.
"""
import csv
import glob
import json
import multiprocessing as mp
import os
import pickle

import numpy as np

import agent as ag
import camera
import twin
from experiment import World


def path_gap_cm(robot_can, demo):
    """How far the robot's can strays from my path: for every point of the robot's carry, the distance to the nearest
    point of my carry, averaged (cm). 'Carry' = while the can is in the air."""
    mine = camera.my_path(demo)
    mine = mine[mine[:, 2] > twin.CAN_HEIGHT / 2 + 0.01]
    ours = robot_can[robot_can[:, 2] > twin.CAN_HEIGHT / 2 + 0.01 + twin.support_height(*robot_can[0, :2])]
    if len(mine) < 2 or len(ours) < 2:
        return None
    d = np.linalg.norm(ours[:, None] - mine[None], axis=2).min(1)
    return 100 * float(d.mean())


def main():
    summary = json.load(open("results/summary.json"))
    rounds, tries = summary["settings"]["rounds"], summary["settings"]["exam_tries"]
    world = World()
    policy = pickle.load(open("results/policy_ladder.pkl", "rb"))
    base = np.array(twin.ROBOT_BASE[:2])
    copied = {}
    if os.path.exists("sim/replay_summary.csv"):
        copied = {r["clip"]: r["success"] == "True" for r in csv.DictReader(open("sim/replay_summary.csv"))}
    report = {}
    lines = ["# The final robot, in detail", "",
             f"Ladder, run 1 (my 7 videos, then {rounds} rounds). The exam is the same {7 * tries} attempts as its last "
             "exam in experiment.py.", ""]

    # 1. the exam with the simulator telling it where the can is, and 3. the spots it learned from
    rng = np.random.default_rng(1000 + rounds)            # exactly the experiment's last exam
    tasks = []
    for spot in world.exam_spots:
        for _ in range(tries):
            tasks.append((np.asarray(spot) + rng.normal(0, 0.01, 2), int(rng.integers(1 << 30))))
    start_clips = world.start_set(0)
    train_tasks = [(world.start_of(c), 11 * k + j) for k, c in enumerate(start_clips) for j in range(3)]
    with mp.get_context("fork").Pool(max(1, min(16, (os.cpu_count() or 2) - 2)), initializer=ag.init_worker,
                                     initargs=(world.coaster,)) as pool:
        eps = pool.map(ag.run_task, [("policy", policy, s, seed, 0.0, True, False) for s, seed in tasks])
        train_eps = pool.map(ag.run_task, [("policy", policy, s, seed, 0.0, True, False) for s, seed in train_tasks])

    lines += ["## 1. Exam, spot by spot", "",
              "| exam video | cm from robot base | robot could copy my video there | success | what went wrong |",
              "|---|---|---|---|---|"]
    per_spot = []
    for k, clip in enumerate(world.exam_clips):
        e = eps[k * tries:(k + 1) * tries]
        wins = sum(x["success"] for x in e)
        whys = sorted({x["why"] for x in e if not x["success"]})
        per_spot.append(wins / tries)
        copy = {True: "yes", False: "no"}.get(copied.get(clip), "?")
        lines.append(f"| {clip} | {100 * np.linalg.norm(world.exam_spots[k] - base):.0f} | {copy} | {wins}/{tries} | "
                     f"{', '.join(whys) or '-'} |")
    score = float(np.mean([x["success"] for x in eps]))
    touched = sum(bool(x["hits"]) for x in eps)
    lines += ["", f"Overall {100 * score:.0f}%. Episodes where the arm or the carried can touched another object: "
                  f"{touched} of {len(eps)} (a touch makes the episode fail).", ""]
    report["exam"] = {"success": score, "per_spot": per_spot, "touched": touched, "episodes": len(eps)}

    # 2. the same exam with the robot's own camera
    import mujoco
    model = twin.build_scene(world.coaster)
    renderer = mujoco.Renderer(model, 480, 640)
    env = twin.TableEnv(model, world.coaster)
    eye_ok, eye_err = [], []
    for spot, seed in tasks:
        ctrl = camera.with_eye(policy.controller(seed), env, renderer, eye_err)
        eye_ok.append(twin.rollout(env, ctrl, spot)["success"])
    renderer.close()
    eye_score = float(np.mean(eye_ok))
    eye_err = 100 * np.array(eye_err)
    lines += ["## 2. The same exam, finding the can with its own camera", "",
              "At every decision the robot takes a colour + depth picture with a camera on the far side of the table, "
              "keeps the light-blue pixels and fits a circle of the can's radius to their 3D points. That estimate "
              "replaces the simulator's answer. Its own gripper position and finger opening come from its joints.",
              "", "| where the can is, according to | exam success | error of that estimate |", "|---|---|---|",
              f"| the simulator | {100 * score:.0f}% | 0 |",
              f"| its own camera | {100 * eye_score:.0f}% | {np.mean(eye_err):.2f} cm on average, 95% of the time under "
              f"{np.percentile(eye_err, 95):.2f} cm |", ""]
    report["own_camera"] = {"success": eye_score, "error_cm_mean": float(np.mean(eye_err)),
                            "error_cm_95": float(np.percentile(eye_err, 95))}

    # 3. the spots it learned from
    lines += ["## 3. The 7 spots it learned from", "",
              "Does it still do them, and how far does its can stray from my path while carrying it? For comparison, "
              "the robot copying my video directly (retarget.py).", "",
              "| my video | success | gap to my path (learned policy) | gap to my path (direct copy) |",
              "|---|---|---|---|"]
    gaps, copy_gaps = [], []
    for k, clip in enumerate(start_clips):
        e = train_eps[3 * k:3 * k + 3]
        g = [path_gap_cm(np.array([q[9:12] for q in x["q"]]), world.demos[clip]) for x in e if x["success"]]
        g = [x for x in g if x is not None]
        c = path_gap_cm(world.videos[clip][0]["states"][:, 3:6], world.demos[clip]) if world.videos[clip][0]["success"] else None
        gaps += g
        copy_gaps += [c] if c is not None else []
        lines.append(f"| {clip} | {sum(x['success'] for x in e)}/3 | {f'{np.mean(g):.1f} cm' if g else '-'} | "
                     f"{f'{c:.1f} cm' if c is not None else '-'} |")
    train_score = float(np.mean([x["success"] for x in train_eps]))
    lines += ["", f"Success at the spots it learned from: {100 * train_score:.0f}%. Average gap to my path: "
                  f"{np.mean(gaps) if gaps else float('nan'):.1f} cm (direct copy: "
                  f"{np.mean(copy_gaps) if copy_gaps else float('nan'):.1f} cm).", ""]
    report["learned_spots"] = {"success": train_score, "path_gap_cm": float(np.mean(gaps)) if gaps else None,
                               "direct_copy_path_gap_cm": float(np.mean(copy_gaps)) if copy_gaps else None}

    # 4. every strategy, spot by spot
    names = {**ag.STRATEGIES}
    runs = {s: [json.load(open(f)) for f in sorted(glob.glob(f"results/{s}_run*.json"))] for s in names}
    runs = {s: r for s, r in runs.items() if r}
    if runs:
        first = next(iter(runs.values()))
        start = np.mean([r["history"][0]["exam_per_spot"] for r in first], 0)
        end = {s: np.mean([r["history"][-1]["exam_per_spot"] for r in rs], 0) for s, rs in runs.items()}
        upper = summary["strategies"].get("all_videos", {}).get("exam_per_spot")
        head = "| exam video | cm from robot base | start (7 videos) | " + " | ".join(names[s] for s in runs)
        head += " | all videos at once |" if upper else " |"
        lines += ["## 4. Every strategy, spot by spot (all runs)", "", head,
                  "|---|---|---|" + "---|" * (len(runs) + bool(upper))]
        for k, clip in enumerate(world.exam_clips):
            row = f"| {clip} | {100 * np.linalg.norm(world.exam_spots[k] - base):.0f} | {100 * start[k]:.0f}% | "
            row += " | ".join(f"{100 * end[s][k]:.0f}%" for s in runs)
            lines.append(row + (f" | {100 * upper[k]:.0f}% |" if upper else " |"))
        row = f"| **all 7** | | **{100 * start.mean():.0f}%** | " + " | ".join(f"**{100 * end[s].mean():.0f}%**" for s in runs)
        lines.append(row + (f" | **{100 * np.mean(upper):.0f}%** |" if upper else " |"))
        near_train = min(world.trainable, key=lambda c: np.linalg.norm(world.start_of(c) - base))
        d_train = 100 * np.linalg.norm(world.start_of(near_train) - base)
        closer = [k for k in range(len(world.exam_clips)) if 100 * np.linalg.norm(world.exam_spots[k] - base) < d_train - 1]
        lines += ["", f"None of the videos the robot can learn from starts closer than {d_train:.0f} cm to its base "
                      f"({near_train})."]
        if closer:
            rest = [k for k in range(len(world.exam_clips)) if k not in closer]
            lines.append(f"Exam spots closer than that: {', '.join(world.exam_clips[k] for k in closer)}. On the other "
                         f"{len(rest)}: start {100 * start[rest].mean():.0f}% -> " +
                         ", ".join(f"{names[s]} {100 * end[s][rest].mean():.0f}%" for s in runs) +
                         (f", all videos {100 * np.mean(np.array(upper)[rest]):.0f}%" if upper else "") + ".")
        report["per_spot"] = {"start": start.tolist(), **{s: end[s].tolist() for s in runs}, "all_videos": upper}
    text = "\n".join(lines) + "\n"
    open("results/evaluation.md", "w").write(text)
    json.dump(report, open("results/evaluation.json", "w"), indent=1)
    print(text)
    print("Saved results/evaluation.md and results/evaluation.json")


if __name__ == "__main__":
    main()
