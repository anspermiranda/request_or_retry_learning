#!/usr/bin/env python3
"""
speed.py: make a standard policy run much faster, and check it still works.

Baseline: the robot's own diffusion policy (results/policy_ladder.pkl, from experiment.py) sampled the standard way,
DDPM with 100 steps: 100 network passes for every decision. Faster versions of the very same network:
  DDIM 10 / DDIM 4 : 10 or 4 deterministic steps instead of 100 (Song et al., 2021). DDIM 4 is what the robot
                     uses during the experiment. It works because the network predicts the clean chunk rather than the
                     noise; predicting the noise, 4 steps were three times less accurate on this data.
  distilled student: a separate small network trained to give in one pass what the 10-step version gives.
For reference: chunked behaviour cloning trained on the same data (one pass of 3 small networks).

Every variant takes the experiment's last exam: the same 7 spots x 10 tries, the same nudges, the same seeds. So the
DDIM 4 row is the robot exactly as it was tested in experiment.py, and must match its final exam.
Speed = the time for one decision on this computer's CPU, one decision at a time, as on a robot.

    python speed.py            # about 5-10 minutes, after experiment.py
    python speed.py --quick    # smoke test
Writes results/speed.json and results/speed.png
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import agent as ag
import twin
from policy import K, MLP, OBS, BCPolicy, chunk_controller


class Student:
    """One network pass, trained to copy the 10-step teacher's decisions (policy distillation)."""

    def __init__(self, teacher, obs, sample_p, seed=0, steps=4000):
        rng = np.random.default_rng(seed)
        targets = np.vstack([teacher.sample(obs[i:i + 512], "ddim", 10, rng) for i in range(0, len(obs), 512)])
        self.coaster_xy = teacher.coaster_xy
        self.net = MLP(OBS, 4 * K, (256, 256), seed).fit(obs, targets, steps=steps, noise=np.r_[[0.0015] * (OBS - 1), 0.002],
                                                        weights=[1, 1, 1, 4] * K, sample_p=sample_p, seed=seed)

    def sample(self, obs, method=None, n=None, rng=None):
        return self.net.predict(obs)


VARIANTS = [("diffusion, DDPM 100 steps (baseline)", "teacher", "ddpm", 100),
            ("diffusion, DDIM 10 steps", "teacher", "ddim", 10),
            ("diffusion, DDIM 4 steps (what the robot uses)", "teacher", "ddim", 4),
            ("distilled student, 1 pass", "student", None, 1),
            ("behaviour cloning, 1 pass x 3 nets", "bc", None, 1)]


class Sampled:
    """A model + sampling method, packaged so a worker can build a controller from it."""

    def __init__(self, model, method, n):
        self.model, self.method, self.n = model, method, n

    def controller(self, seed=0):
        rng = np.random.default_rng(seed)
        return chunk_controller(lambda o: self.model.sample(o, self.method, self.n, rng), self.model.coaster_xy)


def main():
    p = argparse.ArgumentParser(description="Make the policy run faster and check it still works.")
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    summary = json.load(open("results/summary.json"))
    rounds, tries = summary["settings"]["rounds"], summary["settings"]["exam_tries"]
    if a.quick:
        tries = 1
    teacher = pickle.load(open("results/policy_ladder.pkl", "rb"))
    d = np.load("results/final_training_data.npz")
    obs, chunks, sp, spots = d["obs"], d["chunks"], d["sample_p"], d["exam_spots"]
    coaster = json.load(open(os.path.join("out", summary["start_sets"][0][0], "demo.json")))["coaster"]
    print(f"teacher: the robot's final policy | its training data: {len(obs)} decisions | distilling a one-pass "
          f"student...", flush=True)
    t0 = time.time()
    student = Student(teacher, obs, sp, steps=1000 if a.quick else 4000)
    bc = BCPolicy(teacher.coaster_xy, 0).fit_arrays(obs, chunks, sp, 500 if a.quick else BCPolicy.STEPS)
    print(f"   {time.time() - t0:.0f} s", flush=True)
    models = {"teacher": teacher, "student": student, "bc": bc}
    probe = obs[np.random.default_rng(1).choice(len(obs), 200)]
    rng = np.random.default_rng(1000 + rounds)                      # exactly the experiment's last exam
    exam = [(np.asarray(s) + rng.normal(0, 0.01, 2), int(rng.integers(1 << 30))) for s in spots for _ in range(tries)]
    print(f"exam: {len(spots)} unseen spots x {tries} tries = {len(exam)} episodes per variant", flush=True)
    rows = []
    with mp.get_context("fork").Pool(a.workers, initializer=ag.init_worker, initargs=(coaster,)) as pool:
        for name, which, method, n in VARIANTS:
            model = models[which]
            r = np.random.default_rng(0)
            t = time.perf_counter()
            for o in probe:                                       # one decision at a time, as on a robot
                model.sample(o[None], method, n, r)
            ms = 1000 * (time.perf_counter() - t) / len(probe)
            runner = Sampled(model, method, n)
            ok = [e["success"] for e in pool.map(ag.run_task, [("policy", runner, s, seed, 0.0, False, False)
                                                               for s, seed in exam])]
            passes = n if which == "teacher" else (len(bc.nets) if which == "bc" else 1)
            rows.append({"policy": name, "network_passes": passes, "ms_per_decision": ms, "decisions_per_second": 1000 / ms,
                         "exam_success": float(np.mean(ok)), "exam_episodes": len(ok)})
            print(f"   {name:46s} {ms:8.2f} ms per decision ({1000 / ms:7.0f} /s) | exam {100 * np.mean(ok):3.0f}%", flush=True)
    for r in rows:
        r["speedup_vs_baseline"] = rows[0]["ms_per_decision"] / r["ms_per_decision"]
    json.dump(rows, open("results/speed.json", "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    names = [r["policy"].replace(", ", "\n") for r in rows]
    colors = ["tab:gray", "tab:blue", "tab:blue", "tab:green", "tab:orange"]
    ax[0].barh(names, [r["ms_per_decision"] for r in rows], color=colors)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("time per decision (ms, CPU, log scale): lower is faster")
    for i, r in enumerate(rows):
        ax[0].text(r["ms_per_decision"] * 1.1, i, f"{r['speedup_vs_baseline']:.0f}x", va="center")
    ax[1].barh(names, [100 * r["exam_success"] for r in rows], color=colors)
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel(f"exam success (%): {len(spots)} unseen spots x {tries} tries")
    ax[1].set_yticklabels([])
    for a_ in ax:
        a_.invert_yaxis()
        a_.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig("results/speed.png", dpi=130)
    print(f"Saved results/speed.json and results/speed.png | DDIM 4 is {rows[2]['speedup_vs_baseline']:.0f}x and the "
          f"student {rows[3]['speedup_vs_baseline']:.0f}x faster than the baseline")
    if not a.quick and os.path.exists("results/ladder_run1.json"):
        loop = json.load(open("results/ladder_run1.json"))["history"][-1]["exam_success"]
        print(f"Check: experiment.py's last exam for this policy: {100 * loop:.0f}% | DDIM 4 here: "
              f"{100 * rows[2]['exam_success']:.0f}%")


if __name__ == "__main__":
    main()
