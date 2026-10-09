#!/usr/bin/env python3
"""
experiment.py: can the robot learn the task from a few of my videos, and does the ladder get it there with less help?

  start  : 7 of my videos. Run 1 starts from the 7 I picked at random; runs 2 and 3 start from 7 drawn at random from
           the 33 videos that aren't in the exam, so the result doesn't hang on one pick.
  exam   : 7 spots, the start spots of 7 other videos (spread over the table). The robot never trains there and never
           sees those videos. 10 tries per spot, the can nudged by about 1 cm each time. Only measured, never used to
           decide anything.
  pile   : every other video. The robot only gets one by asking for it (or at random, for the 'random' baseline).
  rounds : 5. Each round the agent (post-)trains its policy, takes the exam, tests itself at practice spots and fixes
           its weak spots with its kind of help (agent.py).
  all videos : the same policy trained on all 33 non-exam videos at once, as an upper bound.

    python experiment.py --quick        # smoke test, a few minutes
    python experiment.py                # the full experiment, about 40 minutes on a 16-thread laptop
Writes results/: summary.json, learning_curves.png, <strategy>_map_round<r>.png, world_model.png, exam_attempts.mp4,
imagined_vs_real.mp4, the final policies (policy_ladder.pkl, policy_all_videos.pkl) and their training data.
"""
import argparse
import datetime
import glob
import json
import multiprocessing as mp
import os
import pickle
import struct
import time

import cv2
import numpy as np

import agent as ag
import twin
from policy import POLICIES, chunks, make_obs, switch_weights
from world_model import WorldModel, trust_at_spot

COLORS = {"ok": (60, 180, 60), "imagine": (220, 120, 40), "retry": (0, 170, 255), "request": (40, 40, 230),
          "film here": (200, 40, 200)}
STYLE = {"ladder": "tab:blue", "request_only": "tab:red", "retry_only": "tab:orange", "random": "tab:gray"}


# ---------------------------------------------------------------- what every agent shares
class World:
    """My videos as robot demos, the splits, the coaster, the practice spots and the exam spots."""

    def __init__(self, out="out"):
        splits = json.load(open("data/splits.json"))
        self.splits = splits
        names = splits["start"] + splits["pile"] + splits["exam"]
        self.demos = {c: json.load(open(os.path.join(out, c, "demo.json"))) for c in names}
        self.coaster = self.demos[splits["start"][0]]["coaster"]
        self.coaster_xy = np.array(self.coaster["center"], float)
        self.videos = {}
        for c in splits["start"] + splits["pile"]:
            path = os.path.join("sim", c + ".npz")
            if not os.path.exists(path):
                print(f"   {c}: no robot demos yet (run retarget.py first), skipped")
                continue
            runs = twin.load_runs(path)
            for r in runs:
                r.update(clip=c, source="video", spot=r["states"][0, 3:5].copy())
            self.videos[c] = runs
        self.trainable = [c for c in splits["start"] + splits["pile"] if c in self.videos]
        self.exam_clips = list(splits["exam"])
        self.exam_spots = [np.array(self.demos[c]["can"]["start"][:2]) for c in self.exam_clips]
        self.home_tcp = twin.Robot(twin.build_scene(self.coaster), [0.3, 0.3]).tcp()
        self.probe_spots = self.practice_grid()
        self.my_minutes_per_video = my_minutes_per_video()

    def start_of(self, clip):
        return self.videos[clip][0]["spot"] if clip in self.videos else np.array(self.demos[clip]["can"]["start"][:2])

    def practice_grid(self, nx=6, ny=5):
        """Practice spots: a grid over the area where my videos start, minus the coaster, the exam spots (with a 4 cm
        margin) and the tops of other objects."""
        starts = np.array([self.start_of(c) for c in self.trainable])
        spots = []
        for x in np.linspace(starts[:, 0].min(), starts[:, 0].max(), nx):
            for y in np.linspace(starts[:, 1].min(), starts[:, 1].max(), ny):
                p = np.array([x, y])
                if np.linalg.norm(p - self.coaster_xy) < 0.12 or any(np.linalg.norm(p - e) < 0.04 for e in self.exam_spots):
                    continue
                if twin.support_height(x, y, margin=0.04) == 0.0:
                    spots.append(p)
        return spots

    def start_set(self, run):
        """Run 0: my 7. Other runs: 7 drawn at random from the videos that aren't in the exam."""
        n = len(self.splits["start"])
        if run == 0:
            return [c for c in self.splits["start"] if c in self.videos]
        return sorted(np.random.default_rng(run).choice(self.trainable, n, replace=False).tolist())


def recording_time(path):
    """When a phone video was recorded (QuickTime 'mvhd' header), without any extra tools."""
    with open(path, "rb") as f:
        head = f.read(8_000_000)
        i = head.find(b"mvhd")
        if i < 0:
            f.seek(max(0, os.path.getsize(path) - 8_000_000))
            head = f.read()
            i = head.find(b"mvhd")
    if i < 0:
        return None
    v = head[i + 4]
    t = struct.unpack(">Q", head[i + 8:i + 16])[0] if v == 1 else struct.unpack(">I", head[i + 8:i + 12])[0]
    return datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=t)


def my_minutes_per_video(folder="data/demos", default=0.75):
    """What one video really cost me: the typical gap between two recordings in my session (resets included)."""
    stamps = []
    for f in sorted(glob.glob(os.path.join(folder, "*.MOV"))):
        try:
            stamps.append((int("".join(c for c in os.path.basename(f) if c.isdigit())), recording_time(f)))
        except (ValueError, OSError):
            pass
    stamps = [(n, t) for n, t in stamps if t is not None]
    gaps = [(t2 - t1).total_seconds() / (n2 - n1) for (n1, t1), (n2, t2) in zip(stamps, stamps[1:])
            if n2 > n1 and 5 < (t2 - t1).total_seconds() / (n2 - n1) < 600]
    return float(np.median(gaps)) / 60 if gaps else default


def exam(pool, policy, world, r, tries):
    groups = ag.attempts(pool, policy, world.exam_spots, tries, 1000 + r, jitter=0.01)
    return groups, float(np.mean([e["success"] for g in groups for e in g]))


# ---------------------------------------------------------------- one run of one strategy
def run(strategy, world, cfg, pool, seed, log_map):
    agent = ag.Agent(strategy, world, world.start_set(seed), seed, cfg, pool)
    history = []
    for r in range(cfg.rounds + 1):
        t0 = time.time()
        agent.learn(r)
        groups, score = exam(pool, agent.policy, world, r, cfg.exam_tries)
        entry = {"round": r, "exam_success": score, "exam_per_spot": [float(np.mean([e["success"] for e in g])) for g in groups],
                 "exam_failures": [e["why"] for g in groups for e in g if not e["success"]], **agent.bill()}
        if r == cfg.rounds:
            history.append(entry)
            print(f"   [{strategy}] round {r}: exam {100 * score:.0f}% | {agent.videos} videos | "
                  f"{agent.robot_s / 60:.0f} robot-min | {agent.imagined} imagined", flush=True)
            break
        rates, n_skip = agent.self_test(r)
        decisions = agent.improve(r)
        if log_map is not None:
            log_map(strategy, r, rates, decisions, agent.trust)
        entry.update(practice_success=float(np.mean(rates)), practice_rates=rates, decisions=decisions,
                     trust=agent.trust, tests_imagined=n_skip, seconds=time.time() - t0)
        history.append(entry)
        print(f"   [{strategy}] round {r}: exam {100 * score:.0f}% | practice spots {100 * np.mean(rates):.0f}% | "
              f"{agent.videos} videos | {agent.robot_s / 60:.0f} robot-min | {agent.imagined} imagined | "
              f"{sum(bool(t and t['trusted']) for t in agent.trust)} spots trusted | {n_skip} tests imagined | "
              f"{entry['seconds']:.0f}s", flush=True)
    return agent, history


def all_videos(world, cfg, pool, seed):
    """Upper bound: the same policy, trained once on every video that isn't in the exam."""
    train = [e for c in world.trainable for e in world.videos[c] if e["success"]]
    policy = cfg.policy_cls(world.coaster_xy, seed).fit(train, steps=2 * cfg.policy_cls.STEPS, seed=seed)
    groups, score = exam(pool, policy, world, cfg.rounds, cfg.exam_tries)
    print(f"   [all_videos] seed {seed}: exam {100 * score:.0f}% | {len(world.trainable)} videos", flush=True)
    return policy, {"exam_success": score, "exam_per_spot": [float(np.mean([e["success"] for e in g])) for g in groups],
                    "videos": len(world.trainable)}


# ---------------------------------------------------------------- pictures
def table_image(out_dir):
    from read_demos import TABLE_X, TABLE_Y
    for f in sorted(glob.glob(os.path.join(out_dir, "*", "birdseye.jpg"))):
        img = cv2.imread(f)
        if img is not None:
            ppm = img.shape[1] / (TABLE_X[1] - TABLE_X[0])
            return img, lambda x, y: (int(round((x - TABLE_X[0]) * ppm)), int(round((TABLE_Y[1] - y) * ppm)))
    return None, None


def label(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def map_logger(out_dir, world):
    """A picture per round: what happened at each practice spot, drawn on my table."""
    base, px = table_image(out_dir)

    def log(strategy, r, rates, decisions, trust):
        if base is None:
            return
        img = base.copy()
        for e in world.exam_spots:
            cv2.drawMarker(img, px(*e), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
        for k, (p, rate, dec) in enumerate(zip(world.probe_spots, rates, decisions)):
            cv2.circle(img, px(*p), 17, COLORS.get(dec, (128, 128, 128)), -1)
            cv2.circle(img, px(*p), 17, (255, 255, 255), 2)
            if trust[k] and trust[k]["trusted"]:
                cv2.circle(img, px(*p), 24, (255, 220, 0), 3)
            label(img, f"{100 * rate:.0f}", (px(*p)[0] - 14, px(*p)[1] + 6), 0.5)
        for j, (name, c) in enumerate(COLORS.items()):
            cv2.circle(img, (25, 30 + 30 * j), 10, c, -1)
            label(img, name, (44, 36 + 30 * j))
        label(img, "X = exam spot (never trained on)", (25, 36 + 30 * len(COLORS)))
        label(img, f"{strategy}, round {r}: number = success % at the practice spot; cyan ring = world model trusted",
              (25, img.shape[0] - 20), 0.55)
        cv2.imwrite(os.path.join("results", f"{strategy}_map_round{r}.png"), img)
    return log


def plot_curves(histories, upper, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for s, runs in histories.items():
        rounds = np.array([h["round"] for h in runs[0]])
        succ = np.array([[100 * h["exam_success"] for h in hist] for hist in runs])
        videos = np.array([[h["videos"] for h in hist] for hist in runs]).mean(0)
        minutes = np.array([[h["robot_minutes"] for h in hist] for hist in runs]).mean(0)
        mu = succ.mean(0)
        ax[0].plot(rounds, mu, "-o", color=STYLE[s], label=ag.STRATEGIES[s])
        if len(runs) > 1:
            ax[0].fill_between(rounds, succ.min(0), succ.max(0), color=STYLE[s], alpha=0.12)
        ax[1].plot(videos, mu, "-o", color=STYLE[s])
        ax[2].plot(minutes, mu, "-o", color=STYLE[s])
    if upper:
        u = 100 * np.mean([x["exam_success"] for x in upper])
        for a in ax:
            a.axhline(u, color="k", ls="--", lw=1)
        ax[0].plot([], [], "k--", lw=1, label=f"all {upper[0]['videos']} videos at once")
    titles = ["Exam (spots it never trained on)", "...against my videos used", "...against robot time"]
    xlabels = ["round", "my videos used (7 to start)", "robot minutes (every episode + 20 s reset)"]
    for a, t, xl in zip(ax, titles, xlabels):
        a.set_title(t)
        a.set_xlabel(xl)
        a.set_ylabel("exam success (%)")
        a.set_ylim(-5, 105)
        a.grid(alpha=0.3)
    ax[0].set_xticks(rounds)
    ax[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def world_model_report(world, pool, agent, cfg):
    """How good is the imagination? Train the world model on everything the ladder agent experienced, then test it
    on fresh episodes it has never seen (the final policy at practice and exam spots)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wm = WorldModel(world.coaster).fit(agent.real, cfg.wm_steps, 7)
    spots = list(world.probe_spots) + list(world.exam_spots)
    groups = ag.attempts(pool, agent.policy, spots, 3, 5151, jitter=0.01)
    test = [e for g in groups for e in g]
    spot_of = [k for k, g in enumerate(groups) for _ in g]
    H = 40
    errs = np.full((len(test), H), np.nan)
    pred, pred_open, actual, final_err = [], [], [], []
    for j, e in enumerate(test):
        traj = wm.rollout_actions(e["states"][0], e["act"])           # open loop: replay the real actions
        n = min(H, len(e["act"]))
        errs[j, :n] = 100 * np.linalg.norm(traj.mean(0)[1:n + 1, 3:6] - e["states"][1:n + 1, 3:6], axis=1)
        pred_open.append(bool(wm.reward(traj[:, -1]).mean() >= 0.5))
        imag, _ = wm.rollout_controller(e["states"][0], agent.policy.controller, len(e["act"]))   # closed loop
        pred.append(bool(wm.reward(imag[:, -1]).mean() >= 0.5))
        final_err.append(100 * float(np.linalg.norm(imag[:, -1, 3:5].mean(0) - e["states"][-1, 3:5])))
        actual.append(bool(e["success"]))
    pred, pred_open, actual = np.array(pred), np.array(pred_open), np.array(actual)
    # trust is judged on the first try at each spot, accuracy is measured on the other tries
    trusted = {k: trust_at_spot(wm, g[:1] * 2, agent.policy)["trusted"] for k, g in enumerate(groups)}
    first = {spot_of.index(k) for k in range(len(groups))}
    later = np.array([j not in first for j in range(len(test))])
    tr = np.array([trusted[spot_of[j]] for j in range(len(test))])
    rep = {"can_error_cm_after_1s": float(np.nanmean(errs[:, 9])), "after_2s": float(np.nanmean(errs[:, 19])),
           "after_4s": float(np.nanmean(errs[:, 39])), "reward_prediction_accuracy": float(np.mean(pred == actual)),
           "reward_prediction_accuracy_open_loop": float(np.mean(pred_open == actual)),
           "final_can_position_error_cm_median": float(np.median(final_err)),
           "episodes_tested": len(test), "real_successes": int(actual.sum()), "predicted_successes": int(pred.sum()),
           "accuracy_where_trusted": float(np.mean((pred == actual)[later & tr])) if (later & tr).any() else None,
           "accuracy_where_not_trusted": float(np.mean((pred == actual)[later & ~tr])) if (later & ~tr).any() else None,
           "spots_trusted": int(sum(trusted.values())), "spots": len(groups)}
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    t = np.arange(1, H + 1) * twin.STEP_DT
    ax.plot(t, np.nanmean(errs, 0), "-", color="tab:blue", label="mean")
    ax.fill_between(t, np.nanpercentile(errs, 25, 0), np.nanpercentile(errs, 75, 0), alpha=0.2, color="tab:blue",
                    label="middle 50%")
    ax.set_xlabel("seconds imagined ahead, without any correction")
    ax.set_ylabel("can position error (cm)")
    ax.set_title(f"World model on {len(test)} unseen episodes: reward predicted right "
                 f"{100 * rep['reward_prediction_accuracy']:.0f}%", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/world_model.png", dpi=130)
    plt.close(fig)
    return rep, wm


def imagined_poses(model, states):
    """Turn imagined states into poses we can draw: joint angles for each imagined gripper position (inverse
    kinematics), fingers as imagined, the can where the world model puts it."""
    import mink
    import mujoco
    robot = twin.Robot(model, states[0, 3:5])
    poses = []
    for s in states:
        for _ in range(4):
            robot.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(twin.DOWN), s[:3]))
            vel = mink.solve_ik(robot.cfg, [robot.task, robot.posture], twin.CONTROL_DT, solver="daqp", damping=1e-3,
                                limits=robot.limits)
            robot.cfg.integrate_inplace(vel, twin.CONTROL_DT)
        robot.data.qpos[:] = robot.cfg.q
        mujoco.mj_forward(model, robot.data)
        poses.append(np.r_[robot.data.qpos[robot.arm_qadr], np.full(2, s[7] / 2), s[3:6], 1, 0, 0, 0])
    return poses


def render_videos(world, pool, agent, wm):
    """results/exam_attempts.mp4 (the final robot at each exam spot) and results/imagined_vs_real.mp4."""
    import mujoco
    import camera
    model = twin.build_scene(world.coaster)
    renderer = mujoco.Renderer(model, height=540, width=960)
    data = mujoco.MjData(model)
    eps = [e for g in ag.attempts(pool, agent.policy, world.exam_spots, 1, 777, keep_q=True) for e in g]

    def exam_frames():
        for j, (e, clip) in enumerate(zip(eps, world.exam_clips)):
            can = np.array([q[9:12] for q in e["q"]])
            for k, q in enumerate(e["q"]):
                img = camera.render_pose(model, renderer, data, q,
                                         paths=[(camera.my_path(world.demos[clip]), camera.YELLOW), (can[:k + 1], camera.GREEN)])
                label(img, f"exam spot {j + 1}/{len(eps)} ({clip}, never trained here): "
                           f"{'success' if e['success'] else 'failed - ' + e['why']}", (20, 40), 0.7)
                label(img, "yellow: my path in that video (the robot never saw it)   green: the robot's", (20, 520), 0.55)
                yield img
    camera.write_video(exam_frames(), "results/exam_attempts.mp4")
    e = next((x for x in eps if x["success"]), eps[0])               # imagined vs real: same start, same policy
    traj, _ = wm.rollout_controller(e["states"][0], agent.policy.controller, len(e["act"]))
    imag = imagined_poses(model, traj.mean(0))
    real_can, imag_can = np.array([q[9:12] for q in e["q"]]), np.array([q[9:12] for q in imag])

    def side_frames():
        for k in range(max(len(imag), len(e["q"]))):
            a = camera.render_pose(model, renderer, data, e["q"][min(k, len(e["q"]) - 1)],
                                   paths=[(real_can[:k + 1], camera.GREEN)])
            b = camera.render_pose(model, renderer, data, imag[min(k, len(imag) - 1)],
                                   paths=[(imag_can[:k + 1], camera.YELLOW)])
            label(a, "real (physics simulator)", (20, 40), 0.9)
            label(b, "imagined (learned world model)", (20, 40), 0.9)
            yield np.hstack([a, b])
    camera.write_video(side_frames(), "results/imagined_vs_real.mp4")
    renderer.close()
    print("Saved results/exam_attempts.mp4 and results/imagined_vs_real.mp4", flush=True)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Request-or-retry learning from my phone videos.")
    p.add_argument("--out", default="out")
    p.add_argument("--strategies", nargs="*", default=list(ag.STRATEGIES))
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=3, help="runs per strategy (run 1 starts from my 7, the others random)")
    p.add_argument("--probe-tries", type=int, default=2, help="tries per practice spot when testing itself")
    p.add_argument("--exam-tries", type=int, default=10)
    p.add_argument("--practice", type=int, default=6, help="episodes per retry: my adapted video + its own tries")
    p.add_argument("--max-fix", type=int, default=4, help="weak spots handled per round")
    p.add_argument("--policy", choices=list(POLICIES), default="diffusion")
    p.add_argument("--wm-steps", type=int, default=2500)
    p.add_argument("--imagine-steps", type=int, default=160, help="0.1 s steps imagined per adapted video")
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    p.add_argument("--no-videos", action="store_true", help="skip rendering the result videos")
    p.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    cfg = p.parse_args()
    cfg.policy_cls = POLICIES[cfg.policy]
    if cfg.quick:
        cfg.rounds, cfg.seeds, cfg.probe_tries, cfg.exam_tries, cfg.practice, cfg.max_fix = 2, 1, 1, 1, 3, 3
        cfg.wm_steps = 800
        cfg.policy_cls.STEPS, cfg.policy_cls.POST_STEPS = 1500, 600
    os.makedirs("results", exist_ok=True)
    world = World(cfg.out)
    if cfg.quick:
        world.probe_spots = world.probe_spots[::max(1, len(world.probe_spots) // 5)][:5]
    print(f"{len(world.trainable)} of my videos can be learned from, {len(world.exam_spots)} exam spots, "
          f"{len(world.probe_spots)} practice spots | my time per video: {60 * world.my_minutes_per_video:.0f} s | "
          f"{cfg.workers} workers", flush=True)
    for k in range(cfg.seeds):
        print(f"   run {k + 1} starts from: {' '.join(world.start_set(k))}" + ("  (my pick)" if k == 0 else "  (random)"))
    histories, agents, tests, audits, t_start = {}, {}, [], [], time.time()
    with mp.get_context("fork").Pool(cfg.workers, initializer=ag.init_worker, initargs=(world.coaster,)) as pool:
        for s in cfg.strategies:
            histories[s] = []
            for seed in range(cfg.seeds):
                print(f"{s}, run {seed + 1}:", flush=True)
                a, hist = run(s, world, cfg, pool, seed, map_logger(cfg.out, world) if seed == 0 else None)
                histories[s].append(hist)
                tests += a.test_audits                  # checks of its imagination, from every run
                audits += a.imagine_audits
                if seed == 0:
                    agents[s] = a
                json.dump({"strategy": s, "run": seed, "start": world.start_set(seed), "owned": a.owned, "history": hist,
                           "film_here": a.film_here, "imagine_audits": a.imagine_audits, "test_audits": a.test_audits},
                          open(f"results/{s}_run{seed + 1}.json", "w"), indent=1, default=float)
        print("all videos at once (upper bound):", flush=True)
        upper, upper_policy = [], None
        for seed in range(cfg.seeds):
            pol, res = all_videos(world, cfg, pool, seed)
            upper.append(res)
            upper_policy = upper_policy or pol
        plot_curves(histories, upper, "results/learning_curves.png")
        main_agent = agents.get("ladder") or next(iter(agents.values()))
        wm_rep, wm = world_model_report(world, pool, main_agent, cfg)
        if not cfg.no_videos:
            render_videos(world, pool, main_agent, wm)
    pickle.dump(main_agent.policy, open("results/policy_ladder.pkl", "wb"))
    pickle.dump(upper_policy, open("results/policy_all_videos.pkl", "wb"))
    np.savez_compressed("results/final_training_data.npz",
                        obs=np.vstack([make_obs(e["states"][:-1], world.coaster_xy) for e in main_agent.train]),
                        chunks=np.vstack([chunks(e["act"]) for e in main_agent.train]),
                        sample_p=switch_weights(main_agent.train), exam_spots=np.array(world.exam_spots))
    if tests:
        wm_rep["imagined_self_tests"] = len(tests)
        wm_rep["imagined_self_tests_really_passed"] = float(np.mean([t["real_success"] for t in tests]))
    if audits:
        wm_rep["imagined_demos_checked"] = len(audits)
        wm_rep["imagined_demos_that_really_worked"] = float(np.mean([a["real"] for a in audits]))
    json.dump(wm_rep, open("results/world_model.json", "w"), indent=1)
    summary = {s: {"exam_success_start": float(np.mean([h[0]["exam_success"] for h in hs])),
                   "exam_success_end": float(np.mean([h[-1]["exam_success"] for h in hs])),
                   "exam_success_end_per_run": [h[-1]["exam_success"] for h in hs],
                   "my_videos_used": float(np.mean([h[-1]["videos"] for h in hs])),
                   "my_minutes": float(np.mean([h[-1]["minutes_of_my_time"] for h in hs])),
                   "robot_minutes": float(np.mean([h[-1]["robot_minutes"] for h in hs])),
                   "imagined_demos": float(np.mean([h[-1]["imagined"] for h in hs]))}
               for s, hs in histories.items()}
    summary["all_videos"] = {"exam_success_end": float(np.mean([u["exam_success"] for u in upper])),
                             "exam_success_end_per_run": [u["exam_success"] for u in upper],
                             "exam_per_spot": np.mean([u["exam_per_spot"] for u in upper], 0).tolist(),
                             "my_videos_used": float(upper[0]["videos"])}
    json.dump({"settings": {k: v for k, v in vars(cfg).items() if k != "policy_cls"},
               "start_sets": [world.start_set(k) for k in range(cfg.seeds)],
               "practice_spots": [q.tolist() for q in world.probe_spots], "exam_clips": world.exam_clips,
               "strategies": summary, "world_model": wm_rep, "minutes_total": (time.time() - t_start) / 60},
              open("results/summary.json", "w"), indent=1, default=float)
    print(f"\nResults (exam = {len(world.exam_spots)} spots it never trained on, {cfg.exam_tries} tries each, "
          f"averaged over {cfg.seeds} runs):")
    for s, v in summary.items():
        if s == "all_videos":
            print(f"  {'all videos at once':33s} exam        {100 * v['exam_success_end']:3.0f}% | "
                  f"my videos {v['my_videos_used']:.0f} (upper bound, no rounds)")
            continue
        print(f"  {ag.STRATEGIES[s]:33s} exam {100 * v['exam_success_start']:3.0f}% -> {100 * v['exam_success_end']:3.0f}% | "
              f"my videos {v['my_videos_used']:.1f} | robot {v['robot_minutes']:.0f} min | imagined demos "
              f"{v['imagined_demos']:.1f}")
    print(f"World model on unseen episodes: reward predicted right {100 * wm_rep['reward_prediction_accuracy']:.0f}% "
          f"(where trusted: {100 * (wm_rep['accuracy_where_trusted'] or 0):.0f}%, elsewhere: "
          f"{100 * (wm_rep['accuracy_where_not_trusted'] or 0):.0f}%) | can position error "
          f"{wm_rep['can_error_cm_after_1s']:.1f} cm after 1 s, {wm_rep['after_2s']:.1f} cm after 2 s, open loop")
    if tests:
        print(f"Self-tests done in imagination instead of on the robot: {len(tests)} "
              f"(the robot really succeeded in {100 * wm_rep['imagined_self_tests_really_passed']:.0f}% of those)")
    if audits:
        print(f"Imagined demos that really worked when checked in physics: "
              f"{100 * wm_rep['imagined_demos_that_really_worked']:.0f}% of {len(audits)}")
    print(f"Saved results/ in {(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    main()
