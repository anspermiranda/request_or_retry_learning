#!/usr/bin/env python3
"""
agent.py: the learner, and how it decides what help to get.

The agent owns
  - data: robot demos made from the videos it has been given, plus its own episodes that earned reward 1
  - a policy trained on that data (policy.py), post-trained every round
  - a world model trained on everything it has experienced (world_model.py)
  - a bill: how many of my videos it used, and how many minutes of robot time

Every round it tests itself at practice spots spread over the table (never the exam spots). For each weak spot it
takes the cheapest help that works. This is the ladder:
  1. imagine  (free)       : if its world model has proven right at this spot, imagine my nearest video adapted to
                             the spot. If the world model predicts reward 1, train on that imagined demo.
  2. retry    (robot time) : if it has something to build on at this spot (a video of mine that starts within 10 cm,
                             or some successes of its own there), practise: my nearest video adapted to the spot,
                             then 5 tries of its own. Keep every episode that earned reward 1. This is the RL part:
                             the policy improves on its own successes (filtered behaviour cloning, the simplest form
                             of reward-weighted regression).
  3. request  (my time)    : nothing to build on, or practice found no success: ask for the video in the pile that
                             starts closest to this spot (within 10 cm). If there's none, the spot goes on a 'film
                             here' list (the videos I'd record next), and if it hasn't practised there, it does now.
The baselines get one kind of help only (request only, retry only) or two random videos per round.
"""
import numpy as np

import twin
from retarget import Expert
from world_model import WorldModel, start_state, trust_at_spot

SLOW = 1.5               # the robot moves 1.5x slower than I did
RESET_S = 20.0           # every real episode costs its duration + 20 s for someone to put the can back
NEAR = 0.10              # "near this spot" = within 10 cm
STRATEGIES = {"ladder": "imagine > retry > request (mine)", "request_only": "request only",
              "retry_only": "retry only (RL)", "random": "random videos"}


# ---------------------------------------------------------------- episodes in parallel (one simulator per worker)
_ENV = None


def init_worker(coaster):
    global _ENV
    _ENV = twin.TableEnv(twin.build_scene(coaster), coaster)


def run_task(args):
    """One episode in a worker. kind 'policy': the learned policy; kind 'expert': my video adapted to the spot."""
    kind, payload, spot, seed, noise, keep_q, executed = args
    controller = payload.controller(seed) if kind == "policy" else Expert(payload, SLOW, start=spot).act
    r = twin.rollout(_ENV, controller, spot, noise=noise, seed=seed, keep_q=keep_q, record_executed=executed)
    r["spot"] = np.asarray(spot, float)
    r["source"] = kind
    return r


def attempts(pool, policy, spots, tries, seed, noise=0.0, jitter=0.0, keep_q=False, executed=False):
    """The policy `tries` times at every spot. jitter moves the can by ~jitter metres each time."""
    rng = np.random.default_rng(seed)
    tasks = []
    for spot in spots:
        for _ in range(tries):
            p = np.asarray(spot, float) + (rng.normal(0, jitter, 2) if jitter else 0)
            tasks.append(("policy", policy, p, int(rng.integers(1 << 30)), noise, keep_q, executed))
    eps = pool.map(run_task, tasks)
    return [eps[i * tries:(i + 1) * tries] for i in range(len(spots))]


def robot_seconds(episodes):
    return sum(e["seconds"] + RESET_S for e in episodes)


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2]))


# ---------------------------------------------------------------- the agent
class Agent:

    def __init__(self, strategy, world, start, seed, cfg, pool):
        self.strategy, self.world, self.cfg, self.pool, self.seed = strategy, world, cfg, pool, seed
        self.rng = np.random.default_rng(seed)
        self.pile = [c for c in world.trainable if c not in start]
        self.owned, self.train, self.real = [], [], []
        self.videos, self.robot_s, self.imagined = 0, 0.0, 0
        self.policy = cfg.policy_cls(world.coaster_xy, seed)
        self.wm, self.trust, self.rates = None, None, None
        self.film_here, self.imagine_audits, self.test_audits = [], [], []
        for c in start:
            self.take(c)

    # -- data
    def take(self, clip):
        """A video arrives. Its robot demos join the data; turning it into demos costs robot time."""
        eps = self.world.videos[clip]
        self.owned.append(clip)
        self.videos += 1
        self.robot_s += robot_seconds(eps)
        self.real.extend(eps)
        self.train.extend(e for e in eps if e["success"])

    def nearest_owned(self, spot):
        """The owned video that starts closest to this spot (among those the robot could copy), and how far it is."""
        ok = [c for c in self.owned if any(e["success"] for e in self.world.videos[c])] or self.owned
        c = min(ok, key=lambda c: dist(self.world.start_of(c), spot))
        return c, dist(self.world.start_of(c), spot)

    # -- learning
    def learn(self, r):
        """Round 0 trains the policy from scratch; later rounds keep training it on everything so far."""
        self.policy.fit(self.train, seed=self.seed + 100 * r)

    # -- testing itself
    def self_test(self, r):
        """Test at the practice spots. Where its world model has proven right and it already succeeds, the ladder agent
        tests in imagination instead, which costs no robot time (the real result is kept only to check it)."""
        spots = self.world.probe_spots
        skip = []
        if self.strategy == "ladder" and self.wm is not None:
            for i, spot in enumerate(spots):
                if self.trust[i] and self.trust[i]["trusted"] and self.rates[i] == 1.0:
                    traj, _ = self.wm.rollout_controller(start_state(self.world.home_tcp, spot), self.policy.controller, 200)
                    if self.wm.reward(traj[:, -1]).mean() >= 0.8:
                        skip.append(i)
        probe = attempts(self.pool, self.policy, spots, self.cfg.probe_tries, 2000 + r + 97 * self.seed, jitter=0.01)
        rates = [float(np.mean([e["success"] for e in eps])) for eps in probe]
        for i in skip:
            self.test_audits.append({"round": r, "spot": i, "real_success": rates[i]})
            rates[i] = 1.0
        self.robot_s += sum(robot_seconds(eps) for i, eps in enumerate(probe) if i not in skip)
        old = self.trust
        self.trust = [None] * len(spots)
        if self.strategy == "ladder":
            self.wm = WorldModel(self.world.coaster).fit(self.real, self.cfg.wm_steps, self.seed + r)
            for i, eps in enumerate(probe):            # judged on attempts it has not been trained on
                self.trust[i] = old[i] if i in skip else trust_at_spot(self.wm, eps, self.policy)
        for i, eps in enumerate(probe):
            if i not in skip:
                self.real.extend(eps)
        self.rates = rates
        return rates, len(skip)

    # -- getting help
    def improve(self, r):
        """Fix the weak spots, weakest first, with this agent's kind of help. Returns what it did at each spot."""
        decisions = ["ok"] * len(self.rates)
        if self.strategy == "random":
            free = [c for c in self.pile if c not in self.owned]
            for c in (self.rng.choice(free, size=min(2, len(free)), replace=False) if free else []):
                self.take(str(c))
            return decisions
        weak = [i for i in np.argsort(self.rates, kind="stable") if self.rates[i] < 1.0][:self.cfg.max_fix]
        for i in weak:
            spot = self.world.probe_spots[i]
            if self.strategy == "ladder":
                decisions[i] = self.climb(i, spot, r)
            elif self.strategy == "retry_only":
                self.retry(spot, self.nearest_owned(spot)[0], r, i)
                decisions[i] = "retry"
            else:
                decisions[i] = self.request(spot, r)
        return decisions

    def climb(self, i, spot, r):
        """The ladder at one weak spot: the cheapest help that works."""
        src, d = self.nearest_owned(spot)
        if self.trust[i]["trusted"] and self.imagine(spot, r):
            return "imagine"
        something_to_build_on = d <= NEAR or self.rates[i] > 0
        if something_to_build_on and self.retry(spot, src, r, i):
            return "retry"
        asked = self.request(spot, r)
        if asked == "film here" and not something_to_build_on:      # nothing to ask for: practise anyway
            self.retry(spot, src, r, i)
            return "retry"
        return asked

    def imagine(self, spot, r):
        """Free: my nearest video adapted to the spot, played inside the world model."""
        src, _ = self.nearest_owned(spot)
        demo = self.world.demos[src]
        traj, acts = self.wm.rollout_controller(start_state(self.world.home_tcp, spot),
                                                lambda: Expert(demo, SLOW, start=spot).act, self.cfg.imagine_steps)
        ok = self.wm.reward(traj[:, -1])
        spread = float(traj[:, -1, 3:5].std(0).max())
        if ok.mean() < 0.6 or spread > 0.03:          # at least 3 of the 5 networks must agree it works
            return False
        for m in np.flatnonzero(ok)[:3]:
            end = int(np.argmax(self.wm.finished(traj[m]))) or len(acts[m])
            self.train.append({"states": traj[m][:end + 1], "act": acts[m][:end], "success": True, "source": "imagined",
                               "spot": np.array(spot), "clip": src})
        self.imagined += 1
        check = self.pool.map(run_task, [("expert", demo, spot, 0, 0.0, False, False)])[0]   # only to report on it
        self.imagine_audits.append({"round": r, "spot": [float(spot[0]), float(spot[1])], "real": bool(check["success"])})
        return True

    def retry(self, spot, src, r, i):
        """Robot time: practise at the spot. Keep every episode that earned reward 1. True if any did."""
        rng = np.random.default_rng(3000 + 31 * r + i + 7 * self.seed)
        noise = self.cfg.policy_cls.PRACTICE_NOISE
        tasks = [("expert", self.world.demos[src], spot, 0, 0.0, False, False)]            # my video, adapted
        tasks += [("policy", self.policy, spot, int(rng.integers(1 << 30)), noise, False, noise > 0)
                  for _ in range(self.cfg.practice - 1)]                                   # its own tries
        eps = self.pool.map(run_task, tasks)
        self.robot_s += robot_seconds(eps)
        self.real.extend(eps)
        wins = [e for e in eps if e["success"]]
        self.train.extend(wins)
        return bool(wins)

    def request(self, spot, r):
        """My time: the pile video that starts closest to this spot, if it's within 10 cm."""
        free = [c for c in self.pile if c not in self.owned]
        near = min(free, key=lambda c: dist(self.world.start_of(c), spot)) if free else None
        if near is not None and dist(self.world.start_of(near), spot) <= NEAR:
            self.take(near)
            return "request"
        self.film_here.append({"strategy": self.strategy, "round": r, "spot": [float(spot[0]), float(spot[1])]})
        return "film here"

    def bill(self):
        return {"videos": self.videos, "minutes_of_my_time": self.videos * self.world.my_minutes_per_video,
                "robot_minutes": self.robot_s / 60, "imagined": self.imagined, "train_episodes": len(self.train)}
