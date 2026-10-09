#!/usr/bin/env python3
"""
world_model.py: the robot's imagination. A learned model of what its actions do, and a measure of where to trust it.

The model predicts the next state (0.1 s later) from the state and the action: how the gripper really moves, how the
fingers open or close, and how the can moves. It is 5 small networks, each trained on a different resample of the
robot's real experience (an ensemble). It only sees relative positions and heights, never where on the table it is,
so what it learns about grasping and carrying applies everywhere. What it can't know (the laptop, the arm's reach)
shows up as wrong predictions in those places, which is why trust is measured spot by spot.

Uses:
  - imagine an episode from start to finish without the robot (a policy, or my video adapted to a new spot)
  - predict the reward of that episode
  - trust: at a spot, the model is trusted if it predicted every real attempt there correctly
"""
import numpy as np

import twin
from policy import MLP


class WorldModel:

    def __init__(self, coaster, n=5):
        self.coaster = coaster
        self.coaster_xy = np.asarray(coaster["center"], float)
        self.members = [MLP(11, 7, (200, 200), seed=i) for i in range(n)]

    @staticmethod
    def features(S, A):
        tcp, can, closed, width = S[:, :3], S[:, 3:6], S[:, 6:7], S[:, 7:8]
        return np.c_[tcp[:, 2:3], can - tcp, can[:, 2:3], closed, width, A]

    def fit(self, episodes, steps, seed=0):
        S = np.vstack([e["states"][:-1] for e in episodes])
        A = np.vstack([e["act"] for e in episodes])
        nxt = np.vstack([e["states"][1:] for e in episodes])
        X = self.features(S, A)
        # targets: how far the gripper missed its commanded move, how the can moved, how the fingers moved
        D = np.c_[nxt[:, :3] - S[:, :3] - A[:, :3], nxt[:, 3:6] - S[:, 3:6], nxt[:, 7:8] - S[:, 7:8]]
        rng = np.random.default_rng(seed)
        for i, m in enumerate(self.members):
            idx = rng.integers(0, len(X), len(X))                  # each member sees a different resample
            m.fit(X[idx], D[idx], steps=steps, seed=seed + i)
        return self

    def _step(self, S, A):
        X = self.features(S, A)
        out = np.array([m.predict(X[i:i + 1])[0] for i, m in enumerate(self.members)])
        S2 = S.copy()
        S2[:, :3] += A[:, :3] + out[:, :3]                         # the gripper moves as told, plus a learned error
        S2[:, 3:6] += out[:, 3:6]
        S2[:, 7] = np.clip(S2[:, 7] + out[:, 6], 0.0, 0.08)
        S2[:, 6] = A[:, 3]
        S2[:, 5] = np.maximum(S2[:, 5], twin.CAN_HEIGHT / 2 - 0.003)   # the can can't sink into the table
        return S2

    def rollout_actions(self, s0, actions):
        """Imagine a fixed sequence of actions, with no correction from reality (open loop)."""
        S = np.repeat(np.asarray(s0, float)[None], len(self.members), 0)
        traj = [S]
        for a in actions:
            S = self._step(S, np.repeat(np.asarray(a, float)[None], len(self.members), 0))
            traj.append(S)
        return np.stack(traj, 1)                                  # (members, T+1, 8)

    def rollout_controller(self, s0, make_controller, T):
        """Imagine a controller acting: each member reacts to its own predictions (closed loop) and stops, like a real
        episode, once the can is on the coaster and the hand has moved away."""
        S = np.repeat(np.asarray(s0, float)[None], len(self.members), 0)
        ctrls = [make_controller() for _ in self.members]
        traj, acts = [S], []
        done = np.zeros(len(S), bool)
        for _ in range(T):
            A = []
            for i, c in enumerate(ctrls):
                if done[i]:
                    A.append(np.r_[0.0, 0.0, 0.0, S[i, 6]])
                    continue
                d, g = c(S[i])
                A.append(np.r_[np.clip(d, -twin.DMAX, twin.DMAX), float(g == twin.CLOSED)])
            A = np.array(A)
            S2 = self._step(S, A)
            S = np.where(done[:, None], S, S2)
            done |= self.finished(S)
            traj.append(S)
            acts.append(A)
        return np.stack(traj, 1), np.stack(acts, 1)

    def finished(self, S):
        on = np.linalg.norm(S[:, 3:5] - self.coaster_xy, axis=1) < self.coaster["diameter"] / 2
        resting = S[:, 5] < twin.CAN_HEIGHT / 2 + twin.COASTER_THICKNESS + 0.01
        return on & resting & (S[:, 6] < 0.5) & (S[:, 2] > S[:, 5] + twin.CAN_HEIGHT / 2 + 0.06)

    def reward(self, final):
        """Predicted reward per member from the last imagined state: can on the coaster, set down, gripper open.
        (It can't predict touching the laptop: it doesn't know where the laptop is.)"""
        on = np.linalg.norm(final[:, 3:5] - self.coaster_xy, axis=1) < self.coaster["diameter"] / 2 - 0.005
        down = final[:, 5] < twin.CAN_HEIGHT / 2 + twin.COASTER_THICKNESS + 0.015
        return on & down & (final[:, 6] < 0.5)


def trust_at_spot(wm, episodes, policy):
    """Was the world model right here? Imagine the same policy from the start of each real attempt at this spot and
    compare with what really happened. Trusted = every predicted reward right, and where the attempt worked, the
    imagined can ended within 3 cm of the real one."""
    errs, spreads, agree = [], [], []
    for e in episodes:
        traj, _ = wm.rollout_controller(e["states"][0], policy.controller, len(e["act"]))
        final = traj[:, -1]
        errs.append(float(np.linalg.norm(final[:, 3:5].mean(0) - e["states"][-1, 3:5])))
        spreads.append(float(final[:, 3:5].std(0).max()))
        agree.append(bool(wm.reward(final).mean() >= 0.5) == bool(e["success"]))
    close = all(err < 0.03 for err, e in zip(errs, episodes) if e["success"])
    return {"error_cm": 100 * float(np.mean(errs)), "spread_cm": 100 * float(np.max(spreads)),
            "agree": float(np.mean(agree)), "trusted": bool(all(agree) and close and len(episodes) >= 2)}


def start_state(home_tcp, spot):
    """The state at the start of an episode: gripper at home, can standing at `spot`, fingers open."""
    return np.r_[home_tcp, spot[0], spot[1], twin.support_height(spot[0], spot[1]) + twin.CAN_HEIGHT / 2, 0.0, 0.08]
