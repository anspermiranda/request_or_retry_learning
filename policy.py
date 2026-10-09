#!/usr/bin/env python3
"""
policy.py: the robot's brain. It maps what the robot sees to what it does next.

What it sees (the observation, 8 numbers), all relative to the objects so that it carries over to new spots:
  the gripper's height, where the can is relative to the gripper (3), where the coaster is relative to the can (2),
  the can's height above the table, and how far the fingers are open.
What it does: the next 8 decisions (0.8 s) at once, an "action chunk" as in ACT. The robot carries out the first 4,
then looks again. Each decision is a move of up to 2.5 cm and open/close.

Two policies, both small numpy networks that train on a laptop CPU in under a minute:
  DiffusionPolicy (default): starts from random numbers and cleans them up into a chunk, step by step (Chi et al.,
                  2023). It can keep several different ways of doing the task apart, where plain regression averages
                  them into something that does neither.
  BCPolicy       : plain behaviour cloning, 3 networks averaged. My first attempt; kept for comparison.

Post-training: the first fit() trains from scratch. Every later fit() keeps training the same network on everything
collected so far (fewer steps, smaller learning rate), so nothing it learned is thrown away.
"""
import numpy as np

import twin

K, M = 8, 4               # predict 8 decisions ahead, carry out 4, look again
OBS = 8


# ---------------------------------------------------------------- a small neural network in numpy
class MLP:
    def __init__(self, n_in, n_out, hidden=(256, 256), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.mx = self.sx = self.my = self.sy = None

    def _forward(self, X):
        hs = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return hs

    def predict(self, X):
        Xn = ((np.atleast_2d(X) - self.mx) / self.sx).astype(np.float32)
        return self._forward(Xn)[-1] * self.sy + self.my

    def fit(self, X, Y, steps=3000, lr=1e-3, batch=256, noise=None, weights=None, sample_p=None, seed=0, rescale=True):
        """Adam on a (weighted) squared error. noise: input noise, so small mistakes of its own don't throw it off.
        sample_p: how often each row is picked (rare but important moments more often). rescale=False keeps the
        input/output scaling from the first fit (post-training)."""
        if rescale or self.mx is None:
            self.mx, self.sx = X.mean(0), X.std(0) + 1e-6
            self.my, self.sy = Y.mean(0), Y.std(0) + 1e-6
        Xn, Yn = ((X - self.mx) / self.sx).astype(np.float32), ((Y - self.my) / self.sy).astype(np.float32)
        noise_n = None if noise is None else (np.asarray(noise) / self.sx).astype(np.float32)

        def batch_fn(rng):
            idx = rng.choice(len(Xn), batch, p=sample_p) if sample_p is not None else rng.integers(0, len(Xn), batch)
            xb = Xn[idx]
            if noise_n is not None:
                xb = xb + rng.normal(0, 1, xb.shape).astype(np.float32) * noise_n
            return xb, Yn[idx]
        return self.train(batch_fn, steps, lr, weights, seed)

    def train(self, batch_fn, steps, lr=1e-3, weights=None, seed=0):
        """The training loop on ready-made batches: Adam, a short warm-up, then a cosine decay of the learning rate."""
        rng = np.random.default_rng(seed)
        w = 1.0 if weights is None else np.asarray(weights, np.float32)
        params = self.W + self.b
        m, v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        for t in range(1, steps + 1):
            xb, yb = batch_fn(rng)
            hs = self._forward(xb)
            g = 2 * (hs[-1] - yb) / len(xb) * w
            gW, gb = [None] * len(self.W), [None] * len(self.W)
            for i in range(len(self.W) - 1, -1, -1):
                gW[i], gb[i] = hs[i].T @ g, g.sum(0)
                if i > 0:
                    g = (g @ self.W[i].T) * (hs[i] > 0)
            lr_t = lr * min(1.0, t / 200) * 0.5 * (1 + np.cos(np.pi * t / steps))
            for j, (p, gr) in enumerate(zip(params, gW + gb)):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr * gr
                p -= lr_t * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        return self


# ---------------------------------------------------------------- what it sees, what it does
def make_obs(S, coaster_xy):
    """State rows -> observations. Not its own last gripper command: a policy that sees it learns to repeat it and
    never switches (the copycat problem)."""
    S = np.atleast_2d(S)
    tcp, can, width = S[:, :3], S[:, 3:6], S[:, 7:8]
    return np.c_[tcp[:, 2:3], can - tcp, np.asarray(coaster_xy) - can[:, :2], can[:, 2:3] - twin.CAN_HEIGHT / 2, width]


def chunks(act):
    """For every step, the next K actions (padded at the end with 'stay still, gripper as it was')."""
    pad = np.vstack([act, np.repeat(np.r_[0.0, 0.0, 0.0, act[-1, 3]][None], K, 0)])
    return np.hstack([pad[k:k + len(act)] for k in range(K)])


def switch_weights(episodes, near=2, boost=10.0):
    """Closing and letting go are a few steps in a hundred, so a network trained on all steps equally blurs them away
    and never lets go. The steps around each gripper switch are picked 10x more often."""
    w = []
    for e in episodes:
        g = e["act"][:, 3]
        we = np.ones(len(g))
        for i in np.flatnonzero(np.diff(g) != 0):
            we[max(0, i - near + 1):i + near + 2] = boost
        w.append(we)
    w = np.concatenate(w)
    return w / w.sum()


def dataset(episodes, coaster_xy):
    X = np.vstack([make_obs(e["states"][:-1], coaster_xy) for e in episodes])
    Y = np.vstack([chunks(e["act"]) for e in episodes])
    return X, Y, switch_weights(episodes)


def chunk_controller(sample, coaster_xy):
    """A controller for one episode: state -> (move, gripper), planning a new chunk every M steps."""
    queue = []

    def act(s):
        if not queue:
            queue.extend(sample(make_obs(s, coaster_xy))[0].reshape(K, 4)[:M])
        a = queue.pop(0)
        return np.clip(a[:3], -twin.DMAX, twin.DMAX), (twin.CLOSED if a[3] > 0.5 else twin.OPEN)
    return act


# ---------------------------------------------------------------- behaviour cloning
class BCPolicy:
    """3 small networks (2 x 256) trained with different seeds; their answers are averaged."""

    N, STEPS, POST_STEPS, POST_LR, PRACTICE_NOISE = 3, 3000, 1500, 3e-4, 0.006

    def __init__(self, coaster_xy, seed=0):
        self.coaster_xy = np.asarray(coaster_xy, float)
        self.nets = [MLP(OBS, 4 * K, (256, 256), seed * 10 + i) for i in range(self.N)]
        self.trained = False

    def fit_arrays(self, X, Y, sample_p, steps, seed=0, lr=1e-3):
        for i, net in enumerate(self.nets):
            net.fit(X, Y, steps=steps, lr=lr, noise=np.r_[[0.0015] * (OBS - 1), 0.002], weights=[1, 1, 1, 4] * K,
                    sample_p=sample_p, seed=seed * 10 + i, rescale=not self.trained)
        self.trained = True
        return self

    def fit(self, episodes, steps=None, seed=0):
        X, Y, p = dataset(episodes, self.coaster_xy)
        if self.trained:
            return self.fit_arrays(X, Y, p, steps or self.POST_STEPS, seed, self.POST_LR)
        return self.fit_arrays(X, Y, p, steps or self.STEPS, seed)

    def sample(self, obs, method=None, n=None, rng=None):
        return np.mean([net.predict(obs) for net in self.nets], 0)

    def controller(self, seed=0):
        return chunk_controller(self.sample, self.coaster_xy)


# ---------------------------------------------------------------- diffusion policy
def alpha_bar(T):
    """Cosine noise schedule (Nichol & Dhariwal, 2021): how much of the clean chunk is left at each of the T levels."""
    s = 0.008
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2) ** 2
    return np.clip(f[1:] / f[0], 1e-5, 0.9999)


def t_embed(t, T, dim=16):
    freqs = np.exp(np.linspace(0, np.log(200), dim // 2))
    x = (np.asarray(t, float) / T)[:, None] * freqs[None]
    return np.c_[np.sin(x), np.cos(x)]


class DiffusionPolicy:
    """A network (3 x 256) sees the observation, a noisy chunk and the noise level, and guesses the clean chunk.
    Trained on 100 noise levels. Predicting the clean chunk (not the noise) keeps sampling in only 4 steps (DDIM,
    Song et al., 2021) as accurate as the standard 100 steps on this data; speed.py measures it."""

    T, STEPS, POST_STEPS, POST_LR, SAMPLE_STEPS = 100, 8000, 3000, 3e-4, 4
    PRACTICE_NOISE = 0.0          # when practising, its own sampling already explores

    def __init__(self, coaster_xy, seed=0, hidden=(256, 256, 256)):
        self.coaster_xy = np.asarray(coaster_xy, float)
        self.ab = alpha_bar(self.T)
        n_in, n_out = OBS + 4 * K + 16, 4 * K
        self.net = MLP(n_in, n_out, hidden, seed)
        self.net.mx, self.net.sx, self.net.my, self.net.sy = np.zeros(n_in), np.ones(n_in), np.zeros(n_out), np.ones(n_out)
        self.trained = False

    def fit_arrays(self, X, Y, sample_p, steps, seed=0, lr=1e-3):
        if not self.trained:                # the scaling is fixed by the first training set
            self.om, self.os = X.mean(0), X.std(0) + 1e-6
            self.cm, self.cs = Y.mean(0), Y.std(0) + 1e-6
        O, C = (X - self.om) / self.os, (Y - self.cm) / self.cs
        T = self.T

        def batch_fn(rng):                  # blur a clean chunk to a random noise level, learn to clean it up
            idx = rng.choice(len(O), 256, p=sample_p)
            t = rng.integers(0, T, 256)
            ab = self.ab[t][:, None]
            x = np.sqrt(ab) * C[idx] + np.sqrt(1 - ab) * rng.normal(size=(256, C.shape[1]))
            return np.c_[O[idx], x, t_embed(t, T)].astype(np.float32), C[idx].astype(np.float32)
        self.net.train(batch_fn, steps, lr=lr, weights=[1, 1, 1, 4] * K, seed=seed)
        self.trained = True
        return self

    def fit(self, episodes, steps=None, seed=0):
        X, Y, p = dataset(episodes, self.coaster_xy)
        if self.trained:                    # post-training: keep going from where it is
            return self.fit_arrays(X, Y, p, steps or self.POST_STEPS, seed, self.POST_LR)
        return self.fit_arrays(X, Y, p, steps or self.STEPS, seed)

    def sample(self, obs, method="ddim", n=None, rng=None):
        """DDIM: n deterministic steps (default 4). DDPM: the standard 100 noisy steps."""
        rng = rng if rng is not None else np.random.default_rng(0)
        n, T = n or self.SAMPLE_STEPS, self.T
        o = (np.atleast_2d(obs) - self.om) / self.os
        x = rng.normal(size=(len(o), 4 * K))
        ts = np.arange(T - 1, -1, -1) if method == "ddpm" else np.linspace(T - 1, 0, n).round().astype(int)
        for i, t in enumerate(ts):
            ab_t = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else 1.0
            x0 = np.clip(self.net.predict(np.c_[o, x, t_embed(np.full(len(o), t), T)]), -4, 4)
            eps = (x - np.sqrt(ab_t) * x0) / np.sqrt(1 - ab_t)
            if method == "ddpm" and i + 1 < len(ts):            # the standard step, with fresh noise
                beta = 1 - ab_t / ab_prev
                mean = (np.sqrt(ab_prev) * beta * x0 + np.sqrt(1 - beta) * (1 - ab_prev) * x) / (1 - ab_t)
                x = mean + np.sqrt(beta * (1 - ab_prev) / (1 - ab_t)) * rng.normal(size=x.shape)
            else:
                x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * eps
        return x * self.cs + self.cm

    def controller(self, seed=0):
        rng = np.random.default_rng(seed)
        return chunk_controller(lambda o: self.sample(o, "ddim", self.SAMPLE_STEPS, rng), self.coaster_xy)


POLICIES = {"diffusion": DiffusionPolicy, "bc": BCPolicy}
