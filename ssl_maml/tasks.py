"""Task distributions used by the experiments."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Task:
    Xs: np.ndarray
    ys: np.ndarray
    Xq: np.ndarray
    yq: np.ndarray
    n_way: int
    info: dict = field(default_factory=dict)

    @property
    def Xu(self):
        """Unlabeled pool available at adaptation time (inputs only)."""
        return self.Xq


def sample_gauss_task(rng, n_way=5, dim=2, K=1, Q=15, radius=1.0, sigma=0.35,
                      warp=0.0, phase=None, extra_unlab=0, noise=0.0,
                      informative=2):
    """N-way few-shot task: class means on a circle (task-specific phase),
    optionally warped by an anisotropic task-specific linear map.

    dim > informative fills the remaining coordinates with pure noise, so a
    learner that does not discover the informative subspace cannot cluster the
    points -- this is what makes the meta-learned representation necessary.

    Unlabeled pool = query inputs (transductive regime), i.e. the model may use
    the *inputs* of the evaluation points but never their labels.
    """
    if phase is None:
        phase = rng.uniform(0.0, 2.0 * np.pi)
    ang = phase + 2.0 * np.pi * np.arange(n_way) / n_way
    means = radius * np.stack([np.cos(ang), np.sin(ang)], 1)

    A = np.eye(2)
    if warp > 0:
        R = np.linalg.qr(rng.normal(size=(2, 2)))[0]
        s = np.exp(rng.uniform(-warp, warp, size=2))
        A = R @ np.diag(s) @ R.T

    zs = rng.normal(0, sigma, size=(n_way, K, 2))
    zq = rng.normal(0, sigma, size=(n_way, Q, 2))
    zx = rng.normal(0, sigma, size=(n_way, extra_unlab, 2)) if extra_unlab else None

    def to_x(z):
        return (z @ A.T) + means[:, None, :]

    Xs = to_x(zs).reshape(n_way * K, 2)
    ys = np.repeat(np.arange(n_way), K)
    Xq = to_x(zq).reshape(n_way * Q, 2)
    yq = np.repeat(np.arange(n_way), Q)
    if zx is not None:
        Xq = np.vstack([Xq, to_x(zx).reshape(-1, 2)])
        yq = np.concatenate([yq, np.repeat(np.arange(n_way), extra_unlab)])

    def embed(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        if dim <= X.shape[1]:
            return X[:, :dim].copy()
        out = np.zeros((n, dim))
        out[:, :X.shape[1]] = X
        if noise > 0:
            out[:, X.shape[1]:] = rng.normal(0.0, noise, size=(n, dim - X.shape[1]))
        return out

    perm = rng.permutation(len(Xs))
    return Task(embed(Xs)[perm], ys[perm], embed(Xq), yq, n_way,
                {"A": A, "phase": phase, "means": means})


def sample_multitask_1d(rng, K=10, Q=10, amp_range=(0.1, 5.0), phase_range=(0.0, np.pi),
                        x_range=5.0):
    """MAML's sinusoid task family, used as a sanity-check benchmark."""
    a = rng.uniform(*amp_range)
    b = rng.uniform(*phase_range)
    xs = rng.uniform(-x_range, x_range, size=(K, 1))
    xq = rng.uniform(-x_range, x_range, size=(Q, 1))
    f = lambda x: a * np.sin(x + b)
    # order the support points by x so that "half the range" experiments work
    return Task(xs, f(xs), xq, f(xq), 1, {"a": a, "b": b, "fn": f})
