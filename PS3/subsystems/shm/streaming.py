"""Single-pass (online) rainflow counting for streaming SHM.

This is a faithful streaming reimplementation of the ``fatpack`` 4-point
counter used to fit the shipped model:

  * identical class-centre convention
    (``centre = lo + round((x-lo)/w)*w`` with ``w = (hi-lo)/k``);
  * identical peak-valley reversal filtering on the *collapsed* centre
    sequence (consecutive equal centres removed);
  * identical 4-point cycle-closure rule;
  * identical residue close-out: the residue is concatenated with itself and
    rainflow-counted a second time.

The only approximation left is the **class grid**: the batch counter derives its
boundaries from each segment's own min/max, whereas a stream must use a fixed
grid known in advance.  The stream state is a bounded ``k``-level cycle-range
histogram; ``S(m)`` is computed from it on demand.
"""
from __future__ import annotations

import numpy as np

K = 64


class OnlineRainflow:
    def __init__(self, lo: float, hi: float, k: int = K):
        self.k = k
        self.lo = float(lo)
        self.hi = float(hi)
        self.w = (self.hi - self.lo) / k if self.hi > self.lo else 1.0
        # Range levels run 0..k (a full-span cycle is k classes wide).
        self.hist = np.zeros(k + 1)
        self.n = 0
        self._residue: list[float] = []
        self._first: float | None = None
        self._prev = 0.0
        self._dir = 0
        self._last_center: float | None = None

    # ---------------------------------------------------------------- grid --
    def _center(self, x: float) -> float:
        m = int(np.floor((x - self.lo) / self.w + 0.5))
        m = min(self.k, max(0, m))
        return self.lo + m * self.w

    def _tally(self, rng: float) -> None:
        level = int(round(rng / self.w))
        if level > 0:
            self.hist[min(self.k, level)] += 1.0

    # ----------------------------------------------------------- reversals --
    def push(self, x: float) -> None:
        self.n += 1
        c = self._center(float(x))
        if self._first is None:
            self._first = self._prev = self._last_center = c
            self._emit(c)
            return
        if c == self._last_center:
            return
        self._last_center = c
        d = 1 if c > self._prev else -1
        if self._dir == 0:
            self._dir, self._prev = d, c
        elif d == self._dir:
            self._prev = c
        else:
            self._emit(self._prev)
            self._dir, self._prev = d, c

    def extend(self, xs: np.ndarray) -> None:
        for x in np.asarray(xs, dtype=float):
            self.push(float(x))

    def _emit(self, v: float) -> None:
        self._residue.append(v)
        while len(self._residue) >= 4:
            s0, s1, s2, s3 = self._residue[-4:]
            d1, d2, d3 = abs(s1 - s0), abs(s2 - s1), abs(s3 - s2)
            if d2 <= d1 and d2 <= d3:
                self._tally(abs(s2 - s1))
                del self._residue[-3]
                del self._residue[-2]
            else:
                break

    # ------------------------------------------------------------- finish --
    def finish(self) -> np.ndarray:
        if self._first is not None:
            self._emit(self._prev)                 # last reversal
            residue = self._residue
            if len(residue) >= 2:
                joined = self._concatenate(residue, residue)
                stack: list[float] = []
                for v in joined:
                    stack.append(float(v))
                    while len(stack) >= 4:
                        s0, s1, s2, s3 = stack[-4:]
                        d1, d2, d3 = abs(s1 - s0), abs(s2 - s1), abs(s3 - s2)
                        if d2 <= d1 and d2 <= d3:
                            self._tally(abs(s2 - s1))
                            del stack[-3]
                            del stack[-2]
                        else:
                            break
        self._residue = []
        return self.hist

    @staticmethod
    def _concatenate(r1, r2) -> np.ndarray:
        a = np.asarray(r1, dtype=float)
        b = np.asarray(r2, dtype=float)
        if len(a) < 2 or len(b) < 2:
            return np.concatenate([a, b])
        d_start, d_end, d_join = b[1] - b[0], a[-1] - a[-2], b[0] - a[-1]
        t1, t2 = d_end * d_start, d_end * d_join
        if t1 > 0 and t2 < 0:
            a, b = a, b
        elif t1 > 0 and t2 >= 0:
            a, b = a[:-1], b[1:]
        elif t1 < 0 and t2 >= 0:
            a, b = a, b[1:]
        elif t1 < 0 and t2 < 0:
            a, b = a[:-1], b
        return np.concatenate([a, b])

    # ------------------------------------------------------------ damage --
    def pseudo_damage(self, m: float) -> float:
        levels = np.arange(self.k + 1)
        return float(np.sum(self.hist * (levels * self.w) ** m))


def online_damage(signal: np.ndarray, lo: float, hi: float, model, grid, k: int = K) -> float:
    """Damage for one segment using the online counter and a fitted model."""
    rf = OnlineRainflow(lo, hi, k)
    rf.extend(signal)
    rf.finish()
    S = np.array([[rf.pseudo_damage(float(m)) for m in grid]])
    return float(np.exp(model.predict(np.log(S))[0]))
