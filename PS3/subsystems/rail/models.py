"""Classifier wrappers for the Rail task.

The key structural idea (T3) is to model the two rails **independently** as
binary detectors ("is Side I corrugated?", "is Side II corrugated?") and then
combine the two probabilities into the required 3-class label.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb


GBM_BASE = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=8,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    verbose=-1,
    random_state=0,
)


class LGBM3:
    """Plain 3-class baseline (E1)."""

    def __init__(self, **kw):
        p = dict(objective="multiclass", num_class=3, class_weight="balanced", **GBM_BASE)
        p.update(kw)
        self.model = lgb.LGBMClassifier(**p)
        self.multi = True

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class TwoSidedLGBM:
    """T3: two independent binary detectors, combined probabilistically.

    p_none = (1-pI)(1-pII);  p_I = pI(1-pII);  p_II = (1-pI)pII.
    When both detectors fire, the side with the larger probability wins.
    """

    def __init__(self, **kw):
        p = dict(objective="binary", **GBM_BASE)
        p.update(kw)
        self.params = p
        self.mI = self.mII = None
        self.multi = False

    def fit(self, X, y):
        yI = (y == 1).astype(int)
        yII = (y == 2).astype(int)
        spwI = (len(yI) - yI.sum()) / max(yI.sum(), 1)
        spwII = (len(yII) - yII.sum()) / max(yII.sum(), 1)
        self.mI = lgb.LGBMClassifier(scale_pos_weight=spwI, **self.params).fit(X, yI)
        self.mII = lgb.LGBMClassifier(scale_pos_weight=spwII, **self.params).fit(X, yII)
        return self

    def predict_sides(self, X):
        """Return the two independent side probabilities (pI, pII)."""
        return self.mI.predict_proba(X)[:, 1], self.mII.predict_proba(X)[:, 1]

    @staticmethod
    def sides_to_label(pI: np.ndarray, pII: np.ndarray,
                       tI: float = 0.5, tII: float = 0.5) -> np.ndarray:
        """Threshold rule: both fire -> stronger side; else the firing side."""
        out = np.zeros(len(pI), dtype=int)
        fireI = pI >= tI
        fireII = pII >= tII
        out[fireI & ~fireII] = 1
        out[fireII & ~fireI] = 2
        both = fireI & fireII
        out[both] = np.where(pI[both] >= pII[both], 1, 2)
        return out

    def predict_proba(self, X):
        pI, pII = self.predict_sides(X)
        p0 = (1 - pI) * (1 - pII)
        p1 = pI * (1 - pII)
        p2 = (1 - pI) * pII
        both = (pI >= 0.5) & (pII >= 0.5)
        if both.any():
            total = pI[both] + pII[both] + 1e-12
            p1[both] = pI[both] / total * (pI[both] + pII[both])
            p2[both] = pII[both] / total * (pI[both] + pII[both])
            p0[both] = 0.0
        P = np.vstack([p0, p1, p2]).T
        return P / P.sum(axis=1, keepdims=True)

    @property
    def feature_importances_(self):
        return self.mI.feature_importances_ + self.mII.feature_importances_


def proba_to_label(P: np.ndarray) -> np.ndarray:
    return P.argmax(axis=1)
