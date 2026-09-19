"""Feature extraction for the Rail Corrugation task.

Two feature families:
  * T1 (physics): per-channel time stats + Welch band energies, aggregated per
    side (odd = Side I, even = Side II) and as side ratios; speed-adaptive
    corrugation-band energy; order-tracked band energies / peak prominence.
  * T7 (cyclostationary / periodicity): spectral kurtosis in the best band,
    autocorrelation periodicity, wavelet-packet band energies.

Everything is computed per recording and cached to a CSV keyed by filename.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt

from rail import io

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
FEATURE_VERSION = 5

# Fixed absolute PSD bands used for both signal types.
PSD_BANDS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800),
             (800, 1500), (1500, 3000), (3000, 5000)]
# Order bands (cycles per wheel revolution).  Corrugation ~ order 4-90.
ORDER_BANDS = [(4, 10), (10, 20), (20, 40), (40, 70), (70, 100)]
# Corrugation physical wavelength range [m] -> speed-adaptive frequency band.
LAMBDA_MIN, LAMBDA_MAX = 0.03, 0.5
SPR = 256  # samples per revolution for order tracking


def _banded(psd: np.ndarray, freqs: np.ndarray, bands) -> np.ndarray:
    """Sum PSD within each band -> (n_bands, n_channels), then log1p."""
    out = np.empty((len(bands), psd.shape[1]), dtype=np.float64)
    for i, (lo, hi) in enumerate(bands):
        m = (freqs >= lo) & (freqs < hi)
        out[i] = psd[m].sum(axis=0)
    return np.log1p(out)


def _localized(vec64: np.ndarray, prefix: str) -> Dict[str, float]:
    """Localisation-aware summaries of a per-channel quantity.

    Corrugation is localised to individual axle boxes, so instead of only
    averaging over all channels we look at the *distribution across cars* of
    each car's Side I / Side II maximum and its asymmetry.
    """
    v = np.nan_to_num(vec64.reshape(8, 8), nan=0.0)
    odd_max = v[:, io.SIDE1_POS].max(axis=1)
    even_max = v[:, io.SIDE2_POS].max(axis=1)
    # signed, scale-free asymmetry that is valid for negative quantities
    # (e.g. kurtosis) as well as positive ones.
    asym = (odd_max - even_max) / (np.abs(odd_max) + np.abs(even_max) + 1e-9)
    amp = np.maximum(np.abs(odd_max), np.abs(even_max))
    s = np.sort(asym)[::-1]
    return {
        f"{prefix}_loc_asym_max": float(asym.max()),
        f"{prefix}_loc_asym_min": float(asym.min()),
        f"{prefix}_loc_asym_mean": float(asym.mean()),
        f"{prefix}_loc_asym_std": float(asym.std()),
        f"{prefix}_loc_asym_gap": float(s[0] - s[1]),
        f"{prefix}_loc_amp_max": float(amp.max()),
        f"{prefix}_loc_amp_conc": float(amp.max() / (amp.mean() + 1e-9)),
    }


def _position_summary(vec64: np.ndarray, prefix: str) -> Dict[str, float]:
    """True **per-position** summary across all 64 axle boxes.

    The quantity is amplitude-normalised upstream, so these describe *where*
    and how concentrated a signature is, independent of overall level.
    """
    v = np.nan_to_num(vec64.reshape(-1), nan=0.0)      # index = car*8 + pos
    order = np.argsort(v)[::-1]
    s = v[order]
    mean, std = float(v.mean()), float(v.std())
    side_top1 = 1.0 if (order[0] % 8) in io.SIDE1_POS else 2.0
    return {
        f"{prefix}_pos_top1": float(s[0]),
        f"{prefix}_pos_top4mean": float(s[:4].mean()),
        f"{prefix}_pos_mean": mean,
        f"{prefix}_pos_top1_side": side_top1,
        f"{prefix}_pos_conc": float(s[0] / (mean + 1e-9)),
        f"{prefix}_pos_n_above": float(np.sum(v > mean + 2 * std + 1e-9)),
    }


def _side_aggregate(vec: np.ndarray) -> Dict[str, float]:
    """vec is (n_cars, n_pos).  Return per-side mean/ratio summaries."""
    vec = np.nan_to_num(vec, nan=0.0)
    s1 = vec[:, io.SIDE1_POS]
    s2 = vec[:, io.SIDE2_POS]
    m1, m2 = float(s1.mean()), float(s2.mean())
    x1, x2 = float(s1.max()), float(s2.max())
    # signed, scale-free asymmetry (valid for positive and negative quantities)
    ratio = (m1 - m2) / (abs(m1) + abs(m2) + 1e-9)
    max_ratio = (x1 - x2) / (abs(x1) + abs(x2) + 1e-9)
    return {
        "side1": m1, "side2": m2, "max1": x1, "max2": x2,
        "ratio": ratio, "max_ratio": max_ratio,
    }


def _signal_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """x shape (T, 8 cars, 8 pos).  Time + PSD + T7 features."""
    T = x.shape[0]
    x2 = x.reshape(T, -1).astype(np.float64)          # (T, 64)
    feats: Dict[str, float] = {}

    # --- time-domain stats per channel -> side aggregates -------------------
    rms = np.sqrt((x2 ** 2).mean(axis=0))
    std = x2.std(axis=0)
    peak = np.abs(x2).max(axis=0)
    kurt = stats.kurtosis(x2, axis=0)
    crest = peak / (rms + 1e-9)
    for name, v in [("rms", rms), ("std", std), ("peak", peak),
                    ("kurt", kurt), ("crest", crest)]:
        agg = _side_aggregate(v.reshape(8, 8))
        for k, val in agg.items():
            feats[f"{prefix}_{name}_{k}"] = val
    feats.update(_localized(rms, f"{prefix}_rms"))
    feats.update(_localized(kurt, f"{prefix}_kurt"))

    # --- Welch PSD band energies -------------------------------------------
    f, psd = signal.welch(x2, fs=io.FS, nperseg=2048, axis=0)   # (n_f, 64)
    band = _banded(psd, f, PSD_BANDS)                            # (n_bands, 64)
    for j, (lo, hi) in enumerate(PSD_BANDS):
        agg = _side_aggregate(band[j].reshape(8, 8))
        for k, val in agg.items():
            feats[f"{prefix}_band{lo}_{hi}_{k}"] = val
    # localisation-aware summaries on the most informative bands
    for lo, hi in [(50, 100), (200, 400), (800, 1500)]:
        j = PSD_BANDS.index((lo, hi))
        feats.update(_localized(band[j], f"{prefix}_band{lo}_{hi}"))

    # spectral centroid / flatness (noise-like vs tonal)
    freqs = f[:, None]
    cen = (psd * freqs).sum(0) / (psd.sum(0) + 1e-12)
    feats[f"{prefix}_centroid_ratio"] = _side_aggregate(cen.reshape(8, 8))["ratio"]

    # --- SHAPE features: amplitude-normalised spectrum ----------------------
    # Each channel's PSD is normalised to unit area, so every quantity below is
    # invariant to how loud that axle box is.  This is the "shape of the
    # anomaly" signal, separated from amplitude.
    norm = psd / (psd.sum(axis=0, keepdims=True) + 1e-12)
    for lo, hi in [(50, 100), (200, 400), (800, 1500), (1500, 3000)]:
        m = (f >= lo) & (f < hi)
        frac = norm[m].sum(0)
        feats[f"{prefix}_shp{lo}_{hi}_ratio"] = _side_aggregate(frac.reshape(8, 8))["ratio"]
        feats[f"{prefix}_shp{lo}_{hi}_max_ratio"] = _side_aggregate(frac.reshape(8, 8))["max_ratio"]
    # spectral entropy (tonal vs broadband) and flatness (geometric/arithmetic)
    ent = -(norm * np.log(norm + 1e-12)).sum(0)
    flat = np.exp(np.log(psd + 1e-12).mean(0)) / (psd.mean(0) + 1e-12)
    feats[f"{prefix}_shp_entropy_ratio"] = _side_aggregate(ent.reshape(8, 8))["ratio"]
    feats[f"{prefix}_shp_flatness_ratio"] = _side_aggregate(flat.reshape(8, 8))["ratio"]
    feats.update(_position_summary(flat, f"{prefix}_shpflat"))
    feats.update(_position_summary(norm[(f >= 200) & (f < 400)].sum(0), f"{prefix}_shp200_400"))
    feats.update(_position_summary(norm[(f >= 800) & (f < 1500)].sum(0), f"{prefix}_shp800_1500"))
    feats.update(_position_summary(rms, f"{prefix}_rms"))

    # --- T7: spectral kurtosis over a coarse filter bank --------------------
    sk = []
    for lo, hi in [(50, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4000)]:
        b, a = signal.butter(4, [lo / (io.FS / 2), hi / (io.FS / 2)], btype="band")
        xb = signal.filtfilt(b, a, x2, axis=0)
        # kurtosis of envelope of the analytic signal
        env = np.abs(signal.hilbert(xb, axis=0))
        sk.append(stats.kurtosis(env, axis=0))
    sk = np.array(sk)                       # (n_bands, 64)
    sk_band = sk.argmax(axis=0)             # best band index per channel
    sk_max = sk.max(axis=0)
    feats[f"{prefix}_sk_bandmean"] = float(sk_band.mean())
    for k, val in _side_aggregate(sk_max.reshape(8, 8)).items():
        feats[f"{prefix}_sk_max_{k}"] = val
    # per-position, scale-free impulsiveness (spectral kurtosis is amplitude-invariant)
    feats.update(_position_summary(sk_max, f"{prefix}_sk"))

    # --- T7: autocorrelation periodicity ------------------------------------
    xc = x2 - x2.mean(axis=0)
    ac = np.fft.irfft(np.abs(np.fft.rfft(xc, axis=0)) ** 2, axis=0)
    ac = ac / (ac[0:1] + 1e-12)
    ac_peak = ac[1:400].max(axis=0)
    feats[f"{prefix}_ac_peak_{'ratio'}"] = _side_aggregate(ac_peak.reshape(8, 8))["ratio"]
    # per-position, scale-free periodicity
    feats.update(_position_summary(ac_peak, f"{prefix}_ac"))

    # --- T7: wavelet-packet band energy ratios ------------------------------
    try:
        wp = pywt.WaveletPacket(data=x2.mean(axis=1), wavelet="db4", maxlevel=4)
        nodes = [n.path for n in wp.get_level(4, order="freq")]
        energies = np.array([np.sum(np.asarray(wp[p].data) ** 2) for p in nodes])
        energies = energies / (energies.sum() + 1e-12)
        for i, e in enumerate(energies):
            feats[f"{prefix}_wp_{i}"] = float(e)
    except Exception:
        pass

    return feats


def _order_features(rec: io.Recording, prefix: str) -> Dict[str, float]:
    """Order-tracked band energies for each side (empty if not reliable)."""
    feats: Dict[str, float] = {}
    if rec.speed["revs"] < 0.5:
        return feats
    vib = rec.vibration
    for side, pos in [("1", io.SIDE1_POS), ("2", io.SIDE2_POS)]:
        y = io.order_resample(vib[:, :, pos].mean(axis=(1, 2)), rec.theta, SPR)
        if y is None or len(y) < 32:
            continue
        y = y - y.mean()
        n = 1 << int(np.ceil(np.log2(len(y))))
        O = np.fft.rfftfreq(n, d=1.0 / SPR)
        P = np.abs(np.fft.rfft(y, n)) ** 2
        tot = P.sum() + 1e-12
        for lo, hi in ORDER_BANDS:
            m = (O >= lo) & (O < hi)
            feats[f"{prefix}_ord{lo}_{hi}_s{side}"] = float(np.log1p(P[m].sum() / tot))
        # peak prominence in corrugation band (order 4-90)
        m = (O >= 4) & (O <= 90)
        band = P[m]
        if band.size:
            feats[f"{prefix}_ord_peak_s{side}"] = float(np.log1p(band.max() / tot))
            feats[f"{prefix}_ord_peakmed_s{side}"] = float(
                np.log((band.max() + 1e-12) / (np.median(band) + 1e-12)))
    # side ratio of the corrugation-band peak
    a = feats.get(f"{prefix}_ord_peak_s1"); b = feats.get(f"{prefix}_ord_peak_s2")
    if a is not None and b is not None:
        feats[f"{prefix}_ord_peak_ratio"] = a - b
    return feats


def _order_phase_features(rec: io.Recording) -> Dict[str, float]:
    """Order-domain wave-propagation features (amplitude-invariant).

    Corrugation is a spatially periodic rail profile, so axles on the same rail
    see the *same order* (cycles per wheel revolution) with a fixed phase
    offset set by axle spacing.  Magnitude-squared coherence between cars in
    the order domain, and the consistency ("resultant length") of the
    inter-axle phase at the dominant corrugation order, are therefore
    amplitude-free signatures of a real rail wave.
    """
    feats: Dict[str, float] = {}
    if rec.speed["revs"] < 0.5:
        return feats
    vib = rec.vibration
    order_bands = [(4, 10), (10, 20), (20, 40), (40, 70), (70, 100)]
    for side, pos in [("1", io.SIDE1_POS), ("2", io.SIDE2_POS)]:
        Y = np.empty((0, 8))
        cols = []
        for c in range(io.N_CARS):
            y = io.order_resample(vib[:, c, pos].mean(axis=1), rec.theta, SPR)
            if y is None:
                break
            cols.append(y)
        if len(cols) < 8:
            continue
        n = min(len(c) for c in cols)
        Y = np.stack([c[:n] for c in cols], axis=1)          # (N, 8)
        Y = Y - Y.mean(0, keepdims=True)
        # dominant corrugation order from the mean order spectrum
        nfft = 1 << int(np.ceil(np.log2(n)))
        P = np.abs(np.fft.rfft(Y.mean(1), nfft)) ** 2
        O = np.fft.rfftfreq(nfft, d=1.0 / SPR)
        band = (O >= 4) & (O <= 90)
        peak_o = float(O[band][np.argmax(P[band])]) if band.any() else 0.0
        feats[f"ordpk_s{side}"] = peak_o
        # pairwise coherence in order bands + at the peak order
        for lo, hi in order_bands:
            vals = []
            for i in range(8):
                for j in range(i + 1, 8):
                    fq, C = signal.coherence(Y[:, i], Y[:, j], fs=SPR, nperseg=256)
                    m = (fq >= lo) & (fq < hi)
                    if m.any():
                        vals.append(float(np.nanmean(C[m])))
            if vals:
                feats[f"ordx_s{side}_{lo}_{hi}"] = float(np.mean(vals))
        # phase consistency at the peak order (wave propagation)
        if peak_o > 0:
            phases = []
            for i in range(8):
                for j in range(i + 1, 8):
                    fq, Cxy = signal.csd(Y[:, i], Y[:, j], fs=SPR, nperseg=256)
                    k = int(np.argmin(np.abs(fq - peak_o)))
                    phases.append(np.angle(Cxy[k]))
            if phases:
                feats[f"ordphase_s{side}"] = float(np.abs(np.mean(np.exp(1j * np.array(phases)))))
        feats[f"ordx_s{side}_adapt"] = feats.get(f"ordx_s{side}_40_70", np.nan)
    for lo, hi in order_bands:
        a, b = feats.get(f"ordx_s1_{lo}_{hi}"), feats.get(f"ordx_s2_{lo}_{hi}")
        if a is not None and b is not None and not (np.isnan(a) or np.isnan(b)):
            feats[f"ordx_ratio_{lo}_{hi}"] = a - b
    if "ordphase_s1" in feats and "ordphase_s2" in feats:
        feats["ordphase_ratio"] = feats["ordphase_s1"] - feats["ordphase_s2"]
    return feats


def _coherence_features(rec: io.Recording) -> Dict[str, float]:
    """Curve / shared-excitation features (amplitude-invariant).

    * ``xcar``: magnitude-squared coherence between cars on the SAME side.  A
      real rail corrugation is spatially periodic, so every axle on that rail
      sees the same temporal tone -> coherent.  Random/curve vibration is less
      structured.
    * ``xside``: coherence between the two sides of the same car (a curve, or a
      common-mode track input, raises this).
    * ``lf``: low-frequency (<3 Hz) side asymmetry, a possible quasi-static
      curve/roll proxy.
    """
    vib = rec.vibration
    T = vib.shape[0]
    v = rec.speed["speed_mps"]
    side_sig = {1: vib[:, :, io.SIDE1_POS].mean(2), 2: vib[:, :, io.SIDE2_POS].mean(2)}
    feats: Dict[str, float] = {}
    nperseg = 512

    def _coh_band(a, b, lo, hi):
        f, C = signal.coherence(a, b, fs=io.FS, nperseg=nperseg)
        m = (f >= lo) & (f < hi)
        return float(np.nanmean(C[m])) if m.any() else np.nan

    bands = [(5, 50), (50, 200), (200, 800)]
    for s in (1, 2):
        Xs = side_sig[s]
        for lo, hi in bands:
            feats[f"xcar_s{s}_{lo}_{hi}"] = float(
                np.nanmean([_coh_band(Xs[:, i], Xs[:, j], lo, hi)
                            for i in range(8) for j in range(i + 1, 8)]))
        if v > 0:
            lo, hi = v / LAMBDA_MAX, v / LAMBDA_MIN
            feats[f"xcar_s{s}_adapt"] = float(
                np.nanmean([_coh_band(Xs[:, i], Xs[:, j], lo, hi)
                            for i in range(8) for j in range(i + 1, 8)]))
    for lo, hi in bands:
        feats[f"xside_{lo}_{hi}"] = float(
            np.nanmean([_coh_band(side_sig[1][:, c], side_sig[2][:, c], lo, hi)
                        for c in range(8)]))
    feats["xcar_ratio_5_50"] = feats["xcar_s1_5_50"] - feats["xcar_s2_5_50"]
    feats["xcar_ratio_50_200"] = feats["xcar_s1_50_200"] - feats["xcar_s2_50_200"]

    # low-frequency side asymmetry (curve / roll proxy)
    x2 = vib.reshape(T, -1).astype(np.float64)
    f, psd = signal.welch(x2, fs=io.FS, nperseg=2048, axis=0)
    for lo, hi in [(0.3, 1), (1, 3), (3, 8)]:
        m = (f >= lo) & (f < hi)
        e = np.log1p(psd[m].sum(0))
        for k, val in _side_aggregate(e.reshape(8, 8)).items():
            if k in ("ratio", "max_ratio"):
                feats[f"lf{lo}_{hi}_{k}"] = val
    return feats


def _speed_adaptive_features(rec: io.Recording) -> Dict[str, float]:
    """Energy in the band where corrugation of wavelength [0.03,0.5] m lives."""
    v = rec.speed["speed_mps"]
    feats: Dict[str, float] = {}
    if v <= 0:
        return feats
    lo, hi = v / LAMBDA_MAX, v / LAMBDA_MIN        # Hz
    x2 = rec.vibration.reshape(rec.vibration.shape[0], -1).astype(np.float64)
    f, psd = signal.welch(x2, fs=io.FS, nperseg=2048, axis=0)
    m = (f >= lo) & (f < min(hi, io.FS / 2))
    band = psd[m].sum(axis=0)
    tot = psd.sum(axis=0) + 1e-12
    frac = np.log1p(band / tot)
    agg = _side_aggregate(frac.reshape(8, 8))
    for k, val in agg.items():
        feats[f"adapt_frac_{k}"] = val
    feats.update(_localized(frac, "adapt_frac"))
    # amplitude-normalised (shape) version + true per-position summary
    shp = band / (tot + 1e-12)
    for k, val in _side_aggregate(shp.reshape(8, 8)).items():
        feats[f"shp_adapt_{k}"] = val
    feats.update(_position_summary(shp, "shp_adapt"))
    feats["adapt_lo_hz"], feats["adapt_hi_hz"] = float(lo), float(hi)
    return feats


def extract_recording(rec) -> Dict[str, float]:
    """Feature vector for an already-loaded (or synthetic) recording."""
    feats: Dict[str, float] = {"file": rec.name}
    s = rec.speed
    feats.update(
        speed_kmh=s["speed_kmh"], speed_mps=s["speed_mps"], log_speed=np.log1p(s["speed_mps"]),
        revs=s["revs"], n_edges=s["n_edges"], reliable=int(s["reliable"]),
        order_reliable=int(s["order_reliable"]),
    )
    feats.update(_signal_features(rec.vibration, "vib"))
    feats.update(_signal_features(rec.shock, "shk"))
    feats.update(_order_features(rec, "vib"))
    feats.update(_speed_adaptive_features(rec))
    # NOTE: _order_phase_features (order-domain inter-axle coherence / wave
    # propagation) was tested as feature version 7; it did not improve results
    # (0.823 vs 0.833) and is left available but unused.
    # NOTE: _coherence_features (curve / shared-excitation proxies) was tested
    # as feature version 6; it did not improve results (0.821 vs 0.833) and is
    # left available for reproducibility but not used in the final model.
    return feats


def extract(path: str) -> Dict[str, float]:
    return extract_recording(io.load_recording(path))


def build(split: str, force: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, f"features_v{FEATURE_VERSION}_{split}.csv")
    if os.path.exists(out) and not force:
        return pd.read_csv(out)
    rows = [extract(p) for p in io.list_split(split)]
    df = pd.DataFrame(rows)
    if split == "train":
        labels = io.load_labels()
        df.insert(1, "label", [labels[f] for f in df["file"]])
        df.insert(2, "y", [io.LABEL_TO_INT[l] for l in df["label"]])
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    for s in ("train", "test"):
        df = build(s, force=True)
        print(f"{s}: {df.shape[0]} rows x {df.shape[1]} cols")
