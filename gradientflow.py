# GradientFlow - Track gradient-flow health across stacked weight layers
# Part of the Kapildhinwa2010-dev Toolkit.

"""DeepAnalyze - layer-wise spectral analysis for deep networks.

Lineage: Reverse.Dean (b2/b4). Given a dict of weight matrices ("layers"),
this module computes singular-value spectra, effective ranks and condition
numbers, and ranks layers by bottleneck severity.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "layer_spectrum",
    "effective_rank",
    "layer_report",
    "analyze_model",
    "find_bottlenecks",
    "gradient_health",
    "flow_report",
]


def layer_spectrum(weights: np.ndarray) -> np.ndarray:
    """Singular values of ``weights`` in descending order."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2-D matrix")
    return np.linalg.svd(weights, compute_uv=False)


def effective_rank(weights: np.ndarray, energy: float = 0.99) -> int:
    """Smallest k whose singular values cover ``energy`` of total energy."""
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    s = layer_spectrum(weights)
    total = float(s.sum())
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(s) / total
    k = int(np.searchsorted(cumulative, energy, side="left") + 1)
    return int(min(k, s.size))


def layer_report(weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.size == 0:
        raise ValueError("weights must be a non-empty 2-D matrix")
    s = layer_spectrum(weights)
    s_max = float(s[0])
    s_min = float(s[-1])
    return {
        "shape": (int(weights.shape[0]), int(weights.shape[1])),
        "frobenius_norm": float(np.linalg.norm(weights)),
        "spectral_norm": s_max,
        # Numerically singular: s_min below machine-eps scale => inf
        "condition_number": (
            s_max / s_min
            if s_min > s_max * np.finfo(np.float64).eps * max(weights.shape)
            else float("inf")
        ),
        "effective_rank": effective_rank(weights),
    }


def analyze_model(layers: dict) -> dict:
    """Report every layer of a model dict {name: 2-D matrix}."""
    if not layers:
        raise ValueError("layers must be non-empty")
    return {name: layer_report(w) for name, w in layers.items()}


def find_bottlenecks(reports: dict, top_k: int = 1) -> list:
    """Layer names with the smallest effective rank, worst first."""
    if not reports:
        return []
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ranked = sorted(reports.items(), key=lambda kv: (kv[1]["effective_rank"], kv[0]))
    return [name for name, _ in ranked[:top_k]]


def gradient_health(weights: np.ndarray) -> dict:
    """Spectral proxy for gradient-flow health of one layer.

    NOTE (honest scope): this is a static spectral proxy computed from
    the weight matrix alone - no actual gradients are tracked. The
    condition number of a layer bounds how unevenly gradients are
    scaled across input directions; very large condition numbers are
    associated with slow or unstable training. Treat the output as a
    screening signal, not a measurement.

    Returns dict with condition_number, effective_rank and a "risk"
    label: "low" (cond < 1e3), "medium" (< 1e6), "high" otherwise.
    """
    report = layer_report(weights)
    cond = report["condition_number"]
    if cond < 1e3:
        risk = "low"
    elif cond < 1e6:
        risk = "medium"
    else:
        risk = "high"
    return {
        "condition_number": cond,
        "effective_rank": report["effective_rank"],
        "risk": risk,
    }


def flow_report(layers: dict) -> dict:
    """Gradient-flow health for every layer of a {name: matrix} dict.

    Returns dict {layer_name: health dict} plus a "highest_risk" key
    naming the layer with the largest condition number.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    out = {name: gradient_health(w) for name, w in layers.items()}
    def _cond(entry):
        c = entry["condition_number"]
        return c if c != float("inf") else float("1e308")
    out["highest_risk"] = max(
        (n for n in layers), key=lambda n: _cond(out[n])
    )
    return out