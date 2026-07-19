"""Estimator helpers for the empirical naive-method experiments.

The formulas implemented here correspond to
``primary_experiment/estimator_formula_audit.tex``.

The expected logit dictionary keys are literal formula subscripts:

    z[(j, t)] = reference-model base logit z_{j,t}

where ``j`` is the context slot in ``{1, 2, 4}`` and ``t`` is the token
subscript in ``{0, 1}``. In the current experiment, token subscript ``0`` is
the formula's ``a`` token and token subscript ``1`` is the formula's ``b``
token. By default, ``a`` is Qwen token ID 15 (string "0") and ``b`` is Qwen
token ID 16 (string "1").

The estimator functions consume context-level ``g`` values:

    g[j] = log(n_{j,a}) - log(n_{j,b})

where the count-level work happens upstream in ``sample_g_until_convergence``.
The naive estimator functions therefore take ``z`` and ``g``, not raw count
dictionaries.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from logit_helpers import CumulativeProbabilityVector
from logit_helpers import sample_from_cumulative_probability_vector


Subscript = tuple[int, int]
GSubscript = int

REQUIRED_LOGIT_SUBSCRIPTS: tuple[Subscript, ...] = (
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (4, 0),
    (4, 1),
)
REQUIRED_G_SUBSCRIPTS: tuple[GSubscript, ...] = (1, 2, 4)

DEFAULT_TOKEN_A: int = 15
DEFAULT_TOKEN_B: int = 16
DEFAULT_G_CONVERGENCE_TOLERANCE: float = 1e-2
DEFAULT_G_CONVERGENCE_WINDOW_SIZE: int = 200
DEFAULT_MAX_G_SAMPLES: int = 10_000_000


def sample_g_until_convergence(
    cumulative_probs: CumulativeProbabilityVector,
    rng: np.random.Generator | None = None,
    tolerance: float = DEFAULT_G_CONVERGENCE_TOLERANCE,
    window_size: int = DEFAULT_G_CONVERGENCE_WINDOW_SIZE,
    token_a: int = DEFAULT_TOKEN_A,
    token_b: int = DEFAULT_TOKEN_B,
    max_samples: int = DEFAULT_MAX_G_SAMPLES,
) -> list[float]:
    """Sample one context until its empirical ``g`` sequence stabilizes.

    This function is used during empirical run allocation. For one fixed context
    ``C_j``, it repeatedly samples from a precomputed collapsed CDF and tracks
    the sequence

        ``g_j = log(n_{j,a}) - log(n_{j,b})``.

    The returned list contains the finite ``g_j`` values observed up to the
    stopping point. Values are appended after every sample once both target-token
    counts are positive. If an OTHER category is sampled after that point, the
    counts do not change and the newest ``g_j`` value repeats; this preserves the
    per-sample convergence trace for later interleaving and plotting.

    Parameters
    ----------
    cumulative_probs:
        Collapsed CDF produced by ``logit_helpers.make_cumulative_probability_vector``.
        It must include ``token_a`` and ``token_b`` as explicit interest-token
        categories. Any non-target category, usually ``OTHER_TOKEN_ID``, is
        ignored for the counts.
    rng:
        Optional NumPy random generator. Pass a seeded generator such as
        ``np.random.default_rng(123)`` for reproducible trajectories. If omitted,
        a fresh default generator is created.
    tolerance:
        Rolling beam-width tolerance. Convergence is declared when the maximum
        minus minimum of the last ``window_size`` finite ``g_j`` values is at
        most this value.
    window_size:
        Number of most recent finite ``g_j`` values used for the rolling
        beam-width convergence check.
    token_a:
        Token id counted as the numerator token in ``log(n_a) - log(n_b)``.
        Defaults to Qwen token ID 15, decoded as ``"0"``.
    token_b:
        Token id counted as the denominator token in ``log(n_a) - log(n_b)``.
        Defaults to Qwen token ID 16, decoded as ``"1"``.
    max_samples:
        Hard cap on total samples drawn from ``cumulative_probs``. This prevents
        infinite loops if a context never reaches the stabilization criterion.

    Returns
    -------
    list[float]
        Finite ``g_j`` values up to convergence or until ``max_samples`` is hit.
        If one target token is never sampled, the list may be empty.
    """

    token_a = int(token_a)
    token_b = int(token_b)
    if token_a == token_b:
        raise ValueError("token_a and token_b must be distinct token ids")

    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite nonnegative number")

    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    max_samples = int(max_samples)
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative")

    categories = set(cumulative_probs.categories)
    if token_a not in categories or token_b not in categories:
        raise ValueError(
            "cumulative_probs must include token_a and token_b as explicit "
            "interest-token categories"
        )

    rng = rng if rng is not None else np.random.default_rng()
    count_a = 0
    count_b = 0
    g_values: list[float] = []

    for _ in range(max_samples):
        sampled_token = sample_from_cumulative_probability_vector(cumulative_probs, rng)
        if sampled_token == token_a:
            count_a += 1
        elif sampled_token == token_b:
            count_b += 1

        if count_a == 0 or count_b == 0:
            continue

        g_value = math.log(count_a) - math.log(count_b)
        g_values.append(g_value)

        if len(g_values) >= window_size:
            recent_values = g_values[-window_size:]
            beam_width = max(recent_values) - min(recent_values)
            if beam_width <= tolerance:
                break

    return g_values


def sample_g_for_fixed_iterations(
    cumulative_probs: CumulativeProbabilityVector,
    num_samples: int,
    rng: np.random.Generator | None = None,
    token_a: int = DEFAULT_TOKEN_A,
    token_b: int = DEFAULT_TOKEN_B,
) -> list[float]:
    """Sample one context for exactly ``num_samples`` one-token calls.

    This helper is intended for graphing variants where every context should
    run for the same fixed budget rather than stopping when a convergence rule
    fires. For one fixed context ``C_j``, it repeatedly samples from a
    precomputed collapsed CDF and tracks

        ``g_j = log(n_{j,a}) - log(n_{j,b})``.

    The returned list has length exactly ``num_samples``. Before both target
    token counts are positive, the corresponding entries are ``np.nan``. Once
    both counts are positive, every subsequent sample contributes one entry:
    either an updated ``g_j`` value if token ``a`` or token ``b`` was sampled,
    or the repeated current ``g_j`` value if the OTHER category was sampled.

    Parameters
    ----------
    cumulative_probs:
        Collapsed CDF produced by
        ``logit_helpers.make_cumulative_probability_vector``. It must include
        ``token_a`` and ``token_b`` as explicit interest-token categories.
    num_samples:
        Exact number of samples to draw from ``cumulative_probs``.
    rng:
        Optional NumPy random generator. Pass a seeded generator such as
        ``np.random.default_rng(123)`` for reproducible trajectories.
    token_a:
        Token id counted as the numerator token in ``log(n_a) - log(n_b)``.
        Defaults to Qwen token ID 15, decoded as ``"0"``.
    token_b:
        Token id counted as the denominator token in ``log(n_a) - log(n_b)``.
        Defaults to Qwen token ID 16, decoded as ``"1"``.

    Returns
    -------
    list[float]
        A per-sample ``g_j`` trace of length exactly ``num_samples``. Entries
        are ``np.nan`` until both target-token counts are nonzero.
    """

    token_a = int(token_a)
    token_b = int(token_b)
    if token_a == token_b:
        raise ValueError("token_a and token_b must be distinct token ids")

    num_samples = int(num_samples)
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative")

    categories = set(cumulative_probs.categories)
    if token_a not in categories or token_b not in categories:
        raise ValueError(
            "cumulative_probs must include token_a and token_b as explicit "
            "interest-token categories"
        )

    rng = rng if rng is not None else np.random.default_rng()
    count_a = 0
    count_b = 0
    g_values: list[float] = []

    for _ in range(num_samples):
        sampled_token = sample_from_cumulative_probability_vector(cumulative_probs, rng)
        if sampled_token == token_a:
            count_a += 1
        elif sampled_token == token_b:
            count_b += 1

        if count_a == 0 or count_b == 0:
            g_values.append(np.nan)
        else:
            g_values.append(math.log(count_a) - math.log(count_b))

    return g_values


def _get_float(mapping: Mapping[Subscript, float], key: Subscript) -> float:
    """Return a finite float value for ``key``, or ``np.nan`` if unavailable."""

    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError):
        return np.nan

    if not math.isfinite(value):
        return np.nan
    return value


def _get_g_value(mapping: Mapping[GSubscript, float], slot: int) -> float:
    """Return a finite g value for ``slot``, or ``np.nan`` if unavailable."""

    try:
        value = float(mapping[slot])
    except (KeyError, TypeError, ValueError):
        return np.nan

    if not math.isfinite(value):
        return np.nan
    return value


def _logit_difference(z: Mapping[Subscript, float], slot: int) -> float:
    """Return z_{slot,0} - z_{slot,1}, or ``np.nan`` if unavailable."""

    logit_0 = _get_float(z, (slot, 0))
    logit_1 = _get_float(z, (slot, 1))
    if math.isnan(logit_0) or math.isnan(logit_1):
        return np.nan
    return logit_0 - logit_1


def naive_method_i_estimator(
    z: Mapping[Subscript, float],
    g: Mapping[GSubscript, float],
) -> float:
    """Compute the empirical Naive Method I estimator.

    Parameters
    ----------
    z:
        Dictionary-like object keyed by ``(context_slot, token_subscript)``.
        It must provide the owned/reference model logits:
        ``z[(1, 0)]``, ``z[(1, 1)]``, ``z[(2, 0)]``, ``z[(2, 1)]``,
        ``z[(4, 0)]``, and ``z[(4, 1)]``.

    g:
        Dictionary-like object keyed by context slot. It must provide
        ``g[1]``, ``g[2]``, and ``g[4]``, where each value is already computed
        as ``log(n_{j,a}) - log(n_{j,b})`` from candidate-model samples.

    Returns
    -------
    float
        The estimator value, or ``np.nan`` if it is not currently computable.
        ``np.nan`` is returned for missing inputs, non-finite logits, non-finite
        ``g`` values, or zero denominators. This lets Matplotlib naturally skip
        uncomputable points in convergence plots.

    Runtime formula
    ---------------
    This implements the audit formula after the count-level expressions have
    already been compressed into ``g[1]``, ``g[2]``, and ``g[4]``:

    ((z_10 - z_11) - (z_20 - z_21)) / (g_1 - g_2)

    minus

    ((z_20 - z_21) - (z_40 - z_41)) / (g_2 - g_4).
    """

    z_1 = _logit_difference(z, 1)
    z_2 = _logit_difference(z, 2)
    z_4 = _logit_difference(z, 4)
    g_1 = _get_g_value(g, 1)
    g_2 = _get_g_value(g, 2)
    g_4 = _get_g_value(g, 4)

    if any(math.isnan(value) for value in (z_1, z_2, z_4, g_1, g_2, g_4)):
        return np.nan

    first_denominator = g_1 - g_2
    second_denominator = g_2 - g_4
    if first_denominator == 0.0 or second_denominator == 0.0:
        return np.nan

    return ((z_1 - z_2) / first_denominator) - ((z_2 - z_4) / second_denominator)


def naive_method_ii_estimator(
    z: Mapping[Subscript, float],
    g: Mapping[GSubscript, float],
) -> float:
    """Compute the empirical Naive Method II estimator.

    Parameters
    ----------
    z:
        Dictionary-like object keyed by ``(context_slot, token_subscript)``.
        It must provide the owned/reference model logits:
        ``z[(1, 0)]``, ``z[(1, 1)]``, ``z[(2, 0)]``, ``z[(2, 1)]``,
        ``z[(4, 0)]``, and ``z[(4, 1)]``.

    g:
        Dictionary-like object keyed by context slot. It must provide
        ``g[1]``, ``g[2]``, and ``g[4]``, where each value is already computed
        as ``log(n_{j,a}) - log(n_{j,b})`` from candidate-model samples.

    Returns
    -------
    float
        The estimator value, or ``np.nan`` if it is not currently computable.
        ``np.nan`` is returned for missing inputs, non-finite logits, non-finite
        ``g`` values, or zero denominators. This lets Matplotlib naturally skip
        uncomputable points in convergence plots.

    Runtime formula
    ---------------
    This implements the audit formula after the count-level expressions have
    already been compressed into ``g[1]``, ``g[2]``, and ``g[4]``:

    (g_2(z_10 - z_11) - g_1(z_20 - z_21)) / (g_1 - g_2)

    minus

    (g_4(z_20 - z_21) - g_2(z_40 - z_41)) / (g_2 - g_4).
    """

    z_1 = _logit_difference(z, 1)
    z_2 = _logit_difference(z, 2)
    z_4 = _logit_difference(z, 4)
    g_1 = _get_g_value(g, 1)
    g_2 = _get_g_value(g, 2)
    g_4 = _get_g_value(g, 4)

    if any(math.isnan(value) for value in (z_1, z_2, z_4, g_1, g_2, g_4)):
        return np.nan

    first_denominator = g_1 - g_2
    second_denominator = g_2 - g_4
    if first_denominator == 0.0 or second_denominator == 0.0:
        return np.nan

    first_numerator = (g_2 * z_1) - (g_1 * z_2)
    second_numerator = (g_4 * z_2) - (g_2 * z_4)

    return (first_numerator / first_denominator) - (
        second_numerator / second_denominator
    )
