"""Helpers for saved next-token logit payloads.

The Colab notebook saves each logit payload as a PyTorch ``.pt`` file containing
a dictionary. The key used by these helpers is:

    payload["logits"]

In the current notebook, ``payload["logits"]`` is a one-dimensional CPU
``torch.Tensor`` with dtype ``torch.float32`` and shape ``[vocab_size]``. For
Qwen2.5 runs in this project, ``vocab_size`` is currently 152064.

This module operates on NumPy arrays rather than importing PyTorch. To use a
saved payload, load it with ``torch.load(...)`` and convert the tensor:

    logits = payload["logits"].detach().cpu().numpy()

The probability helpers immediately copy logits into ``np.float64`` before
doing any numerical work. This preserves as much precision as is available from
the saved float32 payload and avoids doing softmax/top-p calculations in lower
precision.

The intended workflow is:

    1. load a saved ``.pt`` payload with PyTorch;
    2. convert ``payload["logits"]`` to a NumPy array;
    3. call ``make_probability_vector`` with decoding parameters;
    4. call ``make_cumulative_probability_vector`` once with interest tokens;
    5. call ``sample_from_cumulative_probability_vector`` repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np


RANDOM_DECODING_TEMPERATURE_RANGE: tuple[float, float] = (0.85, 1.10)
RANDOM_DECODING_TOP_P_RANGE: tuple[float, float] = (0.80, 0.95)
RANDOM_DECODING_LOGIT_BIAS_RANGE: tuple[int, int] = (-2, 2)
DEFAULT_RANDOM_DECODING_BIAS_TOKENS: tuple[int, int] = (15, 16)
DEFAULT_INTEREST_TOKENS: tuple[int, int] = (15, 16)
OTHER_TOKEN_ID: int = -1


@dataclass(frozen=True)
class RandomDecodingParams:
    """A reproducible random-decoding configuration.

    Attributes
    ----------
    temperature:
        Positive temperature sampled uniformly from the hard-coded experiment
        range ``[0.85, 1.10]``.
    top_p:
        Top-p threshold sampled uniformly from the hard-coded experiment range
        ``[0.80, 0.95]``.
    logit_bias:
        Mapping from token id to additive integer logit bias. Each bias is
        sampled uniformly from the hard-coded inclusive integer range
        ``{-2, -1, 0, 1, 2}``.
    seed:
        Seed used to generate this parameter object. Storing it makes plots and
        runs easier to audit later.
    """

    temperature: float
    top_p: float
    logit_bias: dict[int, int]
    seed: int


@dataclass(frozen=True)
class CumulativeProbabilityVector:
    """Collapsed cumulative distribution for fast repeated token sampling.

    Attributes
    ----------
    cumulative_probs:
        One-dimensional ``np.float64`` CDF over collapsed categories.
    categories:
        Category labels corresponding to entries in ``cumulative_probs``. The
        labels are the requested interest token ids followed by
        ``OTHER_TOKEN_ID``. A sample from this object returns one of these
        labels, not a raw category position.
    """

    cumulative_probs: np.ndarray
    categories: tuple[int, ...]


def make_random_decoding_params(
    seed: int = 0,
    tokens_to_bias: Sequence[int] = DEFAULT_RANDOM_DECODING_BIAS_TOKENS,
) -> RandomDecodingParams:
    """Generate one deterministic pseudorandom decoding configuration.

    This helper should be called once per estimator trajectory when the same
    decoding parameters are meant to be reused across all contexts. The only
    random quantities are temperature, top-p, and one integer logit bias for
    each token id in ``tokens_to_bias``. The allowed ranges are intentionally
    hard-coded as part of the experimental protocol:

    - temperature is sampled uniformly from ``[0.85, 1.10]``;
    - top-p is sampled uniformly from ``[0.80, 0.95]``;
    - each listed token receives an integer logit bias sampled uniformly from
      ``{-2, -1, 0, 1, 2}``.

    Parameters
    ----------
    tokens_to_bias:
        Token ids that should receive sampled logit biases. For the current
        Qwen2.5 binary-token experiment, this defaults to ``(15, 16)``. Token
        ID 15 decodes to ``"0"`` and token ID 16 decodes to ``"1"``. Duplicate
        token ids are allowed in the input but are collapsed so each token id
        receives exactly one sampled bias.
    seed:
        Seed for NumPy's ``default_rng``. Passing the same seed and the same
        ``tokens_to_bias`` sequence produces the same output.

    Returns
    -------
    RandomDecodingParams
        A frozen parameter object that can be passed to
        ``make_random_decoding_probability_vector`` for every context logit
        vector in the same experiment run.
    """

    token_ids: list[int] = []
    seen: set[int] = set()
    for token_id in tokens_to_bias:
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError("tokens_to_bias must contain nonnegative token ids")
        if token_id not in seen:
            token_ids.append(token_id)
            seen.add(token_id)

    rng = np.random.default_rng(seed)

    temperature_low, temperature_high = RANDOM_DECODING_TEMPERATURE_RANGE
    top_p_low, top_p_high = RANDOM_DECODING_TOP_P_RANGE
    bias_low, bias_high = RANDOM_DECODING_LOGIT_BIAS_RANGE

    temperature = float(rng.uniform(temperature_low, temperature_high))
    top_p = float(rng.uniform(top_p_low, top_p_high))
    logit_bias = {
        token_id: int(rng.integers(bias_low, bias_high + 1)) for token_id in token_ids
    }

    return RandomDecodingParams(
        temperature=temperature,
        top_p=top_p,
        logit_bias=logit_bias,
        seed=int(seed),
    )


def make_random_decoding_probability_vector(
    logits: np.ndarray,
    params: RandomDecodingParams,
) -> np.ndarray:
    """Create a probability vector using generated random-decoding parameters.

    This is a thin, audit-friendly wrapper around ``make_probability_vector``.
    It consumes a ``RandomDecodingParams`` object generated once by
    ``make_random_decoding_params`` and applies that exact same configuration to
    the provided logit vector.

    Parameters
    ----------
    logits:
        One-dimensional next-token logit vector. As with
        ``make_probability_vector``, saved payload logits should be converted to
        a NumPy array before calling this function.
    params:
        Random decoding parameters generated by
        ``make_random_decoding_params``. Reuse the same ``params`` object across
        contexts when the experiment should use the same decoding layer for all
        contexts.

    Returns
    -------
    np.ndarray
        A one-dimensional ``np.float64`` probability vector with the same length
        as ``logits``.
    """

    return make_probability_vector(
        logits,
        temperature=params.temperature,
        logit_bias=params.logit_bias,
        top_p=params.top_p,
    )


def make_probability_vector(
    logits: np.ndarray,
    temperature: float = 1.0,
    logit_bias: Mapping[int, float] | None = None,
    top_p: float = 1.0,
) -> np.ndarray:
    """Create a numerically stable probability vector from a logit vector.

    This function simulates the probability-distribution step of a decoder for
    one saved next-token logit vector. It does not call an LLM and it does not
    generate text. The returned array is indexed by token id: ``probs[15]`` is
    the probability assigned to token id 15, ``probs[16]`` is the probability
    assigned to token id 16, and so on.

    The transformation order is deliberately fixed:

    1. copy logits into ``np.float64``;
    2. apply additive logit bias to selected token ids;
    3. divide all logits by temperature;
    4. compute a max-shifted softmax;
    5. optionally apply top-p filtering and renormalize.

    Parameters
    ----------
    logits:
        One-dimensional next-token logits. In this project these usually come
        from ``payload["logits"]`` in a saved ``.pt`` payload. The saved payload
        logits are CPU ``torch.float32`` tensors; pass them to this function as a
        NumPy array, for example ``payload["logits"].detach().cpu().numpy()``.
        The function copies the input into ``np.float64`` before computation.
    temperature:
        Positive temperature. Values below 1 sharpen the distribution; values
        above 1 flatten it. This function rejects non-positive temperatures
        instead of silently switching to greedy decoding.
    logit_bias:
        Optional mapping from token id to additive logit bias. Biases are applied
        before temperature scaling. Token ids must be valid indices into
        ``logits``. Bias values must be finite real numbers.
    top_p:
        Nucleus sampling threshold in ``(0, 1]``. If less than 1, the function
        keeps the smallest highest-probability token set whose cumulative mass is
        at least ``top_p``, then renormalizes.

    Returns
    -------
    np.ndarray
        A one-dimensional ``np.float64`` probability vector with the same length
        as ``logits``.

    Notes
    -----
    Numerical best practices used here:

    - all probability computation is performed in ``np.float64``;
    - softmax uses max-shifting to avoid overflow;
    - sums/cumulative sums are explicitly accumulated in ``np.float64``;
    - inputs are validated for dimensionality, finiteness, and nonzero support;
    - top-p filtering keeps the boundary token that crosses the threshold.

    Example
    -------
    ``logits`` is usually obtained from a saved payload:

    >>> logits = payload["logits"].detach().cpu().numpy()
    >>> probs = make_probability_vector(
    ...     logits,
    ...     temperature=1.0,
    ...     logit_bias={15: 1.0, 16: -1.0},
    ...     top_p=1.0,
    ... )
    >>> probs[15], probs[16]
    """
    logits64 = np.asarray(logits, dtype=np.float64)
    if logits64.ndim != 1:
        raise ValueError("logits must be a one-dimensional array")
    if logits64.size == 0:
        raise ValueError("logits must not be empty")

    logits64 = logits64.copy()
    if not np.all(np.isfinite(logits64)):
        raise ValueError("logits must contain only finite values")

    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite positive number")

    top_p = float(top_p)
    if not np.isfinite(top_p) or not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be a finite number in (0, 1]")

    if logit_bias:
        vocab_size = logits64.shape[0]
        for token_id, bias in logit_bias.items():
            token_id = int(token_id)
            if token_id < 0 or token_id >= vocab_size:
                raise IndexError(
                    f"logit_bias token id {token_id} is outside [0, {vocab_size})"
                )

            bias = float(bias)
            if not np.isfinite(bias):
                raise ValueError("logit_bias values must be finite numbers")
            logits64[token_id] += bias

    scaled_logits = logits64 / temperature

    # Stable softmax: subtract the maximum before exponentiating.
    max_logit = np.max(scaled_logits)
    shifted_logits = scaled_logits - max_logit
    exp_shifted = np.exp(shifted_logits, dtype=np.float64)
    total = np.sum(exp_shifted, dtype=np.float64)
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("softmax normalization failed")

    probs = exp_shifted / total

    if top_p < 1.0:
        # Stable descending sort by probability. A stable sort gives deterministic
        # behavior for exact finite-precision ties.
        sorted_ids = np.argsort(-probs, kind="stable")
        sorted_probs = probs[sorted_ids]
        cumulative = np.cumsum(sorted_probs, dtype=np.float64)

        keep_count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        keep_count = min(max(keep_count, 1), probs.size)

        filtered = np.zeros_like(probs, dtype=np.float64)
        keep_ids = sorted_ids[:keep_count]
        filtered[keep_ids] = probs[keep_ids]

        filtered_total = np.sum(filtered, dtype=np.float64)
        if not np.isfinite(filtered_total) or filtered_total <= 0.0:
            raise FloatingPointError("top-p filtering removed all probability mass")
        probs = filtered / filtered_total

    prob_sum = np.sum(probs, dtype=np.float64)
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise FloatingPointError("probability vector normalization failed")

    # Final renormalization absorbs tiny roundoff from softmax/top-p operations.
    probs = probs / prob_sum
    return probs


def make_cumulative_probability_vector(
    probs: np.ndarray,
    interest_tokens: Sequence[int] = DEFAULT_INTEREST_TOKENS,
) -> CumulativeProbabilityVector:
    """Precompute a collapsed cumulative probability vector.

    This helper is the preferred bridge between probability construction and
    large-scale simulation. Build ``probs`` once with ``make_probability_vector``
    or ``make_random_decoding_probability_vector``, then call this function once
    and reuse the returned object for all samples from that same distribution.

    The full vocabulary distribution is collapsed into the requested
    ``interest_tokens`` plus one complement category, ``OTHER_TOKEN_ID``. For
    example, if ``interest_tokens=(15, 16)``, the categories are:

    - token id 15;
    - token id 16;
    - ``OTHER_TOKEN_ID``, meaning any token other than 15 or 16.

    This is both faster and cleaner for estimator simulations that only care
    about counts for a small set of target tokens.

    Parameters
    ----------
    probs:
        One-dimensional probability vector. It may contain tiny floating-point
        normalization drift; this function renormalizes it in ``np.float64``.
    interest_tokens:
        Token ids to keep as explicit categories. Duplicate ids are collapsed
        while preserving first occurrence order. All remaining probability mass
        is grouped into the final ``OTHER_TOKEN_ID`` category.

    Returns
    -------
    CumulativeProbabilityVector
        A collapsed cumulative distribution. Its category labels are the
        interest token ids followed by ``OTHER_TOKEN_ID``. The final CDF entry is
        forced to exactly ``1.0`` so random values near one cannot fall off the
        end due to roundoff.

    Notes
    -----
    Numerical care:

    - probabilities are copied to ``np.float64``;
    - total mass and category masses are summed in ``np.float64``;
    - the OTHER mass is computed by directly summing the complement of the
      interest-token mask, avoiding cancellation from ``1 - selected_mass``;
    - the collapsed category vector is renormalized in ``np.float64``;
    - the final CDF entry is forced to exactly ``1.0``.

    Example
    -------
    >>> probs = make_random_decoding_probability_vector(logits, params)
    >>> cumulative = make_cumulative_probability_vector(probs, interest_tokens=(15, 16))
    >>> token_id = sample_from_cumulative_probability_vector(cumulative, rng)
    """

    probs64 = np.asarray(probs, dtype=np.float64)
    if probs64.ndim != 1:
        raise ValueError("probs must be a one-dimensional array")
    if probs64.size == 0:
        raise ValueError("probs must not be empty")
    if np.any(probs64 < 0.0) or not np.all(np.isfinite(probs64)):
        raise ValueError("probs must contain only finite nonnegative values")

    total = np.sum(probs64, dtype=np.float64)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("probs must have positive finite total mass")

    token_ids: list[int] = []
    seen: set[int] = set()
    vocab_size = probs64.shape[0]
    for token_id in interest_tokens:
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            raise IndexError(
                f"interest token id {token_id} is outside [0, {vocab_size})"
            )
        if token_id not in seen:
            token_ids.append(token_id)
            seen.add(token_id)

    interest_mask = np.zeros(vocab_size, dtype=bool)
    if token_ids:
        interest_mask[np.asarray(token_ids, dtype=np.int64)] = True
        interest_masses = probs64[np.asarray(token_ids, dtype=np.int64)]
    else:
        interest_masses = np.empty(0, dtype=np.float64)

    other_mass = np.sum(probs64[~interest_mask], dtype=np.float64)
    if not np.isfinite(other_mass) or other_mass < 0.0:
        raise FloatingPointError("failed to compute OTHER probability mass")

    category_masses = np.concatenate(
        [interest_masses, np.asarray([other_mass], dtype=np.float64)]
    )
    category_total = np.sum(category_masses, dtype=np.float64)
    if not np.isfinite(category_total) or category_total <= 0.0:
        raise FloatingPointError("collapsed probability normalization failed")

    category_probs = category_masses / category_total
    cumulative = np.cumsum(category_probs, dtype=np.float64)
    cumulative[-1] = 1.0
    cumulative.setflags(write=False)

    return CumulativeProbabilityVector(
        cumulative_probs=cumulative,
        categories=tuple(token_ids) + (OTHER_TOKEN_ID,),
    )


def sample_from_cumulative_probability_vector(
    cumulative_probs: CumulativeProbabilityVector,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one category from a precomputed cumulative probability vector.

    This is faster than sampling directly from the full probability vector when
    the same distribution is sampled repeatedly, because the distribution is
    collapsed to interest tokens plus OTHER and the cumulative distribution is
    computed once and reused.

    Parameters
    ----------
    cumulative_probs:
        Output from ``make_cumulative_probability_vector``.
    rng:
        Optional NumPy random generator. Pass an explicit generator for
        reproducible experiments.

    Returns
    -------
    int
        The sampled category label: either one of the interest token ids or
        ``OTHER_TOKEN_ID``.

    Notes
    -----
    This function assumes ``cumulative_probs`` was produced by
    ``make_cumulative_probability_vector`` and has not been manually mutated.
    Validation is intentionally performed during CDF construction rather than on
    every sample, keeping repeated one-token sampling fast.
    """

    rng = rng if rng is not None else np.random.default_rng()
    draw = float(rng.random())
    category_index = int(
        np.searchsorted(cumulative_probs.cumulative_probs, draw, side="right")
    )
    category_index = min(category_index, len(cumulative_probs.categories) - 1)
    return int(cumulative_probs.categories[category_index])
