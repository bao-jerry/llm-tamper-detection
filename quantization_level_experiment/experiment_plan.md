# Quantization-Level Experiment Plan

## Goal

Test whether the same-model estimator detects different loading or quantization
levels of the same Qwen2.5-14B-Instruct model as non-identical implementation
variants.

This supports the practical use case of undisclosed model modification
detection and hosted model integrity verification: a provider may claim to serve
the original open-source model while actually serving a quantized backend.

## Model Family

The experiment fixes the model identity as:

```text
Qwen/Qwen2.5-14B-Instruct
```

The compared load modes currently recorded are:

- `bf16`
- `8bit`
- `4bit`

`fp16` is still optional/pending.

## Target Tokens

The target token pair is:

| Role | Qwen token ID | Decoded string |
| --- | ---: | --- |
| `a` | `15` | `0` |
| `b` | `16` | `1` |

## Canary Prompts

The quantization experiment reuses the 14B canaries from the model re-routing
experiment:

| Context slot | Prompt |
| --- | ---: |
| `C_1` | Prompt 1 |
| `C_2 = C_3` | Prompt 4 |
| `C_4` | Prompt 5 |

The rationale is that the observed `0`/`1` balance remained feasible across the
recorded quantization settings, so no separate prompt search was needed.

## Heatmap Protocol

The theoretical-limit heatmap notebook computes 8 total `3 x 3` grids:

- 4 decoding conditions
- 2 naive estimators

Rows are requested load modes; columns are actual load modes.

For each cell, the notebook computes the theoretical limit of Naive Method I
and Naive Method II using:

```text
requested-load-mode logits on the 14B canaries
actual-load-mode decoding-altered probabilities on the same 14B canaries
```

Undefined cells are shown as `N/A` when the top-p cutoff removes one of the two
target tokens or when an estimator denominator vanishes.

## Expected Outcome

The diagonal cells should be near zero whenever defined. Off-diagonal cells
should be nonzero if quantization meaningfully changes the next-token behavior
on the chosen canaries.

If a quantized load mode remains near zero, that is also informative: either
the quantization is behaviorally close on these canaries, or the canaries are
not sensitive to that modification.
