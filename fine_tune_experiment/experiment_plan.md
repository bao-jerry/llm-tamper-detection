# Fine-Tune / Specialized Variant Experiment Plan

## Goal

Test whether the same-model estimator detects fine-tuned or specialized
Qwen2.5-14B variants as distinct from the base Qwen2.5-14B-Instruct model.

This checks whether the estimator is sensitive to exact functional
implementation details rather than merely shared architecture, tokenizer, model
family, or broad lineage.

## Model Set

The current recorded model set is:

| Model key | Model ID |
| --- | --- |
| `14b_instruct` | `Qwen/Qwen2.5-14B-Instruct` |
| `14b_instruct_1m` | `Qwen/Qwen2.5-14B-Instruct-1M` |
| `14b_coder_instruct` | `Qwen/Qwen2.5-Coder-14B-Instruct` |

All recorded payloads use the `8bit` load mode.

## Target Tokens

The target token pair is:

| Role | Qwen token ID | Decoded string |
| --- | ---: | --- |
| `a` | `15` | `0` |
| `b` | `16` | `1` |

## Canary Prompts

The fine-tune experiment uses the same 14B canary slots as the primary
experiment:

| Context slot | Prompt |
| --- | ---: |
| `C_1` | Prompt 1 |
| `C_2 = C_3` | Prompt 4 |
| `C_4` | Prompt 5 |

Some fine-tuned variants are much more imbalanced on these prompts. When top-p
removes one of the target tokens, the corresponding theoretical-limit cell is
shown as `N/A`.

## Heatmap Protocol

The theoretical-limit heatmap notebook computes 8 total `3 x 3` grids:

- 4 decoding conditions
- 2 naive estimators

Rows are reference models `A`; columns are suspect models `B`.

For each cell, the notebook computes the theoretical limit of Naive Method I
and Naive Method II using:

```text
A logits on the 14B canaries
B decoding-altered probabilities on the same 14B canaries
```

Undefined cells are shown as `N/A` when the top-p cutoff removes one of the two
target tokens or when an estimator denominator vanishes.

## Expected Outcome

The diagonal cells should be near zero whenever defined. Off-diagonal cells
should be nonzero when fine-tuning or specialization changes next-token behavior
on the selected canaries.

The observed results are expected to be sensitive: the base instruct model,
the long-context 1M derivative, and the coder-specialized derivative can have
substantially different theoretical estimator limits even though they share the
Qwen2.5 family and tokenizer.
