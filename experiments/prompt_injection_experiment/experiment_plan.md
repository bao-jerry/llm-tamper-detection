# Prompt-Injection Experiment Plan

## Objective

Test whether the estimator detects actual prompt-injection / prompt-formatting
changes as non-identical hosted implementations, even when the underlying model
weights are the same.

## Fixed Model

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Load mode: `8bit` by default, held fixed throughout this experiment.
- Target tokens:
  - token `15`: string `0`
  - token `16`: string `1`

## Prompt-Formatting Variants

The prompt-selection notebook intentionally leaves the three variants under
user control. It initializes all three variants as identical baseline
prompt-formatting copies with `system_prompt = None`. Define exactly three
entries in `PROMPT_FORMATTING_VARIANTS`.

Each entry should have:

```python
"variant_key": {
    "label": "Short plot label",
    "system_prompt": None,  # or a actual prompt-injection string
}
```

The prompt-selection notebook should load the model once and save 18 logit
payloads total:

```text
3 prompt-formatting variants x 6 prompts = 18 logit payloads
```

## Canary Mapping

Use the same 14B canary mapping as the model re-routing experiment:

```text
C1 = prompt 1
C2 = C3 = prompt 4
C4 = prompt 5
```

## Theoretical-Limit Heatmaps

The heatmap notebook treats prompt-formatting variants as the compared
implementations:

- rows: requested prompt-formatting variant `A`
- columns: actual prompt-formatting variant `B`

For each decoding condition and each naive estimator, compute a `3 x 3`
matrix. The diagonal should be near zero whenever defined. Off-diagonal entries
indicate that the actual prompt wrapper changes the effective next-token
distribution.

## Decoding Conditions

Use the same four deterministic decoding conditions as the other follow-up
experiments:

- two random `(temperature, top_p, logit_bias)` triplets
- two random `(temperature, logit_bias)` pairs with `top_p = 1`

The seed values are written directly in the heatmap notebook for readability.
