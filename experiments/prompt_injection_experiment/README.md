# Prompt-Injection Experiment

This folder contains the prompt-injection / prompt-formatting follow-up
experiment for the Qwen2.5-14B same-model detection project.

The experiment fixes the underlying checkpoint and loading mode, then changes
only the prompt-formatting variant. This tests whether the estimator can flag an
endpoint as non-identical when the provider silently injects or modifies actual
prompting before the user context.

## Files

- `qwen2_5_colab_prompt_selection_prompt_injection_experiment.ipynb`
  generates the logit payloads. One successful run creates 18 payloads total:
  3 user-defined prompt-formatting variants times 6 prompts.
- `qwen2_5_theoretical_limit_heatmaps_prompt_injection_experiment.ipynb`
  computes theoretical Naive Method I and Naive Method II limits and plots
  heatmaps.
- `context_token_tracker.md` records the prompt inventory, canary mapping, and
  prompt-formatting variant placeholders.
- `experiment_plan.md` records the intended protocol.
- `logit_payloads/` is where the `.pt` payloads should live after generation.

## User-Controlled Variants

The notebook initializes all three variants as byte-for-byte identical baseline
prompt-formatting copies: each variant has `system_prompt = None` and reuses the
same six prompt strings. Edit `PROMPT_FORMATTING_VARIANTS` in the
prompt-selection notebook when you want to introduce actual prompt-injection
variants. Keep exactly three keys for the downstream `3 x 3` heatmap.

Each variant has:

- a stable key used in filenames
- a short label used for plots
- a `system_prompt`, where `None` means no explicit extra system message

## Expected Result

The diagonal compares each prompt-formatting variant against itself and should
be near zero whenever the estimator is defined. Off-diagonal entries compare
different prompt-formatting variants and should move away from zero if the
prompt injection changes the relevant next-token geometry.
