# Model Re-Routing Experiment Materials

This folder contains the Qwen2.5 backend model re-routing proof-of-concept
materials.

The model re-routing protocol is requested-configuration-canary based:

```text
requested-configuration logits on requested-configuration canaries
actual-configuration sampling distributions on requested-configuration canaries
```

For each fixed requested model configuration, the requested model configuration supplies the canary
prompt slots `C_1`, `C_2 = C_3`, and `C_4`. Those same canary prompts are then
used for every actual model configuration.

Contents:

- `qwen2_5_colab_prompt_selection_model_routing_experiment.ipynb`: Colab notebook for
  selecting prompts and saving next-token logit payloads.
- `qwen2_5_estimator_convergence_model_routing_experiment.ipynb`: local estimator
  convergence notebook using the requested-configuration-canary protocol.
- `qwen2_5_estimator_convergence_7b_actual_config_model_routing_experiment.ipynb`: focused
  fixed-budget graph notebook for the cases where the `Qwen2.5-7B-Instruct`
  actual config is compared against the `Qwen2.5-14B-Instruct` and
  `Qwen2.5-32B-Instruct` requested configurations using 7B canaries.
- `logit_payloads/`: saved prompt-level next-token logit payloads.
- `context_token_tracker.md`: prompt/token/model tracking notes.
- `unique_prompt_inventory.md`: the six unique prompt strings.
- `estimator_formula_audit.tex`: exact formula audit for the implemented
  estimators.
- `experiment_plan.md`: model re-routing experiment protocol notes.

Defined contexts in the main graph notebook run for exactly `7,500` one-token
samples. The graph treats
the three context traces as parallel sequences: at parallel step `t`, the
estimator uses `g_1[t]`, `g_2[t]`, and `g_4[t]`. The plotted x-value is `3t`,
the total one-token call count across `C_1`, `C_2`, and `C_4`. Contexts where
the fixed decoding parameters make token ID `15` or token ID `16` unsampleable
are marked undefined.

In the main requested-configuration-canary notebook, the requested model configuration supplies canaries.
Under this protocol, the `Qwen2.5-7B-Instruct` actual-config curve is undefined for
the `Qwen2.5-14B-Instruct` and `Qwen2.5-32B-Instruct` requested configs because the
fixed decoding/top-p setup removes one target token on those requested-configuration
canaries. The focused 7B actual-config notebook is a prompt-engineering workaround:
it keeps the same fixed decoding setup but uses the 7B canaries for both the
requested-config curve and the 7B actual-config curve. This produces visual comparison
graphs for the 14B requested config vs 7B actual config and the 32B requested config vs 7B actual config
without changing the model payloads or decoding seed. In that focused notebook,
the 14B requested config uses `7,500` samples per context and the 32B requested config uses
`15,000` samples per context.

Loaded payload logits are cast to `np.float64` immediately after loading. This
does not recover precision beyond what was saved in the `.pt` payloads, but it
prevents additional notebook-side precision loss during diagnostics.
