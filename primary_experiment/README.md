# Primary Experiment Materials

This folder contains the primary Qwen2.5 same-model detection proof-of-concept
materials.

The primary protocol is reference-canary based:

```text
reference-model logits on reference-model canaries
suspect-model sampling distributions on reference-model canaries
```

For each fixed reference model, the reference model supplies the canary
prompt slots `C_1`, `C_2 = C_3`, and `C_4`. Those same canary prompts are then
used for every suspect model.

Contents:

- `qwen2_5_colab_prompt_selection_primary_experiment.ipynb`: Colab notebook for
  selecting prompts and saving next-token logit payloads.
- `qwen2_5_estimator_convergence_primary_experiment.ipynb`: local estimator
  convergence notebook using the reference-canary protocol.
- `qwen2_5_estimator_convergence_7b_suspect_primary_experiment.ipynb`: focused
  fixed-budget graph notebook for the cases where the `Qwen2.5-7B-Instruct`
  suspect is compared against the `Qwen2.5-14B-Instruct` and
  `Qwen2.5-32B-Instruct` references using 7B canaries.
- `logit_payloads/`: saved prompt-level next-token logit payloads.
- `context_token_tracker.md`: prompt/token/model tracking notes.
- `unique_prompt_inventory.md`: the six unique prompt strings.
- `estimator_formula_audit.tex`: exact formula audit for the implemented
  estimators.
- `experiment_plan.md`: primary experiment protocol notes.

Defined contexts in the main graph notebook run for exactly `7,500` one-token
samples. The graph treats
the three context traces as parallel sequences: at parallel step `t`, the
estimator uses `g_1[t]`, `g_2[t]`, and `g_4[t]`. The plotted x-value is `3t`,
the total one-token call count across `C_1`, `C_2`, and `C_4`. Contexts where
the fixed decoding parameters make token ID `15` or token ID `16` unsampleable
are marked undefined.

In the main reference-canary notebook, the reference model supplies canaries.
Under this protocol, the `Qwen2.5-7B-Instruct` suspect curve is undefined for
the `Qwen2.5-14B-Instruct` and `Qwen2.5-32B-Instruct` references because the
fixed decoding/top-p setup removes one target token on those reference-owned
canaries. The focused 7B-suspect notebook is a prompt-engineering workaround:
it keeps the same fixed decoding setup but uses the 7B canaries for both the
reference curve and the 7B suspect curve. This produces visual comparison
graphs for the 14B reference vs 7B suspect and the 32B reference vs 7B suspect
without changing the model payloads or decoding seed. In that focused notebook,
the 14B reference uses `7,500` samples per context and the 32B reference uses
`15,000` samples per context.

Loaded payload logits are cast to `np.float64` immediately after loading. This
does not recover precision beyond what was saved in the `.pt` payloads, but it
prevents additional notebook-side precision loss during diagnostics.
