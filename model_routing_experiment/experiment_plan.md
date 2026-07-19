# Model Re-Routing Experiment Plan

## Core Protocol

The model re-routing proof-of-concept uses requested-configuration-specific canaries:

```text
requested-configuration logits on requested-configuration canaries
actual-configuration sampling distributions on requested-configuration canaries
```

For each fixed requested model configuration, all three actual model configurations are evaluated on the
same canary prompt slots supplied by the requested config.

This is operationally simpler than actual-config-specific canaries. A requested-configuration
owner can find a canary suite once for their own model and reuse it against any
actual model configuration.

## Token Pair

The target token pair is:

| Role | Qwen token ID | Decoded string |
| --- | ---: | --- |
| `a` | `15` | `0` |
| `b` | `16` | `1` |

For each context slot `C_j`, the sampled quantity is:

```text
g_j = log(n_{j,a}) - log(n_{j,b})
```

where `n_{j,a}` counts token ID `15` and `n_{j,b}` counts token ID `16`.

## Canary Assignments

The canary prompt maps are requested-configuration-owned:

| Requested model configuration | C_1 | C_2 = C_3 | C_4 |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | Prompt 1 | Prompt 2 | Prompt 3 |
| Qwen2.5-14B-Instruct | Prompt 1 | Prompt 4 | Prompt 5 |
| Qwen2.5-32B-Instruct | Prompt 2 | Prompt 6 | Prompt 4 |

When `Qwen2.5-7B-Instruct` is the requested config, all actual configs are tested on prompts
`1, 2, 3`. When `Qwen2.5-14B-Instruct` is the requested config, all actual configs are
tested on prompts `1, 4, 5`. When `Qwen2.5-32B-Instruct` is the requested config, all
actual configs are tested on prompts `2, 6, 4`.

## Ordered Runs

The experiment has nine ordered runs:

| Requested model configuration | Actual model configuration |
| --- | --- |
| Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct |
| Qwen2.5-7B-Instruct | Qwen2.5-32B-Instruct |
| Qwen2.5-14B-Instruct | Qwen2.5-7B-Instruct |
| Qwen2.5-14B-Instruct | Qwen2.5-14B-Instruct |
| Qwen2.5-14B-Instruct | Qwen2.5-32B-Instruct |
| Qwen2.5-32B-Instruct | Qwen2.5-7B-Instruct |
| Qwen2.5-32B-Instruct | Qwen2.5-14B-Instruct |
| Qwen2.5-32B-Instruct | Qwen2.5-32B-Instruct |

The diagonal runs are the identity cases and should be closest to zero. The
off-diagonal runs should be visibly nonzero or undefined because the fixed
decoding/top-p setup removes one of the target tokens.

## Random Decoding Seed

We use one shared random decoding seed:

```text
shared_decoding_seed = 20260722
```

The selected decoding parameters are:

```text
temperature = 1.0370942971540018
top_p = 0.9425723689529342
logit_bias = {15: 2, 16: 1}
```

## Fixed Run Length And Undefined Runs

Each defined context runs for exactly:

```text
7,500 one-token samples
```

The graph treats the three context traces as parallel sequences. At parallel
step `t`, the estimator uses `g_1[t]`, `g_2[t]`, and `g_4[t]`. Since one sample
has been drawn from each of the three context slots, the plotted x-value is
`3t`, i.e. the total number of one-token calls across `C_1`, `C_2`, and `C_4`.

If top-p truncation gives token ID `15` or token ID `16` zero probability for a
context, the notebook records the run as undefined.

## 7B Actual-Config Prompt Engineering Workaround

The main graph notebook follows the requested-configuration-canary protocol above: the
requested model configuration supplies the canary prompts, and every actual model configuration is
evaluated on the requested model configuration's canaries.

Under the fixed decoding setup, the `Qwen2.5-7B-Instruct` actual-config curve is
undefined when the requested config is `Qwen2.5-14B-Instruct` or
`Qwen2.5-32B-Instruct`, because the top-p cutoff removes token ID `16` on some
of the requested-configuration-owned canaries. To produce a visual comparison for these
cases, we use a separate fixed-budget notebook that keeps
`Qwen2.5-7B-Instruct` as the actual config but uses the 7B canaries for both curves in
each graph:

| Requested config | Curves shown | Canary provider |
| --- | --- | --- |
| Qwen2.5-14B-Instruct | Qwen2.5-14B-Instruct requested config and Qwen2.5-7B-Instruct actual config | Qwen2.5-7B-Instruct |
| Qwen2.5-32B-Instruct | Qwen2.5-32B-Instruct requested config and Qwen2.5-7B-Instruct actual config | Qwen2.5-7B-Instruct |

This is a prompt-engineering workaround, not a change to the model payloads or
decoding seed. The purpose is to show the 7B actual-config curve under canaries where
both target tokens remain sampleable. In this focused notebook, the
Qwen2.5-14B-Instruct requested config uses `7,500` samples per context and the
Qwen2.5-32B-Instruct requested config uses `15,000` samples per context.
