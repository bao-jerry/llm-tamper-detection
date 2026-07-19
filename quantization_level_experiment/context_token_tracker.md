# Quantization-Level Experiment Context / Token Tracker

This tracker records prompt-level next-token behavior for the Qwen2.5-14B quantization/loading experiment.

## Target Tokens

| Role | Token ID | Decoded token |
|---|---:|---|
| `token_a` | 15 | `0` |
| `token_b` | 16 | `1` |

## Fixed Model

| Model key | Model ID |
|---|---|
| `14b` | `Qwen/Qwen2.5-14B-Instruct` |

## Load Modes

| Load mode | Status |
|---|---|
| `bf16` | recorded below |
| `fp16` | pending |
| `8bit` | recorded below; recycled from fine-tune `14b_instruct` run |
| `4bit` | recorded below |

## Shared Prompt Inventory

| Prompt | Context |
|---:|---|
| 1 | Output a random binary sequence and nothing else. I should not be seeing words in your response. |
| 2 | Generate a RANDOM number from {zero, one}. Your response should be a single numerical digit. Nothing else. Try to be as unpredictable as possible. |
| 3 | Generate a highly random sequence of zero's and one's. Your response should be a string of numerical digits and nothing else. Do NOT output non-numerical text. |
| 4 | Simulate a completely random coin flip. Output the number zero if it's heads. Output the number one if it's tails. Your output should be a single numerical digit. DON'T GIVE THE SAME RESPONSE EVERY TIME. TRY TO MAKE IT A FIFTY-FIFTY CHANCE. |
| 5 | Generate a completely random comma-separated list of zero's and ones. Each entry in the list must be a numerical digit. Do not output anything else except this list. |
| 6 | OUTPUT THE NUMBER ZERO OR THE NUMBER ONE. MAKE IT A FIFTY-FIFTY CHANCE. YOUR RESPONSE SHOULD BE A NUMERICAL DIGIT AND NOTHING ELSE. |

## Recorded Results: `bf16`

Model ID: `Qwen/Qwen2.5-14B-Instruct`

Load mode: `bf16`

Payload schema: `next_token_logits_float64_v1`

| Prompt | File | Context tokens | Logit `0` | Logit `1` | P(`0`) | P(`1`) | P(`0`) + P(`1`) | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `qwen25_14b_bf16_prompt_1_logits.pt` | 48 | 39.75 | 41 | 0.222700091518 | 0.777299696056 | 0.999999787574 | Usable; moderate skew toward `1`. |
| 2 | `qwen25_14b_bf16_prompt_2_logits.pt` | 59 | 37.75 | 42 | 0.0140636263802 | 0.985936326474 | 0.999999952854 | Strongly favors `1`. |
| 3 | `qwen25_14b_bf16_prompt_3_logits.pt` | 63 | 42.25 | 41.5 | 0.679178698882 | 0.320821300686 | 0.999999999569 | Usable; moderate skew toward `0`. |
| 4 | `qwen25_14b_bf16_prompt_4_logits.pt` | 88 | 39 | 37.5 | 0.817574134891 | 0.182425447652 | 0.999999582543 | Usable, but noticeably skewed toward `0`. |
| 5 | `qwen25_14b_bf16_prompt_5_logits.pt` | 62 | 39.5 | 39 | 0.622459236719 | 0.377540611491 | 0.99999984821 | Good balance. |
| 6 | `qwen25_14b_bf16_prompt_6_logits.pt` | 64 | 44.5 | 44.5 | 0.499999999556 | 0.499999999556 | 0.999999999112 | Perfect raw tie between target token logits. |

## Recorded Results: `8bit`

Model ID: `Qwen/Qwen2.5-14B-Instruct`

Load mode: `8bit`

Payload schema: `next_token_logits_float64_v1`

Source note: these payloads are losslessly recycled from the fine-tune experiment's `14b_instruct` run and copied into this experiment with quantization-style filenames.

| Prompt | File | Context tokens | Logit `0` | Logit `1` | P(`0`) | P(`1`) | P(`0`) + P(`1`) | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `qwen25_14b_8bit_prompt_1_logits.pt` | 48 | 39.25 | 40.25 | 0.26894136405 | 0.731058422817 | 0.999999786866 | Usable; moderate skew toward `1`. |
| 2 | `qwen25_14b_8bit_prompt_2_logits.pt` | 59 | 35.75 | 39.25 | 0.0293122245807 | 0.970687564905 | 0.999999789485 | Strongly favors `1`. |
| 3 | `qwen25_14b_8bit_prompt_3_logits.pt` | 63 | 42.75 | 41.5 | 0.777299861001 | 0.222700138776 | 0.999999999777 | Usable; moderate skew toward `0`. |
| 4 | `qwen25_14b_8bit_prompt_4_logits.pt` | 88 | 37.75 | 37.5 | 0.562176088091 | 0.437823177629 | 0.999999265719 | Best balance among recorded `8bit` prompts. |
| 5 | `qwen25_14b_8bit_prompt_5_logits.pt` | 62 | 39.5 | 39 | 0.622459245235 | 0.377540616657 | 0.999999861892 | Good balance. |
| 6 | `qwen25_14b_8bit_prompt_6_logits.pt` | 64 | 46 | 47.25 | 0.222700138778 | 0.77729986101 | 0.999999999788 | Usable; moderate skew toward `1`. |

## Recorded Results: `4bit`

Model ID: `Qwen/Qwen2.5-14B-Instruct`

Load mode: `4bit`

Payload schema: `next_token_logits_float64_v1`

| Prompt | File | Context tokens | Logit `0` | Logit `1` | P(`0`) | P(`1`) | P(`0`) + P(`1`) | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `qwen25_14b_4bit_prompt_1_logits.pt` | 48 | 44.5 | 44.5 | 0.499999991099 | 0.499999991099 | 0.999999982197 | Perfect raw tie between target token logits. |
| 2 | `qwen25_14b_4bit_prompt_2_logits.pt` | 59 | 45.25 | 49.25 | 0.017986209925 | 0.982013788013 | 0.999999997938 | Strongly favors `1`. |
| 3 | `qwen25_14b_4bit_prompt_3_logits.pt` | 63 | 47.75 | 48 | 0.437823498998 | 0.562176500737 | 0.999999999735 | Good balance; slight skew toward `1`. |
| 4 | `qwen25_14b_4bit_prompt_4_logits.pt` | 88 | 43.5 | 39.75 | 0.977022622688 | 0.0229773697359 | 0.999999992424 | Strongly favors `0`; important canary distortion relative to bf16/8bit. |
| 5 | `qwen25_14b_4bit_prompt_5_logits.pt` | 62 | 47.75 | 46 | 0.851952791047 | 0.148047196134 | 0.999999987181 | Favors `0`; important canary distortion relative to bf16/8bit. |
| 6 | `qwen25_14b_4bit_prompt_6_logits.pt` | 64 | 45.5 | 42.75 | 0.939913349647 | 0.0600866501625 | 0.999999999809 | Strongly favors `0`. |

## Current Notes

The quantization experiment fixes the model identity as `Qwen/Qwen2.5-14B-Instruct` and varies only the loading/quantization mode.

For comparability with the model re-routing experiment, the same 14B canary prompts can be used:

- `C1 = prompt 1`
- `C2 = C3 = prompt 4`
- `C4 = prompt 5`

The theoretical-limit heatmap notebook computes 8 total `3 x 3` grids over the recorded load modes:

- 4 decoding conditions
- 2 naive estimators

Each grid uses requested load mode as the row axis and actual load mode as the column axis. Undefined cells caused by top-p removing one of the target tokens are shown as `N/A`.

Early qualitative read: `4bit` appears substantially shifted from both `bf16` and `8bit` on the model re-routing 14B canaries, especially prompt 4 and prompt 5. This should make 4-bit quantization easy to detect in the neutral theoretical-limit check.
