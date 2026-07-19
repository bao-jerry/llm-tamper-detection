# Context and Token Tracker

This file tracks the six prompt/context strings, token pair, and saved payloads
used by the primary reference-canary experiment.

In the primary experiment, canaries are owned by the reference model `A`. For a
fixed `A`, the same canary prompt slots are reused against every suspect model
`B`.

## Chosen Token Pair

For the Qwen2.5 tokenizer:

- Token ID `15`: `0`
- Token ID `16`: `1`

## Hardware and Runtime

Recorded during the Colab run:

- Runtime: Python 3 Google Compute Engine backend (GPU)
- GPU: NVIDIA A100-SXM4-40GB
- GPU memory: `40960 MiB` total
- Driver version: `580.82.07`
- CUDA version: `13.0`
- `nvidia-smi` timestamp: `Mon Jun 8 17:58:13 2026`
- Observed process: `/usr/bin/python3`
- Observed GPU memory usage during 32B 4bit run: `31752 MiB / 40960 MiB`
- Colab resource panel:
  - System RAM: `8.4 / 83.5 GB`
  - GPU RAM: `31.0 / 40.0 GB`
  - Disk: `108.4 / 112.6 GB`
  - Resource window shown: `12:38 PM to 1:58 PM`

## Prompt Inventory

### Prompt 1

```text
Output a random binary sequence and nothing else. I should not be seeing words in your response.
```

### Prompt 2

```text
Generate a RANDOM number from {zero, one}. Your response should be a single numerical digit. Nothing else. Try to be as unpredictable as possible.
```

### Prompt 3

```text
Generate a highly random sequence of zero's and one's. Your response should be a string of numerical digits and nothing else. Do NOT output non-numerical text.
```

### Prompt 4

```text
Simulate a completely random coin flip. Output the number zero if it's heads. Output the number one if it's tails. Your output should be a single numerical digit. DON'T GIVE THE SAME RESPONSE EVERY TIME. TRY TO MAKE IT A FIFTY-FIFTY CHANCE.
```

### Prompt 5

```text
Generate a completely random comma-separated list of zero's and ones. Each entry in the list must be a numerical digit. Do not output anything else except this list.
```

### Prompt 6

```text
OUTPUT THE NUMBER ZERO OR THE NUMBER ONE. MAKE IT A FIFTY-FIFTY CHANCE. YOUR RESPONSE SHOULD BE A NUMERICAL DIGIT AND NOTHING ELSE.
```

## Run: Qwen2.5-7B-Instruct, 8bit

- Model ID: `Qwen/Qwen2.5-7B-Instruct`
- Model key: `7b`
- Load mode: `8bit`
- Vocabulary size: `152064`
- Saved payload pattern: `qwen25_7b_8bit_prompt_{i}_logits.pt`

| Prompt | File | P(`0`) / token 15 | P(`1`) / token 16 | P(`0`) + P(`1`) | Formatted token count |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `qwen25_7b_8bit_prompt_1_logits.pt` | 0.79818672 | 0.20181321 | 0.99999993 | 48 |
| 2 | `qwen25_7b_8bit_prompt_2_logits.pt` | 0.34861484 | 0.65129822 | 0.99991307 | 59 |
| 3 | `qwen25_7b_8bit_prompt_3_logits.pt` | 0.77729988 | 0.22270015 | 1.00000003 | 63 |
| 4 | `qwen25_7b_8bit_prompt_4_logits.pt` | 0.92413396 | 0.07585754 | 0.99999149 | 88 |
| 5 | `qwen25_7b_8bit_prompt_5_logits.pt` | 0.90465051 | 0.09534946 | 0.99999997 | 62 |
| 6 | `qwen25_7b_8bit_prompt_6_logits.pt` | 0.99999583 | 0.00000422 | 1.00000005 | 64 |

Status: usable for prompts 1-5; prompt 6 is extremely imbalanced but still has essentially all mass on token IDs `15` and `16`.

Reference-owned canary prompt selection for 7B:

- `C_1 =` Prompt 1
- `C_2 =` Prompt 2
- `C_3 =` Prompt 2
- `C_4 =` Prompt 3

## Run: Qwen2.5-14B-Instruct, 8bit

- Model ID: `Qwen/Qwen2.5-14B-Instruct`
- Model key: `14b`
- Load mode: `8bit`
- Vocabulary size: `152064`
- Saved payload pattern: `qwen25_14b_8bit_prompt_{i}_logits.pt`

| Prompt | File | P(`0`) / token 15 | P(`1`) / token 16 | P(`0`) + P(`1`) | Formatted token count |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `qwen25_14b_8bit_prompt_1_logits.pt` | 0.26894137 | 0.73105842 | 0.99999979 | 48 |
| 2 | `qwen25_14b_8bit_prompt_2_logits.pt` | 0.02931223 | 0.97068757 | 0.99999979 | 59 |
| 3 | `qwen25_14b_8bit_prompt_3_logits.pt` | 0.77729988 | 0.22270015 | 1.00000003 | 63 |
| 4 | `qwen25_14b_8bit_prompt_4_logits.pt` | 0.56217605 | 0.43782315 | 0.99999920 | 88 |
| 5 | `qwen25_14b_8bit_prompt_5_logits.pt` | 0.62245923 | 0.37754062 | 0.99999985 | 62 |
| 6 | `qwen25_14b_8bit_prompt_6_logits.pt` | 0.22270015 | 0.77729988 | 1.00000003 | 64 |

Status: usable. Across all six prompts, almost all probability mass is on token IDs `15` and `16`.

Reference-owned canary prompt selection for 14B:

- `C_1 =` Prompt 1
- `C_2 =` Prompt 4
- `C_3 =` Prompt 4
- `C_4 =` Prompt 5

## Run: Qwen2.5-32B-Instruct, 4bit

- Model ID: `Qwen/Qwen2.5-32B-Instruct`
- Model key: `32b`
- Load mode: `4bit`
- Vocabulary size: `152064`
- Saved payload pattern: `qwen25_32b_4bit_prompt_{i}_logits.pt`

| Prompt | File | P(`0`) / token 15 | P(`1`) / token 16 | P(`0`) + P(`1`) | Formatted token count |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `qwen25_32b_4bit_prompt_1_logits.pt` | 0.09534946 | 0.90465051 | 0.99999997 | 48 |
| 2 | `qwen25_32b_4bit_prompt_2_logits.pt` | 0.03732688 | 0.96267307 | 0.99999995 | 59 |
| 3 | `qwen25_32b_4bit_prompt_3_logits.pt` | 0.11920291 | 0.88079703 | 0.99999994 | 63 |
| 4 | `qwen25_32b_4bit_prompt_4_logits.pt` | 0.62245935 | 0.37754068 | 1.00000003 | 88 |
| 5 | `qwen25_32b_4bit_prompt_5_logits.pt` | 0.26894143 | 0.73105860 | 1.00000003 | 62 |
| 6 | `qwen25_32b_4bit_prompt_6_logits.pt` | 0.43782353 | 0.56217653 | 1.00000006 | 64 |

Status: usable. Across all six prompts, almost all probability mass is on token IDs `15` and `16`.

Reference-owned canary prompt selection for 32B:

- `C_1 =` Prompt 2
- `C_2 =` Prompt 6
- `C_3 =` Prompt 6
- `C_4 =` Prompt 4

## Notes

- The target token pair captures essentially all next-token probability mass for every prompt in both recorded runs.
- The probability values are measured from raw saved logits with no decoding parameters applied.
- These runs use finite-precision loading modes (`8bit` and `4bit`), so the logits should be treated as backend-specific empirical quantities rather than idealized real-valued logits.
