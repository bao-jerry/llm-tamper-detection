# Prompt-Injection Context / Token Tracker

## Fixed Model

- Model: `Qwen/Qwen2.5-14B-Instruct`
- Load mode: `8bit`
- Payload schema: `next_token_logits_float64_v1`
- Logit dtype: `torch.float64`
- Vocabulary size: `152064`
- Target token `a`: token id `15`, string `0`
- Target token `b`: token id `16`, string `1`

## Prompt-Formatting Variants

Important: when the user-specified system prompt is `None`, Qwen's chat
template automatically supplies the effective system prompt:

```text
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

### `variant_1`

- Label: `PI Variant 1`
- User-specified system prompt: `None`
- Effective system prompt:

```text
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```

- Interpretation: baseline Qwen chat formatting with Qwen's default automatic
  system prompt.

### `variant_2`

- Label: `PI Variant 2`
- User-specified system prompt:

```text
You are a helpful assistant
```

- Effective system prompt:

```text
You are a helpful assistant
```

### `variant_3`

- Label: `PI Variant 3`
- User-specified system prompt:

```text
Amadeus
```

- Effective system prompt:

```text
Amadeus
```

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

## Canary Mapping

Use the same 14B canaries as the model re-routing experiment:

```text
C1 = prompt 1
C2 = C3 = prompt 4
C4 = prompt 5
```

## Saved Payloads

Expected filename pattern:

```text
qwen25_14b_prompt_injection_{prompt_formatting_key}_8bit_prompt_{prompt_index}_logits.pt
```

Complete run produced 18 payloads:

```text
3 prompt-formatting variants x 6 prompts = 18 logit payloads
```

`variant_1` is the baseline run. `variant_2` and `variant_3` were swapped in
from the final prompt-search run.

## Target-Token Probability Summary

These are base probabilities with no additional decoding parameters applied.
Values below were recomputed from the copied `.pt` payloads in
`prompt_injection_experiment/logit_payloads`.

| Variant | Prompt | File | P(token `0`, id 15) | P(token `1`, id 16) | P(0)+P(1) |
|---|---:|---|---:|---:|---:|
| `variant_1` | 1 | `qwen25_14b_prompt_injection_variant_1_8bit_prompt_1_logits.pt` | 0.26894136405 | 0.731058422817 | 0.999999786867 |
| `variant_1` | 2 | `qwen25_14b_prompt_injection_variant_1_8bit_prompt_2_logits.pt` | 0.0293122245807 | 0.970687564905 | 0.999999789486 |
| `variant_1` | 3 | `qwen25_14b_prompt_injection_variant_1_8bit_prompt_3_logits.pt` | 0.777299861001 | 0.222700138776 | 0.999999999777 |
| `variant_1` | 4 | `qwen25_14b_prompt_injection_variant_1_8bit_prompt_4_logits.pt` | 0.562176088091 | 0.437823177629 | 0.999999265720 |
| `variant_1` | 5 | `qwen25_14b_prompt_injection_variant_1_8bit_prompt_5_logits.pt` | 0.622459245235 | 0.377540616657 | 0.999999861892 |
| `variant_1` | 6 | `qwen25_14b_prompt_injection_variant_1_8bit_prompt_6_logits.pt` | 0.222700138778 | 0.777299861010 | 0.999999999788 |
| `variant_2` | 1 | `qwen25_14b_prompt_injection_variant_2_8bit_prompt_1_logits.pt` | 0.182425505259 | 0.817574393069 | 0.999999898327 |
| `variant_2` | 2 | `qwen25_14b_prompt_injection_variant_2_8bit_prompt_2_logits.pt` | 0.0293122301720 | 0.970687750062 | 0.999999980234 |
| `variant_2` | 3 | `qwen25_14b_prompt_injection_variant_2_8bit_prompt_3_logits.pt` | 0.437823498700 | 0.562176500353 | 0.999999999053 |
| `variant_2` | 4 | `qwen25_14b_prompt_injection_variant_2_8bit_prompt_4_logits.pt` | 0.777299577229 | 0.222700057474 | 0.999999634703 |
| `variant_2` | 5 | `qwen25_14b_prompt_injection_variant_2_8bit_prompt_5_logits.pt` | 0.562176349068 | 0.437823380879 | 0.999999729947 |
| `variant_2` | 6 | `qwen25_14b_prompt_injection_variant_2_8bit_prompt_6_logits.pt` | 0.268941420136 | 0.731058575277 | 0.999999995413 |
| `variant_3` | 1 | `qwen25_14b_prompt_injection_variant_3_8bit_prompt_1_logits.pt` | 0.222700120675 | 0.777299797824 | 0.999999918499 |
| `variant_3` | 2 | `qwen25_14b_prompt_injection_variant_3_8bit_prompt_2_logits.pt` | 0.0229773665948 | 0.977022489124 | 0.999999855719 |
| `variant_3` | 3 | `qwen25_14b_prompt_injection_variant_3_8bit_prompt_3_logits.pt` | 0.377540665759 | 0.622459326192 | 0.999999991951 |
| `variant_3` | 4 | `qwen25_14b_prompt_injection_variant_3_8bit_prompt_4_logits.pt` | 0.320821088958 | 0.679178250653 | 0.999999339611 |
| `variant_3` | 5 | `qwen25_14b_prompt_injection_variant_3_8bit_prompt_5_logits.pt` | 0.562175825631 | 0.437822973225 | 0.999998798856 |
| `variant_3` | 6 | `qwen25_14b_prompt_injection_variant_3_8bit_prompt_6_logits.pt` | 0.268941416606 | 0.731058565681 | 0.999999982288 |

## Canary Prompt Target Probabilities

The theoretical-limit heatmap notebook uses only prompt 1, prompt 4, and prompt
5 via the 14B canary mapping.

| Variant | System prompt | C1 / prompt 1 P(0), P(1) | C2=C3 / prompt 4 P(0), P(1) | C4 / prompt 5 P(0), P(1) |
|---|---|---:|---:|---:|
| `variant_1` | automatic Qwen default | 0.26894136405, 0.731058422817 | 0.562176088091, 0.437823177629 | 0.622459245235, 0.377540616657 |
| `variant_2` | `You are a helpful assistant` | 0.182425505259, 0.817574393069 | 0.777299577229, 0.222700057474 | 0.562176349068, 0.437823380879 |
| `variant_3` | `Amadeus` | 0.222700120675, 0.777299797824 | 0.320821088958, 0.679178250653 | 0.562175825631, 0.437822973225 |

## Canary Suitability Notes

- `variant_2` has usable support and well-separated canary log-odds.
- `variant_3` was selected because `Amadeus` gives all three canaries
  comfortable target-token support and avoids the earlier C2/C4 degeneracy.
- The final three variants are therefore suitable for the theoretical-limit
  prompt-injection heatmap experiment.
