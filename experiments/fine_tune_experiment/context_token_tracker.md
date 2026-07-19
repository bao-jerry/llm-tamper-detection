# Fine-Tune Experiment Context / Token Tracker

This tracker records prompt-level next-token behavior for the Qwen2.5-14B fine-tune / specialized-variant experiment.

## Target Tokens

| Role | Token ID | Decoded token |
|---|---:|---|
| `token_a` | 15 | `0` |
| `token_b` | 16 | `1` |

## Model Set

| Model key | Model ID | Status |
|---|---|---|
| `14b_instruct` | `Qwen/Qwen2.5-14B-Instruct` | recorded below |
| `14b_instruct_1m` | `Qwen/Qwen2.5-14B-Instruct-1M` | recorded below |
| `14b_coder_instruct` | `Qwen/Qwen2.5-Coder-14B-Instruct` | recorded below |

## Shared Prompt Inventory

| Prompt | Context |
|---:|---|
| 1 | Output a random binary sequence and nothing else. I should not be seeing words in your response. |
| 2 | Generate a RANDOM number from {zero, one}. Your response should be a single numerical digit. Nothing else. Try to be as unpredictable as possible. |
| 3 | Generate a highly random sequence of zero's and one's. Your response should be a string of numerical digits and nothing else. Do NOT output non-numerical text. |
| 4 | Simulate a completely random coin flip. Output the number zero if it's heads. Output the number one if it's tails. Your output should be a single numerical digit. DON'T GIVE THE SAME RESPONSE EVERY TIME. TRY TO MAKE IT A FIFTY-FIFTY CHANCE. |
| 5 | Generate a completely random comma-separated list of zero's and ones. Each entry in the list must be a numerical digit. Do not output anything else except this list. |
| 6 | OUTPUT THE NUMBER ZERO OR THE NUMBER ONE. MAKE IT A FIFTY-FIFTY CHANCE. YOUR RESPONSE SHOULD BE A NUMERICAL DIGIT AND NOTHING ELSE. |

## Recorded Results: `14b_instruct_1m`

Model ID: `Qwen/Qwen2.5-14B-Instruct-1M`

Load mode: `8bit`

Payload schema: `next_token_logits_float64_v1`

| Prompt | File | Context tokens | Logit `0` | Logit `1` | P(`0`) | P(`1`) | P(`0`) + P(`1`) | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `qwen25_14b_instruct_1m_8bit_prompt_1_logits.pt` | 38 | 30.875 | 40.5 | 0.0000660521352728 | 0.9999338028 | 0.999999854935 | Very strongly favors `1`. |
| 2 | `qwen25_14b_instruct_1m_8bit_prompt_2_logits.pt` | 49 | 34 | 45.75 | 0.00000788926256778 | 0.999992108397 | 0.99999999766 | Very strongly favors `1`. |
| 3 | `qwen25_14b_instruct_1m_8bit_prompt_3_logits.pt` | 53 | 35.5 | 40 | 0.010986942612 | 0.989013055697 | 0.999999998309 | Strongly favors `1`. |
| 4 | `qwen25_14b_instruct_1m_8bit_prompt_4_logits.pt` | 78 | 43.75 | 37.75 | 0.997527357322 | 0.00247262310825 | 0.99999998043 | Strongly favors `0`. |
| 5 | `qwen25_14b_instruct_1m_8bit_prompt_5_logits.pt` | 52 | 42.5 | 44.5 | 0.119202915039 | 0.880797026378 | 0.999999941416 | Best balance among the recorded 1M prompts. |
| 6 | `qwen25_14b_instruct_1m_8bit_prompt_6_logits.pt` | 54 | 43 | 40 | 0.952574124842 | 0.047425873079 | 0.999999997921 | Favors `0`, but minority token is still visible in base distribution. |

## Recorded Results: `14b_instruct`

Model ID: `Qwen/Qwen2.5-14B-Instruct`

Load mode: `8bit`

Payload schema: `next_token_logits_float64_v1`

| Prompt | File | Context tokens | Logit `0` | Logit `1` | P(`0`) | P(`1`) | P(`0`) + P(`1`) | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `qwen25_14b_instruct_8bit_prompt_1_logits.pt` | 48 | 39.25 | 40.25 | 0.26894136405 | 0.731058422817 | 0.999999786866 | Usable; moderate skew toward `1`. |
| 2 | `qwen25_14b_instruct_8bit_prompt_2_logits.pt` | 59 | 35.75 | 39.25 | 0.0293122245807 | 0.970687564905 | 0.999999789485 | Strongly favors `1`. |
| 3 | `qwen25_14b_instruct_8bit_prompt_3_logits.pt` | 63 | 42.75 | 41.5 | 0.777299861001 | 0.222700138776 | 0.999999999777 | Usable; moderate skew toward `0`. |
| 4 | `qwen25_14b_instruct_8bit_prompt_4_logits.pt` | 88 | 37.75 | 37.5 | 0.562176088091 | 0.437823177629 | 0.999999265719 | Best balance among recorded `14b_instruct` prompts. |
| 5 | `qwen25_14b_instruct_8bit_prompt_5_logits.pt` | 62 | 39.5 | 39 | 0.622459245235 | 0.377540616657 | 0.999999861892 | Good balance. |
| 6 | `qwen25_14b_instruct_8bit_prompt_6_logits.pt` | 64 | 46 | 47.25 | 0.222700138778 | 0.77729986101 | 0.999999999788 | Usable; moderate skew toward `1`. |

## Recorded Results: `14b_coder_instruct`

Model ID: `Qwen/Qwen2.5-Coder-14B-Instruct`

Load mode: `8bit`

Payload schema: `next_token_logits_float64_v1`

| Prompt | File | Context tokens | Logit `0` | Logit `1` | P(`0`) | P(`1`) | P(`0`) + P(`1`) | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `qwen25_14b_coder_instruct_8bit_prompt_1_logits.pt` | 48 | 18.125 | 19.125 | 0.266429178695 | 0.724229595017 | 0.990658773712 | Usable; moderate skew toward `1`; some mass on code-fence tokens. |
| 2 | `qwen25_14b_coder_instruct_8bit_prompt_2_logits.pt` | 59 | 16.625 | 17.5 | 0.291962970881 | 0.7003827576 | 0.992345728482 | Usable; moderate skew toward `1`. |
| 3 | `qwen25_14b_coder_instruct_8bit_prompt_3_logits.pt` | 63 | 17.875 | 18.75 | 0.291711352739 | 0.699779157055 | 0.991490509794 | Usable; moderate skew toward `1`; some mass on code-fence tokens. |
| 4 | `qwen25_14b_coder_instruct_8bit_prompt_4_logits.pt` | 88 | 17 | 17.125 | 0.462538078583 | 0.524124308231 | 0.986662386814 | Best balance among recorded `14b_coder_instruct` prompts. |
| 5 | `qwen25_14b_coder_instruct_8bit_prompt_5_logits.pt` | 62 | 18.875 | 17.625 | 0.772603681339 | 0.221354660776 | 0.993958342115 | Usable; moderate skew toward `0`. |
| 6 | `qwen25_14b_coder_instruct_8bit_prompt_6_logits.pt` | 64 | 18.5 | 16.375 | 0.885406203043 | 0.105746690951 | 0.991152893994 | Favors `0`, but minority token remains visible. |

## Current Canary Selection Notes

No final fine-tune canary set has been chosen yet.

For `14b_instruct`, prompts 4 and 5 are the best balanced raw prompts, with prompts 1, 3, and 6 also plausible. Prompt 2 is relatively lopsided.

For `14b_instruct_1m`, prompt 5 is currently the most balanced raw prompt for token IDs 15 and 16. Prompts 1 and 2 are likely poor canary candidates under top-p style decoding because the minority token is extremely small before any decoding modifications. Prompt 6 may be useful only if the chosen decoding parameters keep the minority token in support.

For `14b_coder_instruct`, prompt 4 is the best balanced raw prompt. Prompts 1, 2, 3, and 5 are also usable. Prompt 6 is more lopsided but still has visible support for both target tokens. Unlike the non-coder variants, the coder model leaves nontrivial probability mass on code-fence and formatting tokens for several prompts, so P(`0`) + P(`1`) is closer to 0.99 than 1.00.

All three planned fine-tune / specialized-variant prompt inventories have now been recorded. Choose canaries based on practical support for both target tokens under the intended decoding setup, or skip sampling graphs and report neutral-decoding population estimator limits directly.

If the fine-tune experiment is presented as a neutral-decoding theoretical-limit check, graphs may not be necessary. In that case, use `temperature = 1`, no logit bias, and `top_p = 1`, then report the population estimator limits directly from these saved logits.
