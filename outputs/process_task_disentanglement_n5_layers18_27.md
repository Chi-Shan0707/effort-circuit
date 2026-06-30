# Process vs Task Disentanglement

## Summary

| condition | layer | n | gap | target lp | source lp | source-target lp | source beats target | target rank | source rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cross_question_high_replace | 18 | 5 | 0.207214 | -8.571313 | -9.227875 | -0.656561 | 0.200 | 295.4 | 438.6 |
| cross_question_high_replace | 27 | 5 | 0.284528 | -9.575591 | -9.993956 | -0.418365 | 0.200 | 681.4 | 841.6 |
| low | 18 | 5 | 0.000099 | -7.740670 | -8.178222 | -0.437551 | 0.200 | 7.4 | 11.4 |
| low | 27 | 5 | 0.000099 | -7.740670 | -8.178222 | -0.437551 | 0.200 | 7.4 | 11.4 |
| mean_process_direction_add | 18 | 5 | 0.205474 | -8.351502 | -9.049644 | -0.698142 | 0.200 | 330.6 | 383.8 |
| mean_process_direction_add | 27 | 5 | 0.258192 | -9.025686 | -9.887132 | -0.861446 | 0.200 | 440.8 | 947.2 |
| same_question_high_replace | 18 | 5 | 0.194027 | -8.430222 | -8.855582 | -0.425360 | 0.200 | 325.0 | 386.4 |
| same_question_high_replace | 27 | 5 | 0.264117 | -9.265792 | -9.794787 | -0.528995 | 0.200 | 672.8 | 847.4 |

## Guardrail

Cross-question replacement should transfer process posture without making the source answer more likely than the target answer. Mean process-direction addition is safer if it preserves posture effects while reducing source-answer leakage.
