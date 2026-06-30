# Paired Statistical Analysis

Input: `outputs/stop_after_final_heldout_n5_hook_states.json`
Bootstrap iterations: `10000`

## Metric: `first_final_correct`

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer18_t1 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.75 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t1 | 5 | 0.000 | [0.000, 0.434] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |

## Metric: `last_final_correct`

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.400 | [0.118, 0.769] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer18_t1 | 5 | 0.000 | [0.000, 0.434] | -0.400 | [-0.800, +0.000] | 0 | 2 | 0.500 |
| layer27_t0.75 | 5 | 0.600 | [0.231, 0.882] | +0.200 | [+0.000, +0.600] | 1 | 0 | 1.000 |
| layer27_t1 | 5 | 0.200 | [0.036, 0.624] | -0.200 | [-0.600, +0.000] | 0 | 1 | 1.000 |

## Auxiliary Protocol Metrics

| condition | n | strict stop rate | strict malformed rate | mean tokens | repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 0.200 | 0.800 | 105.40 | 0.0800 |
| layer18_t1 | 5 | 0.600 | 0.400 | 74.00 | 0.0466 |
| layer27_t0.75 | 5 | 0.200 | 0.800 | 105.80 | 0.0251 |
| layer27_t1 | 5 | 0.200 | 0.800 | 105.80 | 0.0861 |

## Interpretation Guardrail

For small n, Wilson intervals and exact paired tests are intentionally wide. Treat these tables as a falsification and measurement-quality audit, not as evidence of a settled effect size.
