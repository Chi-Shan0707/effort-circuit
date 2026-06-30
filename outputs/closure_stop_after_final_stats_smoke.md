# Paired Statistical Analysis

Input: `outputs/closure_stop_after_final_smoke.json`
Bootstrap iterations: `1000`

## Metric: `first_final_correct`

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2 | 0.000 | [0.000, 0.658] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.75 | 2 | 0.000 | [0.000, 0.658] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |

## Metric: `last_final_correct`

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2 | 0.000 | [0.000, 0.658] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.75 | 2 | 0.000 | [0.000, 0.658] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |

## Auxiliary Protocol Metrics

| condition | n | strict stop rate | strict malformed rate | mean tokens | repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 2 | 0.000 | 1.000 | 32.00 | 0.0000 |
| layer27_t0.75 | 2 | 0.500 | 0.500 | 24.50 | 0.0833 |

## Interpretation Guardrail

For small n, Wilson intervals and exact paired tests are intentionally wide. Treat these tables as a falsification and measurement-quality audit, not as evidence of a settled effect size.
