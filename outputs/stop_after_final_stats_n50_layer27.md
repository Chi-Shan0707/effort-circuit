# Paired Statistical Analysis

Input: `outputs/stop_after_final_heldout_n50_layer27.json`
Bootstrap iterations: `10000`

## Metric: `first_final_correct`

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 50 | 0.000 | [0.000, 0.071] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.6 | 50 | 0.000 | [0.000, 0.071] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.75 | 50 | 0.000 | [0.000, 0.071] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.9 | 50 | 0.000 | [0.000, 0.071] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t1 | 50 | 0.000 | [0.000, 0.071] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |

## Metric: `last_final_correct`

| condition | n | acc | Wilson 95% CI | diff vs baseline | bootstrap 95% CI | b-wrong/c-right | b-right/c-wrong | McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 50 | 0.280 | [0.175, 0.417] | +0.000 | [+0.000, +0.000] | 0 | 0 | 1.000 |
| layer27_t0.6 | 50 | 0.660 | [0.522, 0.776] | +0.380 | [+0.240, +0.520] | 19 | 0 | 0.000 |
| layer27_t0.75 | 50 | 0.660 | [0.522, 0.776] | +0.380 | [+0.240, +0.520] | 19 | 0 | 0.000 |
| layer27_t0.9 | 50 | 0.660 | [0.522, 0.776] | +0.380 | [+0.240, +0.520] | 19 | 0 | 0.000 |
| layer27_t1 | 50 | 0.320 | [0.208, 0.458] | +0.040 | [-0.140, +0.220] | 12 | 10 | 0.832 |

## Auxiliary Protocol Metrics

| condition | n | strict stop rate | strict malformed rate | mean tokens | repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 50 | 0.260 | 0.740 | 78.08 | 0.1062 |
| layer27_t0.6 | 50 | 0.260 | 0.740 | 75.60 | 0.0515 |
| layer27_t0.75 | 50 | 0.260 | 0.740 | 75.60 | 0.0515 |
| layer27_t0.9 | 50 | 0.260 | 0.740 | 75.60 | 0.0515 |
| layer27_t1 | 50 | 0.240 | 0.760 | 77.16 | 0.1414 |

## Interpretation Guardrail

For small n, Wilson intervals and exact paired tests are intentionally wide. Treat these tables as a falsification and measurement-quality audit, not as evidence of a settled effect size.
