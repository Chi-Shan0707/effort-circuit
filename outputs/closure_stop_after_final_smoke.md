# Stop-After-First-Final Experiment

Final marker protocol: `strict`

## Summary

| condition | first-final acc | last-detectable acc | strict/relaxed stop rate | malformed rate | mean tokens | answer changes | repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 0.000 | 0.000 | 1.000 | 32.00 | 0.00 | 0.0000 |
| layer27_t0.75 | 0.000 | 0.000 | 0.500 | 0.500 | 24.50 | 0.00 | 0.0833 |

## Examples
- `baseline` `heldout-000000` first=None gold='148' correct=False tokens=32:  <step 1>\n<step 2>\nAnswer: 22 + 14 = 36\nReasoning: <step 1>\n
- `layer27_t0.75` `heldout-000000` first='3' gold='148' correct=False tokens=17:  22 + 14 = 36\nFinal answer: 3
- `baseline` `heldout-000001` first=None gold='257' correct=False tokens=32:  <step 1>\n<step 2>\n<step 3>\nAnswer: 15 + 22 + 22 + 22
- `layer27_t0.75` `heldout-000001` first=None gold='257' correct=False tokens=32:  15 + 22 * 11 = ?\nAnswer: 15 + 22 * 11 = 22 + 
