# Data-Driven Token Metric

## Learned Token Clusters

- Continue/high cluster: `[' Diagram', ' Trace', ' Circ', ' Analy', ' Logic', ' Rational', ' Xiao', ' diagram', ' ...,', ' Mark', ' [...]', ' Taking']`
- Final/low cluster: `[' \\', '3', '2', '1']`

## Heldout Summary

| condition | n | cluster gap | continue mass | final mass | KL vs low | answer margin | EOS/Final margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 5 | -0.013686 | 0.000000 | 0.013686 | 0.000000 | -7.716664 | 0.000000 |
| high | 5 | 0.005499 | 0.006327 | 0.000828 | 6.899302 | -7.095190 | -0.006323 |
| layer18_t1 | 5 | 0.000100 | 0.001953 | 0.001853 | 5.099311 | -7.082969 | -0.001949 |
| layer27_t0.75 | 5 | -0.001869 | 0.001072 | 0.002941 | 2.564157 | -7.428588 | -0.001069 |
| layer27_t1 | 5 | 0.005499 | 0.006327 | 0.000828 | 6.899302 | -7.095190 | -0.006323 |

## Guardrail

Clusters are selected only on the train split and evaluated on heldout rows. A useful metric should remain interpretable when combined with KL, answer-token margin, and EOS/Final margin; a high cluster gap alone is not a task-utility result.
