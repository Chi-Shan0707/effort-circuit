# Strict Generation Prefix-Patch Experiment Re-extracted

## Summary
| condition | accuracy | answer rate | final rate | mean reasoning tokens | repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.333 | 1.000 | 1.000 | 96.00 | 0.7260 |
| layer18_t1 | 0.000 | 1.000 | 0.667 | 96.00 | 0.6774 |
| layer20_t1 | 0.333 | 1.000 | 0.667 | 96.00 | 0.5763 |
| layer27_t0.75 | 1.000 | 1.000 | 1.000 | 96.00 | 0.3582 |
| layer27_t1 | 1.000 | 1.000 | 1.000 | 96.00 | 0.2200 |

## Correctness by example
- `baseline` `train-000000` extracted='200' gold='270' correct=False
- `layer18_t1` `train-000000` extracted='200' gold='270' correct=False
- `layer20_t1` `train-000000` extracted='270' gold='270' correct=True
- `layer27_t0.75` `train-000000` extracted='270' gold='270' correct=True
- `layer27_t1` `train-000000` extracted='270' gold='270' correct=True
- `baseline` `train-000001` extracted='49' gold='49' correct=True
- `layer18_t1` `train-000001` extracted='45' gold='49' correct=False
- `layer20_t1` `train-000001` extracted='45' gold='49' correct=False
- `layer27_t0.75` `train-000001` extracted='49' gold='49' correct=True
- `layer27_t1` `train-000001` extracted='49' gold='49' correct=True
- `baseline` `train-000002` extracted='30' gold='105' correct=False
- `layer18_t1` `train-000002` extracted='30' gold='105' correct=False
- `layer20_t1` `train-000002` extracted='30' gold='105' correct=False
- `layer27_t0.75` `train-000002` extracted='105' gold='105' correct=True
- `layer27_t1` `train-000002` extracted='105' gold='105' correct=True
