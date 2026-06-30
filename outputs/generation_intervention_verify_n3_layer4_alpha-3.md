# Generation Intervention Experiment

## Setup
- n: 3
- layer: 4
- alpha: -3.0
- high mode for vector: verify
- prompt mode for generation: cot
- max new tokens: 32

## Summary
| condition | accuracy | mean reasoning tokens | final rate | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 32.00 | 0.000 | 0.0000 |
| effort_candidate | 0.000 | 32.00 | 0.000 | 0.0000 |
| effort_opposite | 0.000 | 32.00 | 0.000 | 0.0000 |
| random_control | 0.000 | 32.00 | 0.000 | 0.0000 |

## Examples
- `baseline` `train-000000` correct=False tokens=32 answer='27' gold='270':  Step 1: 2 + 25 = 27. Step 2: 27 * 10 = 270.
- `effort_candidate` `train-000000` correct=False tokens=32 answer='27' gold='270':  Step 1: 2 + 25 = 27. Step 2: 27 * 10 = 270.
- `effort_opposite` `train-000000` correct=False tokens=32 answer='27' gold='270':  Step 1: 2 + 25 = 27. Step 2: 27 * 10 = 270.
- `random_control` `train-000000` correct=False tokens=32 answer='27' gold='270':  Step 1: 2 + 25 = 27. Step 2: 27 * 10 = 270.
- `baseline` `train-000001` correct=False tokens=32 answer='5' gold='49':  Step 1: 9 rows with 6 items in each row = 9 * 6 = 54 items. Step 2: 5
- `effort_candidate` `train-000001` correct=False tokens=32 answer='3' gold='49':  Step 1: 9 rows * 6 items per row = 54 items. Step 2: 5 items are removed. Step 3
- `effort_opposite` `train-000001` correct=False tokens=32 answer=None gold='49':  Step 1: 9 rows with 6 items in each row = 9*6 = 54 items. Step 2: 5 items
- `random_control` `train-000001` correct=False tokens=32 answer='5' gold='49':  Step 1: 9 rows with 6 items in each row = 9 * 6 = 54 items. Step 2: 5
- `baseline` `train-000002` correct=False tokens=32 answer='35' gold='105':  Step 1: 20 + 15 = 35. Step 2: 35 * 3 = 105.
- `effort_candidate` `train-000002` correct=False tokens=32 answer='35' gold='105':  Step 1: 20 + 15 = 35. Step 2: 35 times 3 is 105.
- `effort_opposite` `train-000002` correct=False tokens=32 answer='35' gold='105':  Step 1: 20 + 15 = 35. Step 2: 35 * 3 = 105.
- `random_control` `train-000002` correct=False tokens=32 answer='35' gold='105':  Step 1: 20 + 15 = 35. Step 2: 35 * 3 = 105.
