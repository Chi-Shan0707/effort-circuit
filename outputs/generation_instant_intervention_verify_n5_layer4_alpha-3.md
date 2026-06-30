# Generation Intervention Experiment

## Setup
- n: 5
- layer: 4
- alpha: -3.0
- high mode for vector: verify
- prompt mode for generation: instant
- max new tokens: 16

## Summary
| condition | accuracy | mean reasoning tokens | final rate | repetition |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 16.00 | 0.000 | 0.0000 |
| effort_candidate | 0.000 | 16.00 | 0.000 | 0.0000 |
| effort_opposite | 0.000 | 16.00 | 0.000 | 0.0000 |
| random_control | 0.000 | 16.00 | 0.000 | 0.0000 |

## Examples
- `baseline` `train-000000` correct=False tokens=16 answer=None gold='270':  200\nOkay, let's see. The question says if x
- `effort_candidate` `train-000000` correct=False tokens=16 answer=None gold='270':  200\nOkay, let's see. The question says if x
- `effort_opposite` `train-000000` correct=False tokens=16 answer=None gold='270':  200\nOkay, let's see. The question says if x
- `random_control` `train-000000` correct=False tokens=16 answer=None gold='270':  200\nOkay, let's see. The question says if x
- `baseline` `train-000001` correct=False tokens=16 answer=None gold='49':  18\nOkay, let's see. The problem says there's a
- `effort_candidate` `train-000001` correct=False tokens=16 answer=None gold='49':  18\nOkay, let's see. The problem says there's a
- `effort_opposite` `train-000001` correct=False tokens=16 answer=None gold='49':  12\nOkay, let's see. The problem says there's a
- `random_control` `train-000001` correct=False tokens=16 answer=None gold='49':  12\nOkay, let's see. The problem says there's a
- `baseline` `train-000002` correct=False tokens=16 answer=None gold='105':  30\nOkay, let's see. The question says if x equals
- `effort_candidate` `train-000002` correct=False tokens=16 answer=None gold='105':  30\nOkay, let's see. The question says if x equals
- `effort_opposite` `train-000002` correct=False tokens=16 answer=None gold='105':  30\nOkay, let's see. The question says if x equals
- `random_control` `train-000002` correct=False tokens=16 answer=None gold='105':  30\nOkay, let's see. The question says if x equals
- `baseline` `train-000003` correct=False tokens=16 answer=None gold='108':  36\nOkay, let's see. The question says if x equals
- `effort_candidate` `train-000003` correct=False tokens=16 answer=None gold='108':  36\nOkay, let's see. The question says if x equals
- `effort_opposite` `train-000003` correct=False tokens=16 answer=None gold='108':  36\nOkay, let's see. The question is asking, if
- `random_control` `train-000003` correct=False tokens=16 answer=None gold='108':  36\nOkay, let's see. The question says if x equals
- `baseline` `train-000004` correct=False tokens=16 answer=None gold='513':  38\nOkay, let's see. The question says if x equals
- `effort_candidate` `train-000004` correct=False tokens=16 answer=None gold='513':  38\nOkay, let's see. The question says if x equals
- `effort_opposite` `train-000004` correct=False tokens=16 answer=None gold='513':  190\nOkay, let's see. The question says if x
- `random_control` `train-000004` correct=False tokens=16 answer=None gold='513':  38\nOkay, let's see. The question says if x equals
