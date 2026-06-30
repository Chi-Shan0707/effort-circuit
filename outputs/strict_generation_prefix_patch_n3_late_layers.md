# Strict Generation Prefix-Patch Experiment

## Setup
- n: 3
- low mode: instant
- high mode: verify
- max new tokens: 96

## Summary
| condition | accuracy | answer rate | final rate | mean reasoning tokens | repetition |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 1.000 | 1.000 | 96.00 | 0.7260 |
| layer18_t1 | 0.000 | 1.000 | 0.667 | 96.00 | 0.6774 |
| layer20_t1 | 0.333 | 1.000 | 0.667 | 96.00 | 0.5763 |
| layer27_t0.75 | 1.000 | 1.000 | 1.000 | 96.00 | 0.3582 |
| layer27_t1 | 0.333 | 1.000 | 1.000 | 96.00 | 0.2200 |

## Examples
- `baseline` `train-000000` correct=False answer='200' gold='270' tokens=96:  <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nF...
- `layer18_t1` `train-000000` correct=False answer='200' gold='270' tokens=96:  <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nFinal answer: 200\n\nReasoning: <brief arithmetic steps>\nF...
- `layer20_t1` `train-000000` correct=True answer='270' gold='270' tokens=96:  The value of x is 2 + 25. Then, x times 10 is 2 + 25 multiplied by 10. Then, 2 + 25 multiplied by 10 is 20 + 250 = 270.\nFinal answer: 270\nReasoning: <brief arithmetic steps>\nFinal answer: 270\nReasoning: <brief arithmetic steps>\nFinal answer: 
- `layer27_t0.75` `train-000000` correct=True answer='270' gold='270' tokens=96:  2 + 25 = 27. Then, 27 * 10 = 270.\nFinal answer: 270\nAnswer: 270\n\nReasoning: 2 + 25 = 27. Then, 27 * 10 = 270.\nFinal answer: 270\n\nReasoning: 2 + 25 = 27. Then, 2
- `layer27_t1` `train-000000` correct=True answer='270' gold='270' tokens=96:  The value of x is 2 + 25. Then, x times 10 is 2 + 25 multiplied by 10. Then, 2 + 25 multiplied by 10 is 20 + 250 = 270.\nFinal answer: 270\nReasoning: <brief arithmetic steps>\nFinal answer: 270\nReasoning: <brief arithmetic steps>\nFinal answer: 
- `baseline` `train-000001` correct=False answer='9' gold='49' tokens=96:  <step-by-step>\nFinal answer: 9 * 6 = 54, then subtract 5.\nReasoning: <step-by-step>\nFinal answer: 54 - 5 = 49\nAnswer: 49\n\nReasoning: <step-by-step>\nFinal answer: 49\n\nReasoning: <step-by-step>\nFinal answer: 49\n\nReasoning: <step-by-step>\nFinal answ...
- `layer18_t1` `train-000001` correct=False answer='45' gold='49' tokens=96:  First, find the total number of items in the box before any items are removed. Then, subtract the number of items removed from that total.\nAnswer: 45\n\nReasoning: First, find the total number of items in the box before any items are removed. Then, subtract ...
- `layer20_t1` `train-000001` correct=False answer='45' gold='49' tokens=96:  First, find the total number of items in the box before any items are removed. Then, subtract the number of items removed from that total.\nAnswer: 45\n\nReasoning: First, find the total number of items in the box before any items are removed. Then, subtract ...
- `layer27_t0.75` `train-000001` correct=True answer='49' gold='49' tokens=96:  9 rows × 6 items per row = 54 items. Then, 5 items are removed. So, 54 - 5 = 49 items remain.\nFinal answer: 49\n\nReasoning: 9 × 6 = 54. Then, 54 - 5 = 49. Final answer: 49\nFinal answer: 49\n\nOkay, let's see. The problem says there's
- `layer27_t1` `train-000001` correct=False answer='54' gold='49' tokens=96:  The box initially has 9 rows × 6 items per row = 54 items. Then 5 items are removed. So the answer is 54 - 5 = 49.\nFinal answer: 49\n\nReasoning: <brief arithmetic steps>\nFinal answer: 49\n\nReasoning: <brief arithmetic steps>\nFinal answer: 49\n\nReasoning...
- `baseline` `train-000002` correct=False answer='30' gold='105' tokens=96:  <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal...
- `layer18_t1` `train-000002` correct=False answer='30' gold='105' tokens=96:  <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal...
- `layer20_t1` `train-000002` correct=False answer='30' gold='105' tokens=96:  <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal answer: 30\n\nReasoning: <brief arithmetic steps>\nFinal...
- `layer27_t0.75` `train-000002` correct=True answer='105' gold='105' tokens=96:  20 + 15 = 35. Then, 35 * 3 = 105.\nFinal answer: 105\nAnswer: 105\n\nReasoning: 20 + 15 = 35. Then, 35 * 3 = 105.\nFinal answer: 105\n\nReasoning: 20 + 15 = 35. Then, 
- `layer27_t1` `train-000002` correct=False answer='60' gold='105' tokens=96:  The value of x is 20 + 15. Then, x times 3 is 3*(20 + 15). Then, 3*20 = 60 and 3*15 = 45. So, the final answer is 60 + 45 = 105.\nAnswer: 105\n\nReasoning: <brief arithmetic steps>\nFinal answer: 105\n\nReasoning
