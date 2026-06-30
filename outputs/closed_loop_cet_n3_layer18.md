# Closed-Loop CET Controller

## Summary

| controller | n | accuracy | strict stop rate | mean tokens | repetition | mean alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | 0.000 | 0.000 | 64.00 | 0.0000 | 0.000 |
| static_mean_direction | 3 | 0.000 | 0.000 | 64.00 | 0.0000 | 1.000 |
| closed_loop_cet | 3 | 0.000 | 0.000 | 64.00 | 0.0000 | 0.275 |

## Examples
- `baseline` `heldout-000000` answer=None gold='148' correct=False tokens=64 mean_alpha=0.000:  <brief reasoning>\nFinal answer: <number>\n\nOkay, let's see. Mia starts with 22 tokens. Then, for each of 9 rounds, she gets 14 more tokens. So, I need to figure out how many tokens she has in total after all those rou...
- `static_mean_direction` `heldout-000000` answer='1' gold='148' correct=False tokens=64 mean_alpha=1.000:  <Step 1, < < For ... <  < < < < < ... < ... < < < < < < ... < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < < <1
- `closed_loop_cet` `heldout-000000` answer=None gold='148' correct=False tokens=64 mean_alpha=0.320:  <Step 1, < < For ...s your, < < < <s\n >\n <: <s < < < < <,\n < < < < <brief < < < < < <\n < <\n < < < < < < < < < < < < < < < < <
- `baseline` `heldout-000001` answer=None gold='257' correct=False tokens=64 mean_alpha=0.000:  <brief reasoning>\nFinal answer: <number>\n\nOkay, let's see. Mia starts with 15 tokens. Then, for each of 11 rounds, she gets 22 more tokens. So, I need to find out how many tokens she has in total after all those roun...
- `static_mean_direction` `heldout-000001` answer=None gold='257' correct=False tokens=64 mean_alpha=1.000:  <Step>\n < ... <\n I I we we I You ... <\n we you, < <, < we, < < | | < < | < < < ... | ... ... < < <, < < < | < < < | < < ... |, | * < < < < < <
- `closed_loop_cet` `heldout-000001` answer=None gold='257' correct=False tokens=64 mean_alpha=0.258:  <Step>\n < ... Mia as we ( < < is Here <1\n::Reason: < and <> < < ( is < < < <number reasoning <number:: < < < < |:| < < < < < < < < < < < < ( < < < the < <
- `baseline` `heldout-000002` answer='20' gold='355' correct=False tokens=64 mean_alpha=0.000:  <brief reasoning>\nFinal answer: <number>\n\nOkay, let's see. The problem says there's a box with 18 rows, and each row has 20 items. So first, I need to figure out how many items are there initially. That would be 18 m...
- `static_mean_direction` `heldout-000002` answer=None gold='355' correct=False tokens=64 mean_alpha=1.000:  <1> < ... I < <1 ... I ... < we here here then previous, < ( ... < < < ( t. ( then is < < we < t ( < < < < < < < < < < < < < < < < < < < < < < < < < < <
- `closed_loop_cet` `heldout-000002` answer=None gold='355' correct=False tokens=64 mean_alpha=0.246:  <1> <.'s< <Then then I I then ( < <.: <\n < < (olving is ( ( ( ( has the ( <number: < ( ( ( ( < < < < < < < < ( < < < < <number < < < < < < < < <
