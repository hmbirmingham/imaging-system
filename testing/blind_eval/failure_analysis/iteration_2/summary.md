# Failure Analysis — Iteration 2

Plates selected as the most false negatives (missed colonies) per failing held-out set.

## dense_pack

| plate | expected | detected | false negatives | recall | count error % |
|---|---|---|---|---|---|
| `plate_021` | 116 | 74 | 42 | 0.638 | 36.21 |
| `plate_038` | 112 | 74 | 38 | 0.661 | 33.93 |
| `plate_043` | 116 | 79 | 37 | 0.681 | 31.90 |

## high_touching

| plate | expected | detected | false negatives | recall | count error % |
|---|---|---|---|---|---|
| `plate_020` | 40 | 18 | 22 | 0.450 | 55.00 |
| `plate_033` | 40 | 20 | 20 | 0.500 | 50.00 |
| `plate_039` | 29 | 10 | 19 | 0.345 | 65.52 |
