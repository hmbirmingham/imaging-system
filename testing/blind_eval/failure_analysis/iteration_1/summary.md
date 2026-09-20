# Failure Analysis — Iteration 1

Plates selected as the most false negatives (missed colonies) per failing held-out set.

## dense_pack

| plate | expected | detected | false negatives | recall | count error % |
|---|---|---|---|---|---|
| `plate_021` | 116 | 73 | 43 | 0.629 | 37.07 |
| `plate_038` | 112 | 73 | 39 | 0.652 | 34.82 |
| `plate_001` | 109 | 71 | 38 | 0.651 | 34.86 |

## high_touching

| plate | expected | detected | false negatives | recall | count error % |
|---|---|---|---|---|---|
| `plate_020` | 40 | 18 | 22 | 0.450 | 55.00 |
| `plate_033` | 40 | 20 | 20 | 0.500 | 50.00 |
| `plate_039` | 29 | 9 | 20 | 0.310 | 68.97 |
