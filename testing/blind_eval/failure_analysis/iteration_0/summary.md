# Failure Analysis — Iteration 0

Plates selected as the most false negatives (missed colonies) per failing held-out set.

## size_variance

| plate | expected | detected | false negatives | recall | count error % |
|---|---|---|---|---|---|
| `plate_012` | 20 | 9 | 11 | 0.450 | 55.00 |
| `plate_023` | 19 | 10 | 9 | 0.526 | 47.37 |
| `plate_039` | 17 | 8 | 9 | 0.471 | 52.94 |

## dense_pack

| plate | expected | detected | false negatives | recall | count error % |
|---|---|---|---|---|---|
| `plate_038` | 112 | 58 | 54 | 0.518 | 48.21 |
| `plate_021` | 116 | 63 | 53 | 0.543 | 45.69 |
| `plate_043` | 116 | 68 | 48 | 0.586 | 41.38 |

## high_touching

| plate | expected | detected | false negatives | recall | count error % |
|---|---|---|---|---|---|
| `plate_020` | 40 | 17 | 23 | 0.425 | 57.50 |
| `plate_033` | 40 | 18 | 22 | 0.450 | 55.00 |
| `plate_039` | 29 | 9 | 20 | 0.310 | 68.97 |
