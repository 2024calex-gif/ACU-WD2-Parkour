# dive4 placement scan

Full grid (2501 rows): scan.csv. Feasible = closer hand at or beyond the far edge + 0.20 m. Cost = shift^2 + 0.50 (s-1)^2 + 0.50 dW^2.

## Shift 0 at each scale (the foot-anchored baseline)

| s | shift (m) | Reference y at takeoff (m) | closer hand y (m) | hand clearance (m) | dW at W (m) | cost | feasible |
|---|---|---|---|---|---|---|---|
| 0.90 | 0.00 | +0.452 | 3.827 | -0.777 | -0.765 | 0.2979 | no |
| 1.00 | 0.00 | +0.517 | 3.981 | -0.623 | -0.516 | 0.1329 | no |
| 1.10 | 0.00 | +0.582 | 4.136 | -0.469 | -0.266 | 0.0403 | no |
| 1.20 | 0.00 | +0.648 | 4.290 | -0.315 | -0.016 | 0.0201 | no |
| 1.30 | 0.00 | +0.713 | 4.444 | -0.161 | +0.234 | 0.0724 | no |
| 1.40 | 0.00 | +0.778 | 4.598 | -0.007 | +0.484 | 0.1972 | no |
| 1.50 | 0.00 | +0.843 | 4.752 | +0.147 | +0.734 | 0.3944 | no |

Best reach in the box: s 1.50, shift 0.40, closer hand 5.152 m (clearance +0.547 m, feasible: yes).
Pick (lowest cost among the feasible rows): s 1.32, shift 0.33, Reference y at takeoff +1.056 m, dW +0.614 m, cost 0.3487.

