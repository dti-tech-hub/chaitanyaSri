# Table 8. Arrival-rate sweep, mean +/- sample standard deviation over seeds 42, 43, 44

Source: `results/simulation/density_sweep.csv`. The EV wait is identical across seeds because the EV schedule is fixed and the controller delays are deterministic.

| Density | Seeds | Baseline EV wait (s) | Priority EV wait (s) | Baseline normal wait (s) | Priority normal wait (s) | Baseline throughput | Priority throughput | Priority events |
|---|---|---|---|---|---|---|---|---|
| low | 3 | 10.00 +/- 0.00 | 3.50 +/- 0.00 | 8.72 +/- 0.13 | 7.42 +/- 0.88 | 724 +/- 56 | 714 +/- 52 | 15 |
| medium | 3 | 10.00 +/- 0.00 | 3.50 +/- 0.00 | 8.50 +/- 1.07 | 8.53 +/- 0.46 | 1682 +/- 130 | 1684 +/- 137 | 15 |
| high | 3 | 10.00 +/- 0.00 | 3.50 +/- 0.00 | 8.37 +/- 0.27 | 8.80 +/- 0.50 | 3516 +/- 131 | 3492 +/- 101 | 15 |
