# Simulated fixed-time vs emergency-priority comparison

Seed: 42 | Duration: 600.0 s | Vehicles: 287 | Emergency vehicles: 5

| Metric | Fixed-time baseline | Proposed priority | Change |
|---|---:|---:|---:|
| Average emergency waiting time (s) | 10.00 | 3.50 | -6.50 |
| Maximum emergency waiting time (s) | 26.00 | 3.50 | -22.50 |
| Average normal vehicle waiting time (s) | 9.71 | 8.97 | -0.74 |
| Priority response time (s) | Not evaluated | 1.50 | - |
| Vehicle throughput (vehicles/hour) | 1704.00 | 1710.00 | +6.00 |
| Priority activations | 0.00 | 5.00 | +5.00 |

This is a deterministic signal-level simulation. It does not model queue spillback, driver behaviour, pedestrian phases, or real-world safety.
