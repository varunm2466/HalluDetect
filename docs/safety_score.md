# HRS Safety Score

```
S_safety = 100 · clip( 1 − α·P_inj − β·U_intrinsic + γ·V_extrinsic , 0, 1 )
```

| Symbol | Range | Source |
|--------|-------|--------|
| `P_inj` | [0, 1] | `ThreatAggregator` (Sentinel logits + MELON cosine) |
| `U_intrinsic` | [0, 1] | `UncertaintyEstimator.normalize(estimate(...))` |
| `V_extrinsic` | [0, 1] | Verified-claim ratio over Layer 3+4 verdicts |
| α, β, γ | tunable | `configs/*.yaml` `scoring.weights` |

Default weights: `(α, β, γ) = (0.5, 0.3, 0.2)`.

Presets:

| Preset | α | β | γ | auto-apply ≥ | block on P_inj ≥ |
|--------|----|----|----|--------------|-------------------|
| **default**  | 0.5 | 0.3 | 0.2 | 80 | 0.6 |
| **strict**   | 0.6 | 0.35 | 0.05 | 92 | 0.4 |
| **lenient**  | 0.4 | 0.2 | 0.4 | 65 | 0.8 |

Interpretation buckets:

| Score | Meaning |
|-------|---------|
| 90–100 | Fully verified — auto-apply suggested patches |
| 75–89  | Mostly verified — review high-risk patches manually |
| 50–74  | Mixed signal — manual audit recommended |
| 25–49  | High risk — block automated rewrites |
| 0–24   | Critically compromised — discard generation |
