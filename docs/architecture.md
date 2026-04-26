# Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          InputDocument                               │
└──────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────▼──────────────────┐
            │  Layer 1 — Security                 │
            │   • Sentinel (ModernBERT-large)     │
            │   • MELON (contrastive masking)     │
            │   • ThreatAggregator → P_inj        │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────▼──────────────────┐
            │  Layer 2 — Extraction               │
            │   • SRL → atomic triplets           │
            │   • Citation alignment              │
            │   • DeBERTa NLI entailment          │
            │   • Token entropy → U_intrinsic     │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────▼──────────────────┐
            │  Layer 3 — Retrieval                │
            │   • Crossref / arXiv / S2 / PubMed  │
            │   • Cascaded multi-pass             │
            │   • Liveness + Wayback CDX          │
            │   • LLM-as-Judge reasoner           │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────▼──────────────────┐
            │  Layer 4 — Linkage                  │
            │   • Enhanced Jaro-Winkler           │
            │   • Rabin-Karp rolling hash         │
            │   • Manifestation hierarchy         │
            └─────────────────┬──────────────────┘
                              │
            ┌─────────────────▼──────────────────┐
            │  Layer 5 — Scoring & Correction     │
            │   • Policy-Gated Rewrite            │
            │   • HRS Safety Score (0-100)        │
            └─────────────────┬──────────────────┘
                              │
                              ▼
                       PipelineReport
```

The whole pipeline is composable; you can short-circuit any layer with a stub
in your own ``HallucinationDetectionPipeline`` subclass.
