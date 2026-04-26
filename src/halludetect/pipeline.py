"""End-to-end orchestrator chaining all five layers.

    InputDocument → L1 (security) → L2 (extraction) → L3 (retrieval)
                  → L4 (linkage) → L5 (scoring + correction) → PipelineReport
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from datetime import datetime, timezone

import httpx

from .config import HallDetectConfig, load_config
from .exceptions import AbortAgentError
from .extraction.citation_aligner import CitationAligner, extract_citations
from .extraction.entailment import EntailmentScorer
from .extraction.srl import SemanticRoleLabeler
from .extraction.triplet_builder import TripletBuilder
from .extraction.uncertainty import UncertaintyEstimator
from .linkage.deduper import Deduper
from .logging_setup import get_logger
from .retrieval.liveness_agent import LivenessAgent
from .retrieval.reasoning_agent import ReasoningAgent
from .retrieval.retrieval_agent import RetrievalAgent
from .scoring.diagnostics import DiagnosticsBuilder
from .scoring.hrs import HRSAggregator
from .scoring.policy_gate import PolicyGate
from .scoring.rewrite_engine import RewriteEngine
from .security.melon import LLMCallable, MelonDefender, default_stub_llm
from .security.sentinel import SentinelClassifier
from .security.threat_report import ThreatAggregator
from .security.trajectory import Trajectory  # noqa: F401  (re-export for tests)
from .types import (
    Citation,
    Claim,
    InputDocument,
    LayerResult,
    PipelineReport,
    SafetyScore,
    Verdict,
    VerdictLabel,
)

log = get_logger(__name__)


# Re-export the offline stub so callers can opt-in explicitly.
_default_llm_callable = default_stub_llm


# ─────────────────────────────────────────────────────────────────────────────


class HallucinationDetectionPipeline:
    """Composable orchestrator for the full five-layer pipeline."""

    def __init__(
        self,
        config: HallDetectConfig | None = None,
        *,
        llm_callable: LLMCallable | None = None,
    ) -> None:
        self.config = config or load_config()
        self._llm = llm_callable or _default_llm_callable

        self.sentinel = SentinelClassifier(self.config.security.sentinel)
        self.melon = MelonDefender(self.config.security.melon)
        self.threat_aggregator = ThreatAggregator(
            melon_threshold=self.config.security.melon.cosine_threshold
        )

        self.srl = SemanticRoleLabeler(self.config.extraction.srl)
        self.triplet_builder = TripletBuilder(self.config.extraction.triplets)
        self.entailment = EntailmentScorer(self.config.extraction.entailment)
        self.uncertainty = UncertaintyEstimator(self.config.extraction.uncertainty)

        self.retrieval = RetrievalAgent(self.config.retrieval)
        self.liveness = LivenessAgent(self.config.retrieval.liveness)
        self.reasoner = ReasoningAgent(self.config.retrieval.reasoning)

        self.linker = Deduper(self.config.linkage)

        self.policy = PolicyGate(self.config.scoring.policy)
        self.rewrite = RewriteEngine(self.policy, self.config.scoring.rewrite)
        self.hrs = HRSAggregator(self.config.scoring.weights)

    # ------------------------------------------------------------------------

    def run(self, doc: InputDocument) -> PipelineReport:
        return asyncio.run(self.arun(doc))

    async def arun(self, doc: InputDocument) -> PipelineReport:
        started = datetime.now(timezone.utc)
        layer_results: list[LayerResult] = []
        aborted = False
        abort_reason: str | None = None

        # ── Layer 1 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        sentinel_result = self.sentinel.classify(doc.user_prompt)
        try:
            melon_result = self.melon.evaluate(
                user_prompt=doc.user_prompt,
                external_chunks=doc.external_context,
                llm_callable=self._llm,
                raise_on_abort=False,
            )
        except AbortAgentError as exc:
            log.warning("MELON abort raised in pipeline", error=str(exc))
            from .security.melon import MelonResult
            from .security.trajectory import Trajectory as _T
            melon_result = MelonResult(
                cosine_similarity=1.0,
                aborted=True,
                reason=str(exc),
                origin_trajectory=_T(),
                masked_trajectory=_T(),
            )
            aborted = True
            abort_reason = str(exc)

        threat = self.threat_aggregator.aggregate(
            sentinel_result, melon_result, sentinel_blocked=sentinel_result.is_malicious
        )
        layer_results.append(
            LayerResult(layer="L1-security", succeeded=True, duration_ms=(time.perf_counter() - t0) * 1000.0,
                        payload=threat.model_dump())
        )

        # If injection blocked, short-circuit but still produce a report.
        if threat.blocked:
            aborted = True
            abort_reason = abort_reason or "Sentinel/MELON blocked the input."
            score = self.hrs.compute(
                threat=threat,
                u_intrinsic_norm=1.0,
                verdicts=[],
            )
            return PipelineReport(
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                threat=threat,
                claims=[],
                verdicts=[],
                patches=[],
                safety=score,
                layer_results=layer_results,
                aborted=True,
                abort_reason=abort_reason,
            )

        # ── Layer 2 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        frames = self.srl.parse(doc.generated_text)
        triplets = self.triplet_builder.build(frames)
        citations = extract_citations(doc.generated_text)
        claims = CitationAligner.align(doc.generated_text, triplets, citations)
        # Pre-compute lexical entailment vs. anchor sentence (will be refined with retrieval evidence)
        for c in claims:
            c.entailment_score = self.entailment.score(c.triplet.sentence, " ".join(c.triplet.as_tuple()))
        layer_results.append(
            LayerResult(
                layer="L2-extraction",
                succeeded=True,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                payload={"claims": len(claims), "citations": len(citations)},
            )
        )

        # ── Layer 3 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        unique_citations = self._unique_citations(citations)
        async with httpx.AsyncClient(
            timeout=self.config.retrieval.http.timeout_s,
            follow_redirects=True,
        ) as client:
            retrieval_tasks = [self.retrieval.retrieve(c, client=client) for c in unique_citations]
            retrieval_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            liveness_tasks = [
                self.liveness.check(c.url or (c.doi or ""), client=client) if (c.url or c.doi) else None
                for c in unique_citations
            ]
            liveness_results = await asyncio.gather(
                *[t for t in liveness_tasks if t is not None],
                return_exceptions=True,
            ) if any(t is not None for t in liveness_tasks) else []

        retrieval_map: dict[int, list] = {}
        for idx, res in enumerate(retrieval_results):
            retrieval_map[idx] = [] if isinstance(res, Exception) else res

        # Map liveness results back to citations
        liveness_map: dict[int, object] = {}
        live_iter = iter(liveness_results)
        for idx, t in enumerate(liveness_tasks):
            if t is None:
                continue
            try:
                r = next(live_iter)
                liveness_map[idx] = None if isinstance(r, Exception) else r
            except StopIteration:
                break
        layer_results.append(
            LayerResult(
                layer="L3-retrieval",
                succeeded=True,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                payload={"queried": len(unique_citations)},
            )
        )

        # ── Layer 4 + Reasoning ─────────────────────────────────────────────
        t0 = time.perf_counter()
        verdicts: list[Verdict] = []
        verdict_tuples: list[tuple[Claim, VerdictLabel, object]] = []
        for idx, citation in enumerate(unique_citations):
            candidates = retrieval_map.get(idx, [])
            ranked = self.linker.link(citation, candidates)
            best = self.linker.best(citation, candidates) if candidates else None
            judgment = self.reasoner.judge(citation, candidates)
            label = judgment.label
            if best and best.similarity >= 0.92:
                label = VerdictLabel.VERIFIED
            elif best and best.similarity >= self.config.linkage.jaro_winkler.base_threshold:
                label = VerdictLabel.PARTIALLY_VERIFIED if label != VerdictLabel.VERIFIED else label

            verdict = Verdict(
                citation=citation,
                label=label,
                rationale=judgment.rationale,
                matches=ranked[:3],
                liveness=liveness_map.get(idx),
                safe_to_auto_apply=False,
            )
            verdicts.append(verdict)
            related_claims = [c for c in claims if c.citation and self._same_citation(c.citation, citation)]
            for cl in related_claims or [Claim(triplet=c.triplet, citation=citation) for c in claims if not c.citation][:1]:
                verdict_tuples.append((cl, label, best))
        if not verdict_tuples and claims:
            for c in claims:
                verdict_tuples.append((c, VerdictLabel.UNVERIFIABLE, None))
        layer_results.append(
            LayerResult(
                layer="L4-linkage",
                succeeded=True,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                payload={"verdicts": len(verdicts)},
            )
        )

        # ── Layer 5 ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        u_raw = self.uncertainty.estimate(claims=claims)
        u_norm = self.uncertainty.normalize(u_raw)
        safety_score: SafetyScore = self.hrs.compute(
            threat=threat,
            u_intrinsic_norm=u_norm,
            verdicts=verdict_tuples,
        )
        patches = self.rewrite.build_patches(verdicts=verdicts, threat=threat, safety_score=safety_score.score)
        for v, p in zip(verdicts, patches):
            v.safe_to_auto_apply = p.safe_to_apply

        layer_results.append(
            LayerResult(
                layer="L5-scoring",
                succeeded=True,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                payload={"safety": safety_score.score, "auto": sum(1 for p in patches if p.safe_to_apply)},
            )
        )

        return PipelineReport(
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            threat=threat,
            claims=claims,
            verdicts=verdicts,
            patches=patches,
            safety=safety_score,
            layer_results=layer_results,
            aborted=aborted,
            abort_reason=abort_reason,
        )

    # ------------------------------------------------------------------------
    # helpers

    @staticmethod
    def _unique_citations(citations: Iterable[Citation]) -> list[Citation]:
        seen: set[tuple[str, str, str]] = set()
        out: list[Citation] = []
        for c in citations:
            key = ((c.doi or "").lower(), (c.title or "").lower(), str(c.year or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    @staticmethod
    def _same_citation(a: Citation, b: Citation) -> bool:
        if a.doi and b.doi and a.doi.lower() == b.doi.lower():
            return True
        if a.title and b.title and a.title.lower() == b.title.lower():
            return True
        return bool(a.span and b.span and a.span == b.span)

    # ------------------------------------------------------------------------

    @staticmethod
    def render(report: PipelineReport) -> str:
        return DiagnosticsBuilder.render(report)

    @staticmethod
    def summarize(report: PipelineReport) -> dict:
        return DiagnosticsBuilder.to_dict(report)
