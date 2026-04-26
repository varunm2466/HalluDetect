"""Pretty-printable diagnostics + structured summaries."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..types import PipelineReport, VerdictLabel


class DiagnosticsBuilder:
    @staticmethod
    def to_dict(report: PipelineReport) -> dict:
        counts = {label.value: 0 for label in VerdictLabel}
        for v in report.verdicts:
            counts[v.label.value] += 1
        return {
            "safety_score": report.safety.score,
            "p_injection": report.safety.p_injection,
            "u_intrinsic": report.safety.u_intrinsic,
            "v_extrinsic": report.safety.v_extrinsic,
            "verdict_counts": counts,
            "auto_apply_patches": sum(1 for p in report.patches if p.safe_to_apply),
            "manual_review_patches": sum(1 for p in report.patches if p.requires_manual_review),
            "aborted": report.aborted,
            "abort_reason": report.abort_reason,
        }

    @staticmethod
    def render(report: PipelineReport) -> str:
        buf = StringIO()
        console = Console(file=buf, record=True, width=110, force_terminal=False)

        score = report.safety.score
        console.print(
            Panel(
                f"[bold]Safety Score[/bold]: {score:5.1f} / 100\n[dim]{report.safety.interpretation}[/dim]",
                title="halludetect — Hallucination Risk Score",
            )
        )

        threat_table = Table(title="Layer 1 — Threat report")
        threat_table.add_column("Signal")
        threat_table.add_column("Value", justify="right")
        threat_table.add_row("Sentinel label", report.threat.sentinel.label)
        threat_table.add_row("Sentinel score", f"{report.threat.sentinel.score:.3f}")
        threat_table.add_row("MELON cosine", f"{report.threat.melon.cosine_similarity:.3f}")
        threat_table.add_row("MELON aborted", str(report.threat.melon.aborted))
        threat_table.add_row("P_inj (aggregated)", f"{report.threat.p_injection:.3f}")
        console.print(threat_table)

        verdict_table = Table(title="Layer 2-4 — Verdicts")
        verdict_table.add_column("Citation")
        verdict_table.add_column("Verdict")
        verdict_table.add_column("Best match", overflow="fold")
        verdict_table.add_column("Sim", justify="right")
        for v in report.verdicts:
            best = v.matches[0] if v.matches else None
            sim = f"{best.similarity:.2f}" if best else "—"
            best_title = best.record.title if best else "—"
            verdict_table.add_row(
                (v.citation.title or v.citation.raw or "?")[:48],
                v.label.value,
                (best_title or "—")[:48],
                sim,
            )
        console.print(verdict_table)

        patch_table = Table(title="Layer 5 — Patches")
        patch_table.add_column("Target")
        patch_table.add_column("Action")
        patch_table.add_column("Auto?")
        patch_table.add_column("Rationale", overflow="fold")
        for p in report.patches:
            patch_table.add_row(
                (p.target_citation.title or p.target_citation.raw or "?")[:48],
                p.action.value,
                "yes" if p.safe_to_apply else "manual",
                "; ".join(p.notes)[:60],
            )
        console.print(patch_table)
        return buf.getvalue()
