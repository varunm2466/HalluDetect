"""Standalone benign-path demo — verifies L2-L5 produce non-empty output."""

from halludetect.config import load_config
from halludetect.pipeline import HallucinationDetectionPipeline
from halludetect.types import InputDocument


def main() -> None:
    cfg = load_config()
    pipe = HallucinationDetectionPipeline(cfg)
    doc = InputDocument(
        user_prompt="Write a short related-work paragraph on indirect prompt injection.",
        generated_text=(
            "Greshake et al. (2023) introduced the concept of indirect prompt injection "
            "and demonstrated its effectiveness against retrieval-augmented agents. "
            "MELON was proposed by Zhu et al. (2025) as a contrastive defense that "
            "executes the agent twice with masked user prompts to detect behavioral "
            "drift caused by adversarial external content. A fabricated 2099 paper by "
            "Hacker, A. claims a 100% defense rate (Hacker, 2099).\n\n"
            "References\n"
            "[1] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., "
            "& Fritz, M. (2023). Not what you've signed up for: Compromising "
            "real-world LLM-integrated applications with indirect prompt injection. "
            "arXiv:2302.12173.\n"
            "[2] Zhu, K., et al. (2025). MELON: Provable Defense Against Indirect "
            "Prompt Injection Attacks in AI Agents. ICML.\n"
            "[3] Hacker, A. (2099). Pwning all LLMs forever. Journal of Imaginary AI. "
            "https://doi.org/10.9999/never/exists\n"
        ),
        external_context=[],
    )
    report = pipe.run(doc)
    print(pipe.render(report))


if __name__ == "__main__":
    main()
