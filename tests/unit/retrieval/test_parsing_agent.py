from pathlib import Path

from halludetect.retrieval.parsing_agent import ParsingAgent


def test_parses_inline_text():
    text = (
        "As shown by Smith and Doe (2025), contrastive learning helps. "
        "[1] also discusses this point.\n\n"
        "References\n"
        "[1] Doe, A. (2025). Some Title. Journal, 1, 1–10."
    )
    cites = ParsingAgent().parse_text(text)
    assert any("Smith" in (c.raw or "") for c in cites)


def test_parses_markdown_file(tmp_path: Path):
    p = tmp_path / "draft.md"
    p.write_text(
        "# Introduction\n\nGreshake et al. (2023) showed indirect prompt injection.\n\n"
        "## References\n[1] Greshake, K. (2023). Indirect injection. arXiv:2302.12173."
    )
    cites = ParsingAgent().parse_path(p)
    assert any("Greshake" in (c.raw or "") for c in cites)
