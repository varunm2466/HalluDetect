"""Enhanced Jaro-Winkler with Rabin-Karp suffix bonus.

Background
----------
Standard Jaro-Winkler favors common *prefixes* — useful for surnames but
brittle for long academic titles where:

    * abbreviations move shared content to the middle ("Proc. NeurIPS 2024"
      vs. "Proceedings of NeurIPS 2024"),
    * suffixes carry highly informative tokens ("…via Contrastive Learning").

Enhancement (per the design doc)
--------------------------------
1. Compute classic Jaro similarity.
2. Apply the standard Winkler prefix bonus.
3. Use Rabin-Karp rolling hash to locate the longest common substring (LCS)
   anywhere in the strings; convert its length into a bounded *suffix-aware*
   bonus.
4. Return ``min(1.0, jaro + prefix_bonus + suffix_bonus)``.

This restores accuracy at high thresholds (≥0.8) where vanilla JW degrades.
"""

from __future__ import annotations

from .rabin_karp import longest_common_substring_rk

# ─────────────────────────────────────────────────────────────────────────────
# Classic Jaro
# ─────────────────────────────────────────────────────────────────────────────


def jaro_similarity(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0 if s1 else 0.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (
        matches / len1
        + matches / len2
        + (matches - transpositions) / matches
    ) / 3.0


def _common_prefix(s1: str, s2: str, limit: int = 4) -> int:
    cap = min(limit, len(s1), len(s2))
    n = 0
    while n < cap and s1[n] == s2[n]:
        n += 1
    return n


def jaro_winkler(s1: str, s2: str, *, prefix_scale: float = 0.10) -> float:
    j = jaro_similarity(s1, s2)
    if j == 0.0:
        return 0.0
    prefix = _common_prefix(s1, s2)
    return j + prefix * prefix_scale * (1.0 - j)


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced JW
# ─────────────────────────────────────────────────────────────────────────────


def enhanced_jaro_winkler(
    s1: str,
    s2: str,
    *,
    prefix_scale: float = 0.10,
    suffix_scale: float = 0.05,
    rabin_karp_window: int = 6,
) -> float:
    """Jaro-Winkler with a Rabin-Karp-driven suffix bonus.

    Parameters
    ----------
    prefix_scale, suffix_scale: bonus weights on the [0, 0.25] interval each.
    rabin_karp_window: minimum window the LCS must reach before contributing.
    """

    if not s1 or not s2:
        return 0.0
    s1_n = s1.lower().strip()
    s2_n = s2.lower().strip()

    j = jaro_similarity(s1_n, s2_n)
    if j == 0.0:
        return 0.0
    prefix = _common_prefix(s1_n, s2_n)
    score = j + prefix * prefix_scale * (1.0 - j)

    lcs_len, _ = longest_common_substring_rk(s1_n, s2_n)
    if lcs_len >= rabin_karp_window:
        denom = max(min(len(s1_n), len(s2_n)), 1)
        suffix_ratio = min(lcs_len / denom, 1.0)
        score += suffix_scale * suffix_ratio * (1.0 - score)

    return float(min(1.0, max(0.0, score)))
