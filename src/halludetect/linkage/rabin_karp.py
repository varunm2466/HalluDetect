"""Rabin-Karp rolling hash — substring discovery for the Enhanced JW algorithm.

The Enhanced Jaro-Winkler implementation uses Rabin-Karp to rapidly find
matching substrings of varying lengths anywhere in the candidate strings,
and to compute a *bonus* for shared suffixes (counterbalancing the classic
JW prefix bias against long titles).
"""

from __future__ import annotations

from dataclasses import dataclass

_DEFAULT_BASE = 257
_DEFAULT_MOD = (1 << 61) - 1


@dataclass
class RabinKarp:
    """Polynomial rolling hash over a string."""

    text: str
    base: int = _DEFAULT_BASE
    mod: int = _DEFAULT_MOD

    def hashes(self, window: int) -> list[int]:
        """Return the hash of every length-``window`` substring (left to right)."""

        if window <= 0 or window > len(self.text):
            return []
        h = 0
        for i in range(window):
            h = (h * self.base + ord(self.text[i])) % self.mod
        out = [h]
        high_pow = pow(self.base, window - 1, self.mod)
        for i in range(window, len(self.text)):
            h = (h - ord(self.text[i - window]) * high_pow) % self.mod
            h = (h * self.base + ord(self.text[i])) % self.mod
            out.append(h)
        return out

    def positions(self, window: int) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i, h in enumerate(self.hashes(window)):
            out.setdefault(h, []).append(i)
        return out


def longest_common_substring_rk(a: str, b: str, *, max_window: int | None = None) -> tuple[int, str]:
    """Return ``(length, substring)`` of the longest common substring.

    Uses Rabin-Karp + binary search over window sizes for O((n+m) log min(n,m))
    expected time; verifies hash collisions by exact comparison.
    """

    if not a or not b:
        return 0, ""

    upper = min(len(a), len(b))
    if max_window is not None:
        upper = min(upper, max_window)

    rk_a = RabinKarp(a)
    rk_b = RabinKarp(b)

    lo, hi = 1, upper
    best_len, best_str = 0, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        match = _find_common_window(rk_a, rk_b, mid)
        if match:
            best_len, best_str = mid, match
            lo = mid + 1
        else:
            hi = mid - 1
    return best_len, best_str


def _find_common_window(rk_a: RabinKarp, rk_b: RabinKarp, window: int) -> str | None:
    pos_a = rk_a.positions(window)
    pos_b = rk_b.positions(window)
    common_hashes = set(pos_a) & set(pos_b)
    for h in common_hashes:
        for i in pos_a[h]:
            cand = rk_a.text[i : i + window]
            for j in pos_b[h]:
                if rk_b.text[j : j + window] == cand:
                    return cand
    return None
