from halludetect.linkage.rabin_karp import RabinKarp, longest_common_substring_rk


def test_hashes_basic_lengths():
    rk = RabinKarp("abcabc")
    assert len(rk.hashes(3)) == 4
    assert len(rk.hashes(6)) == 1
    assert rk.hashes(7) == []


def test_lcs_finds_long_match():
    a = "the quick brown fox jumps over the lazy dog"
    b = "a brown fox jumps over the lazy fence"
    length, sub = longest_common_substring_rk(a, b)
    assert length >= len(" brown fox jumps over the lazy")
    assert "brown fox jumps over the lazy" in sub


def test_lcs_empty():
    assert longest_common_substring_rk("", "abc") == (0, "")
    assert longest_common_substring_rk("abc", "") == (0, "")
