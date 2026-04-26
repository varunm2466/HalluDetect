from halludetect.linkage.jaro_winkler import enhanced_jaro_winkler, jaro_similarity, jaro_winkler


def test_jaro_identity():
    assert jaro_similarity("attention is all you need", "attention is all you need") == 1.0


def test_jaro_disjoint():
    assert jaro_similarity("aaaa", "zzzz") == 0.0


def test_jw_prefix_bonus():
    a = "Attention Is All You Need"
    b = "Attention Is All Yu Need"
    assert jaro_winkler(a, b) > jaro_similarity(a, b)


def test_enhanced_recovers_long_titles():
    a = "Attention is all you need: a survey of transformer architectures"
    b = "A survey of transformer architectures: attention is all you need"
    score = enhanced_jaro_winkler(a, b)
    assert score >= 0.7


def test_enhanced_against_garbage_is_lower_than_real_match():
    a = "Attention is all you need"
    garbage = enhanced_jaro_winkler(a, "Quantum entanglement and bird migration")
    near_match = enhanced_jaro_winkler(a, "attention is all u need")
    same = enhanced_jaro_winkler(a, "Attention is all you need")
    assert garbage < near_match < same
    # Below the canonical 0.88 base linkage threshold.
    assert garbage < 0.7
