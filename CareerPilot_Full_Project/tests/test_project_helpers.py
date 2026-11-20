from backend.audio_analysis import filler_word_count, word_count


def test_word_count():
    assert word_count("Hello, world! I'm ready.") == 4


def test_word_count_empty_and_none_like_input():
    assert word_count("") == 0
    assert word_count(None) == 0


def test_filler_word_count():
    assert filler_word_count("Um, I think, like, we can do it, you know.") == 3


def test_filler_count_is_case_insensitive():
    assert filler_word_count("UM, uh, basically") == 3
