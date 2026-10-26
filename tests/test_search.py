from tabular_lab.config import ExperimentSpec
from tabular_lab.search import choose_random, search_space


def test_shared_space_has_real_preprocessing_choices_and_no_duplicates():
    candidates = search_space()
    fingerprints = [c.fingerprint() for c in candidates]
    assert len(fingerprints) == len(set(fingerprints))
    assert any(not c.preprocessing.scale for c in candidates)
    assert any(c.preprocessing.min_frequency > 1 for c in candidates)
    assert any(c.preprocessing.select_percentile < 100 for c in candidates)
    assert any(c.preprocessing.impute == "most_frequent" for c in candidates)
    for candidate in candidates:
        assert ExperimentSpec.model_validate(candidate.model_dump()).fingerprint() == candidate.fingerprint()


def test_random_search_repeatable_and_avoids_attempted_candidates():
    first = choose_random([], 42)
    assert first.fingerprint() == choose_random([], 42).fingerprint()
    history = [{"fingerprint": first.fingerprint()}]
    assert choose_random(history, 42).fingerprint() != first.fingerprint()
