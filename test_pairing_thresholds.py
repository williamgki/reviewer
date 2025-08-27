from types import SimpleNamespace
from semantic_corpus_sampler import filter_candidates


def make_result(doc_id, content, score):
    return SimpleNamespace(doc_id=doc_id, content=content, combined_score=score)


def test_filters_apply():
    config = {
        "min_pairing_score": 0.38,
        "domain_whitelist": ["arxiv.org"],
        "require_entity_overlap": True,
    }
    results = [
        make_result("http://arxiv.org/abs/1", "alignment study", 0.5),
        make_result("http://example.com/1", "alignment study", 0.6),
        make_result("http://arxiv.org/abs/2", "random text", 0.7),
        make_result("http://arxiv.org/abs/3", "alignment study", 0.1),
    ]
    filtered = filter_candidates(results, config, "alignment research")
    assert len(filtered) == 1
    assert filtered[0].doc_id.endswith("/1")
