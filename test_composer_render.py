from review_schema import ReviewCard, Claim, Evidence, Quote, Experiment
from card_composer import render_review_card

def test_markdown_contains_anchors_and_table():
    pq = Quote(source="paper", doc_id="paper.pdf", locator="p.1 §1", text="paper quote")
    cq = Quote(source="corpus", doc_id="corpus.txt", locator="§2", text="corpus quote")
    ev = Evidence(paper_quotes=[pq], corpus_quotes=[cq], coverage_score=1.0)
    claim = Claim(id="C1", statement="Test claim", importance="high", confidence=0.9, evidence=ev, risks=[])
    card = ReviewCard(
        paper_id="paper-1",
        tl_dr="summary",
        verdict="accept",
        strengths=["good"],
        weaknesses=["bad"],
        key_claims=[claim],
        suggested_experiments=[Experiment(id="E1", goal="goal", setup="setup", success_criteria="pass")],
        related_work_gaps=["gap"],
        open_questions=["question"],
        confidence_calibration="because",
    )
    md = render_review_card(card)
    assert "| ID | Claim" in md
    assert "[Paper: p.1 §1]" in md
    assert "[Corpus: §2]" in md
