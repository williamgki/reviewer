from review_schema import ReviewCard


def _escape_md(s: str) -> str:
    return s.replace("|", "∣").replace("\n", " ").strip()


def render_review_card(card: ReviewCard, mode: str = "deep") -> str:
    lines = []
    lines.append(f"**{card.paper_id} — Review Card**")
    lines.append(f"**TL;DR:** {card.tl_dr}")
    lines.append("")
    lines.append(f"**Verdict:** {card.verdict}")
    if getattr(card, "metrics", None) and card.metrics.anchor_coverage is not None:
        lines.append(f"*Anchor coverage:* {card.metrics.anchor_coverage:.2f}")
    lines.append("")
    if card.strengths:
        lines.append("**Strengths**")
        for s in card.strengths:
            lines.append(f"* {s}")
        lines.append("")
    if card.weaknesses:
        lines.append("**Weaknesses**")
        for w in card.weaknesses:
            lines.append(f"* {w}")
        lines.append("")
    if card.key_claims:
        lines.append("**Key Claims**")
        lines.append("| ID | Claim | Conf. | Paper Anchor | Corpus Anchor |")
        lines.append("| -- | ----- | ----: | ------------ | ------------- |")

        def _sort_key(cl):
            imp = {"high": 2, "med": 1, "low": 0}.get(cl.importance, 0)
            return (-imp, -cl.confidence)

        claims = sorted(card.key_claims, key=_sort_key)
        if mode == "quick":
            claims = claims[:2]

        for cl in claims:
            pq = cl.evidence.paper_quotes[0]
            cq = cl.evidence.corpus_quotes[0]
            claim = _escape_md(cl.statement)
            paper = _escape_md(f"{pq.doc_id} — {pq.locator}")
            corpus = _escape_md(f"{cq.doc_id} — {cq.locator}")
            lines.append(f"| {cl.id} | {claim} | {cl.confidence:.2f} | {paper} | {corpus} |")
        lines.append("")
    if card.suggested_experiments:
        lines.append("**Suggested Experiments**")
        lines.append("| ID | Goal | Setup | Success |")
        lines.append("| -- | ---- | ----- | ------- |")
        for ex in card.suggested_experiments:
            lines.append(f"| {ex.id} | {ex.goal} | {ex.setup} | {ex.success_criteria} |")
        lines.append("")
    if card.related_work_gaps:
        lines.append("**Related-work gaps**")
        for g in card.related_work_gaps:
            lines.append(f"* {g}")
        lines.append("")
    if card.open_questions:
        lines.append("**Open Questions**")
        for q in card.open_questions:
            lines.append(f"* {q}")
        lines.append("")
    return "\n".join(lines)
