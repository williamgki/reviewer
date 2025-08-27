from review_schema import ReviewCard


def render_review_card(card: ReviewCard) -> str:
    lines = []
    lines.append(f"**{card.paper_id} — Review Card**")
    lines.append(f"**TL;DR:** {card.tl_dr}")
    lines.append("")
    lines.append(f"**Verdict:** {card.verdict}")
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
        for cl in card.key_claims:
            pq = cl.evidence.paper_quotes[0]
            cq = cl.evidence.corpus_quotes[0]
            lines.append(
                f"| {cl.id} | {cl.statement} | {cl.confidence:.2f} | [Paper: {pq.locator}] | [Corpus: {cq.locator}] |"
            )
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
