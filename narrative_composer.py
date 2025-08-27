from review_schema import ReviewCard


def render_narrative(card: ReviewCard) -> str:
    lines = ["### What to think about next"]
    for claim in card.key_claims:
        lines.append(f"- Reflect on {claim.id}: {claim.statement}")
    return "\n".join(lines)
