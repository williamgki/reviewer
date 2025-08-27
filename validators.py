from typing import Dict, Set
from pydantic import ValidationError
from review_schema import ReviewCard, ReviewMetrics


def validate_review_card(data: Dict) -> ReviewCard:
    """Validate dict against ReviewCard schema and enforce structural checks."""
    card = ReviewCard(**data)

    if len(card.tl_dr.split()) > 60:
        raise ValueError("TL;DR must be ≤ 60 words")

    seen: Set[str] = set()
    for claim in card.key_claims:
        if claim.id in seen:
            raise ValueError(f"Duplicate claim id: {claim.id}")
        seen.add(claim.id)
        if not (0.0 <= claim.confidence <= 1.0):
            raise ValueError(f"Claim {claim.id} confidence not in [0,1]")
        if not claim.evidence.paper_quotes or not claim.evidence.corpus_quotes:
            raise ValueError(f"Claim {claim.id} missing required quotes")

    card.metrics = ReviewMetrics(anchor_coverage=coverage(card))
    return card


def coverage(card: ReviewCard) -> float:
    """Compute fraction of claims with both paper and corpus quotes."""
    if not card.key_claims:
        return 0.0
    supported = sum(
        1 for c in card.key_claims if c.evidence.paper_quotes and c.evidence.corpus_quotes
    )
    return supported / len(card.key_claims)
