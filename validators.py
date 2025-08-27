from typing import Dict
from pydantic import ValidationError
from review_schema import ReviewCard


def validate_review_card(data: Dict) -> ReviewCard:
    """Validate dict against ReviewCard schema and ensure evidence coverage."""
    card = ReviewCard(**data)
    for claim in card.key_claims:
        if not claim.evidence.paper_quotes or not claim.evidence.corpus_quotes:
            raise ValueError(f"Claim {claim.id} missing required quotes")
    return card


def coverage(card: ReviewCard) -> float:
    """Compute fraction of claims with both paper and corpus quotes."""
    if not card.key_claims:
        return 0.0
    supported = sum(
        1 for c in card.key_claims if c.evidence.paper_quotes and c.evidence.corpus_quotes
    )
    return supported / len(card.key_claims)
