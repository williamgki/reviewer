from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Quote(BaseModel):
    source: Literal["paper", "corpus"]
    doc_id: str
    locator: str
    text: str

class Evidence(BaseModel):
    paper_quotes: List[Quote] = Field(default_factory=list)
    corpus_quotes: List[Quote] = Field(default_factory=list)
    coverage_score: float

class Claim(BaseModel):
    id: str
    statement: str
    importance: Literal["low", "med", "high"]
    confidence: float
    evidence: Evidence
    risks: List[str] = Field(default_factory=list)

class Experiment(BaseModel):
    id: str
    goal: str
    setup: str
    success_criteria: str
    blockers: List[str] = Field(default_factory=list)

class ReviewCard(BaseModel):
    paper_id: str
    tl_dr: str
    verdict: Literal["accept", "weak accept", "borderline", "weak reject", "reject"]
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    key_claims: List[Claim] = Field(default_factory=list)
    suggested_experiments: List[Experiment] = Field(default_factory=list)
    related_work_gaps: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    confidence_calibration: str
