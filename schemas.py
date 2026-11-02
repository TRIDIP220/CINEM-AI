from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class RecommendRequest(BaseModel):
    title: str
    top_n: int = Field(default=8, ge=1, le=20)
    review: str | None = None


class ExplainRequest(BaseModel):
    selected_title: str
    recommended_title: str
    user_review: str | None = None
