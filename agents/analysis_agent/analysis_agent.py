# agents/analysis_agent.py

import pandas as pd
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from agno.agent.agent import Agent
from agno.run.response import RunResponse


# --- 1. Define the Pydantic schemas for inputs & outputs ---

class ExposureItem(BaseModel):
    ticker: str
    region: str
    aum_percentage: float = Field(..., description="Allocation as % of AUM")


class EarningsItem(BaseModel):
    ticker: str
    surprise_percentage: float = Field(..., description="Earnings surprise vs consensus (%)")


class AnalysisRequest(BaseModel):
    exposures: List[ExposureItem]
    earnings: List[EarningsItem]
    date: str = Field(..., description="Analysis date, YYYY-MM-DD")


class AnalysisResult(BaseModel):
    asia_tech_allocation: float
    asia_tech_allocation_change: float = None
    top_earnings_surprises: List[EarningsItem]
    sentiment_summary: str
    # full tables for auditability
    all_exposures: List[Dict[str, Any]]
    all_earnings: List[Dict[str, Any]]


# --- 2. Write the core logic in a run() function ---

def _analyze(data: AnalysisRequest) -> AnalysisResult:
    # Turn into DataFrames
    exp_df = pd.DataFrame([e.dict() for e in data.exposures])
    earn_df = pd.DataFrame([e.dict() for e in data.earnings])

    # Calculate Asia tech allocation
    asia = exp_df[exp_df.region == "Asia"]
    asia_alloc = float(asia.aum_percentage.sum())

    # (Optional) you can load a prior-day value here to compute change
    alloc_change = None

    # Top 3 earnings surprises
    top3 = earn_df.sort_values("surprise_percentage", ascending=False).head(3)
    top_items = [EarningsItem(**r) for r in top3.to_dict("records")]

    # Simple sentiment
    avg_surprise = float(earn_df.surprise_percentage.mean())
    if avg_surprise > 0:
        sentiment = "positive"
    elif avg_surprise < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    summary = f"Regional sentiment is {sentiment} with an average surprise of {avg_surprise:.1f}%."

    return AnalysisResult(
        asia_tech_allocation=asia_alloc,
        asia_tech_allocation_change=alloc_change,
        top_earnings_surprises=top_items,
        sentiment_summary=summary,
        all_exposures=exp_df.to_dict("records"),
        all_earnings=earn_df.to_dict("records"),
    )


# --- 3. Wrap it in an Agno Agent instance ---

analysis_agent = Agent(
    id="analysis_agent",
    name="Analysis Agent",
    description=(
        "Takes cleaned portfolio exposures and earnings surprises, "
        "calculates Asia-tech allocation, top earnings beats, sentiment, "
        "and returns structured metrics."
    ),
    # specify the Pydantic models for I/O validation
    input_model=AnalysisRequest,
    output_model=AnalysisResult,
    run=_analyze,
)
