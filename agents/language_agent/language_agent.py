from typing import Any, Dict
from pydantic import BaseModel, Field
from agno.agent.agent import Agent
from agno.run.response import RunResponse


class AnalysisPayload(BaseModel):
    asia_tech_allocation: float = Field(..., description="Total allocation to Asia tech")
    asia_tech_allocation_change: float = Field(None, description="Day-over-day change in allocation")
    top_earnings_surprises: list = Field(..., description="List of top earnings surprise items")
    sentiment_summary: str = Field(..., description="Sentiment summary text from analysis agent")
    all_exposures: list = Field(..., description="Full exposure records for audit")
    all_earnings: list = Field(..., description="Full earnings records for audit")


class NarrativeResult(BaseModel):
    brief: str = Field(..., description="Short market brief narrative")
    detailed: str = Field(..., description="Expanded narrative with context and next steps")


def _generate_narrative(data: AnalysisPayload) -> NarrativeResult:
    # Build the brief summary
    brief = (
        f"Today's Asia tech allocation is {data.asia_tech_allocation:.1f}%"
    )
    if data.asia_tech_allocation_change is not None:
        sign = '+' if data.asia_tech_allocation_change >= 0 else ''
        brief += (
            f" ({sign}{data.asia_tech_allocation_change:.1f}% vs. previous day)"
        )
    brief += ". " + data.sentiment_summary

    # Build detailed narrative
    lines = []
    lines.append(f"As of today, our portfolio's allocation to Asia tech stands at {data.asia_tech_allocation:.1f}%.")
    if data.asia_tech_allocation_change is not None:
        lines.append(
            f"This marks a {sign}{data.asia_tech_allocation_change:.1f}% change from the prior day, "
            "indicating a shift in regional exposure."
        )
    lines.append(data.sentiment_summary.capitalize() + ".")
    lines.append("Top earnings surprises: ")
    for item in data.top_earnings_surprises:
        lines.append(
            f"  • {item.ticker}: {item.surprise_percentage:.1f}% surprise"
        )
    lines.append(
        "For a full breakdown of allocations and earnings surprises, see the attached details."  # downstream agent can format tables
    )
    detailed = "\n".join(lines)

    return NarrativeResult(brief=brief, detailed=detailed)


language_agent = Agent(
    id="language_agent",
    name="Language Agent",
    description=(
        "Converts analysis metrics into human-friendly narrative: "
        "a concise market brief and a detailed commentary."
    ),
    input_model=AnalysisPayload,
    output_model=NarrativeResult,
    run=_generate_narrative,
)
