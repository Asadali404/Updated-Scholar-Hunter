"""Pydantic data models shared by all agents."""
from typing import Optional
from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    degree: str = "Unknown"
    cgpa: str = "Unknown"
    publications: int = 0
    skills: list[str] = Field(default_factory=list)
    research_areas: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    summary: str = ""


class RawResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    excerpt: str = ""      # page text fetched for verification
    page_ok: bool = False  # link opened successfully (HTTP < 400)
    query: str = ""


class ScholarshipRecord(BaseModel):
    name: str
    provider: str = ""
    country: str = ""
    level: str = "Any"
    funding: str = ""
    eligibility: str = ""
    requirements: str = ""             # comma-separated: IELTS, GRE, SOP, Proposal, CV, Referees
    deadline: str = ""                 # ISO date (YYYY-MM-DD) or ""
    days_left: Optional[int] = None
    official_link: str = ""
    source: str = "web"                # web | curated | demo
    verified: bool = False
    confidence: str = "Low"            # High | Medium | Low | Estimated
    verification_note: str = ""
    fit_score: int = 0
    weaknesses: list[str] = Field(default_factory=list)


class GapItem(BaseModel):
    scholarship: str
    weaknesses: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    priority: str = "Medium"
    prep_time: str = ""


class GapReport(BaseModel):
    strengths: list[str] = Field(default_factory=list)
    common_gaps: list[str] = Field(default_factory=list)
    items: list[GapItem] = Field(default_factory=list)


class PipelineResult(BaseModel):
    mode: str = "Live"
    profile: Optional[CandidateProfile] = None
    records: list[ScholarshipRecord] = Field(default_factory=list)
    rejected: list[dict] = Field(default_factory=list)
    gap_report: Optional[GapReport] = None
    tracker_summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    failed_stage: str = ""
    error: str = ""
    rate_limited: bool = False
