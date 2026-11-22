"""Tests for all 8 specialized single-responsibility agents."""
import pytest
from app.agents.resume_parser import ResumeParserAgent
from app.agents.jd_analyzer import JDAnalyzerAgent
from app.agents.matching_agent import MatchingAgent
from app.agents.skill_gap_agent import SkillGapAgent
from app.agents.resume_optimizer import ResumeOptimizerAgent
from app.agents.cover_letter_agent import CoverLetterAgent
from app.agents.interview_prep_agent import InterviewPrepAgent
from app.agents.research_agent import CompanyResearchAgent
from app.core.schemas import ParsedResume, ParsedJD


@pytest.mark.asyncio
async def test_resume_parser_agent():
    raw_resume = "Alex Rivera\nSenior ML Engineer\nSkills: Python, PyTorch, RAG, AWS\nNeuralScale AI"
    parsed = await ResumeParserAgent.run(raw_resume)
    assert isinstance(parsed, ParsedResume)
    assert parsed.contact.name != ""
    assert len(parsed.technical_skills) > 0


@pytest.mark.asyncio
async def test_jd_analyzer_agent():
    raw_jd = "Senior ML Engineer at Synthetix AI. Requirements: 5+ years experience, Python, PyTorch, RAG."
    parsed = await JDAnalyzerAgent.run(raw_jd)
    assert isinstance(parsed, ParsedJD)
    assert parsed.role_title != ""
    assert len(parsed.required_skills) > 0


@pytest.mark.asyncio
async def test_all_downstream_agents():
    # Setup parsed inputs
    resume = await ResumeParserAgent.run("Alex Rivera, Senior ML Engineer")
    jd = await JDAnalyzerAgent.run("Senior ML Engineer at Synthetix AI")

    # Matching Agent
    match_breakdown = await MatchingAgent.run(resume, jd, "Alex Rivera ML", "Synthetix ML")
    assert 0 <= match_breakdown.overall_score <= 100

    # Skill Gap Agent
    skill_gap = await SkillGapAgent.run(resume, jd)
    assert len(skill_gap.matched_skills) > 0 or len(skill_gap.missing_skills) > 0

    # Resume Optimizer Agent
    opt_report = await ResumeOptimizerAgent.run(resume, jd)
    assert len(opt_report.bullet_rewrites) > 0

    # Cover Letter Agent
    cover_letter = await CoverLetterAgent.run(resume, jd, tone="confident")
    assert "Synthetix" in cover_letter.letter_body or "Alex" in cover_letter.letter_body
    assert cover_letter.tone == "confident"

    # Interview Prep Agent
    prep_report = await InterviewPrepAgent.run(resume, jd)
    assert len(prep_report.questions) > 0
    assert len(prep_report.questions_for_candidate_to_ask) > 0

    # Research Agent
    research_report = await CompanyResearchAgent.run(jd)
    assert research_report.company_name != ""
    assert len(research_report.values_and_culture) > 0
