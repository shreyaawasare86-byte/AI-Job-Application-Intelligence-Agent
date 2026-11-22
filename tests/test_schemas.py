"""Tests for Pydantic v2 schemas and models."""
import pytest
from app.core.schemas import (
    CandidateContact,
    ExperienceItem,
    EducationItem,
    ProjectItem,
    ParsedResume,
    ParsedJD,
    MatchScoreBreakdown,
    SkillGapItem,
    SkillGapReport,
    BulletRewrite,
    ResumeOptimizationReport,
    CoverLetterResponse,
    InterviewQuestion,
    InterviewPrepReport,
    CompanyResearchReport,
    GuardrailVerificationResult,
    ApplicationRecord,
)


def test_parsed_resume_schema():
    contact = CandidateContact(
        name="Alex Rivera",
        email="alex@example.com",
        phone="555-1234",
        location="San Francisco, CA",
        linkedin="https://linkedin.com/in/alex",
        github="https://github.com/alex",
    )
    exp = ExperienceItem(
        role_title="Senior ML Engineer",
        company="NeuralScale AI",
        start_date="2022",
        end_date="Present",
        bullets=["Architected RAG system serving 15M docs."],
        technologies_used=["Python", "PyTorch", "Qdrant"],
    )
    edu = EducationItem(
        degree="M.S. in Computer Science",
        institution="Stanford University",
        graduation_year="2019",
    )
    proj = ProjectItem(
        title="OpenRAG Benchmark",
        description="Benchmarking harness for evaluation",
        technologies=["Python", "PyTorch"],
    )
    resume = ParsedResume(
        contact=contact,
        summary="Senior ML Engineer with 6+ years experience",
        technical_skills=["Python", "PyTorch", "Docker"],
        soft_skills=["Leadership", "Communication"],
        tools_and_frameworks=["Git", "MLflow"],
        experience=[exp],
        education=[edu],
        projects=[proj],
        certifications=["AWS ML Specialty"],
        total_years_experience=6.0,
    )
    assert resume.contact.name == "Alex Rivera"
    assert len(resume.experience) == 1
    assert resume.total_years_experience == 6.0
    assert "Python" in resume.technical_skills


def test_parsed_jd_schema():
    jd = ParsedJD(
        role_title="Senior Machine Learning Engineer",
        company_name="Synthetix AI",
        seniority_level="Senior",
        min_years_experience=5.0,
        required_skills=["Python", "PyTorch", "RAG"],
        preferred_skills=["vLLM", "Ray"],
        core_responsibilities=["Design and build production LLM systems"],
        education_requirements=["B.S. or M.S. in Computer Science"],
        ats_keywords=["RAG", "PyTorch", "LLM", "Vector Search"],
        role_summary="Building next-gen AI knowledge agents",
    )
    assert jd.role_title == "Senior Machine Learning Engineer"
    assert jd.min_years_experience == 5.0
    assert len(jd.required_skills) == 3


def test_match_score_breakdown_validation():
    breakdown = MatchScoreBreakdown(
        overall_score=88.5,
        skills_score=92.0,
        experience_score=85.0,
        education_score=90.0,
        keyword_score=87.0,
        match_tier="Strong Match",
        reasoning="Exceptional alignment with LLM and RAG requirements.",
        strengths=["PyTorch expertise", "Production RAG experience"],
        gaps_summary=["Ray distributed training"],
    )
    assert 0 <= breakdown.overall_score <= 100
    assert breakdown.match_tier == "Strong Match"

    with pytest.raises(Exception):
        # Invalid score exceeding 100
        MatchScoreBreakdown(
            overall_score=110.0,
            skills_score=90.0,
            experience_score=80.0,
            education_score=70.0,
            keyword_score=80.0,
            reasoning="Invalid",
        )


def test_bullet_rewrite_schema():
    rewrite = BulletRewrite(
        original_bullet="Built RAG system with Qdrant.",
        rewritten_bullet="Architected high-throughput RAG pipeline querying 15M+ documents using Qdrant vector database, reducing query latency by 35%.",
        format_used="STAR",
        ats_keywords_incorporated=["RAG", "Qdrant", "Vector Database"],
        requires_confirmation=True,
        confirmation_note="Verify 35% latency reduction metric.",
        groundedness_score=0.95,
    )
    assert rewrite.requires_confirmation is True
    assert "Qdrant" in rewrite.ats_keywords_incorporated


def test_application_record_schema():
    app = ApplicationRecord(
        company="Synthetix AI",
        role="Senior ML Engineer",
        match_score=88.5,
        status="Applied",
        date_applied="2026-11-21",
        notes="Applied via website referral.",
    )
    assert app.status == "Applied"
    assert app.match_score == 88.5
