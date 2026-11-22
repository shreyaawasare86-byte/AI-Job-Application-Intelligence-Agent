"""Tests for embedding similarity and hybrid scoring engine."""
import pytest
from app.core.schemas import ParsedResume, ParsedJD, CandidateContact, ExperienceItem, EducationItem
from app.services.embeddings_service import embedding_service
from app.services.scoring_service import scoring_engine


def test_embedding_service_exact_similarity():
    sim = embedding_service.compute_similarity("Python", "Python")
    assert sim == 1.0


def test_embedding_service_unrelated_similarity():
    sim = embedding_service.compute_similarity("PyTorch deep learning", "Pastry baking croissant")
    assert sim < 0.3


def test_embedding_service_skill_matching():
    candidate_skills = ["Python", "PyTorch", "Docker", "PostgreSQL", "FastAPI"]
    required_skills = ["Python", "PyTorch", "Kubernetes", "AWS"]

    matched, partial, missing, ratio = embedding_service.match_skills(candidate_skills, required_skills)
    assert "Python" in matched
    assert "PyTorch" in matched
    assert "Kubernetes" in missing or "Kubernetes" in partial
    assert 0.0 <= ratio <= 1.0


def test_scoring_engine_high_match():
    resume = ParsedResume(
        contact=CandidateContact(name="Alex Rivera"),
        summary="Senior ML Engineer with 6+ years experience in LLMs",
        technical_skills=["Python", "PyTorch", "Docker", "AWS", "FastAPI", "Transformers", "RAG"],
        soft_skills=["Communication", "Leadership"],
        tools_and_frameworks=["Git", "Qdrant", "MLflow"],
        experience=[
            ExperienceItem(
                role_title="Senior ML Engineer",
                company="NeuralScale AI",
                start_date="2022",
                end_date="Present",
                bullets=[
                    "Architected enterprise RAG pipeline using Qdrant vector store.",
                    "Fine-tuned LLMs with PyTorch and deployed FastAPI inference services on AWS.",
                ],
                technologies_used=["PyTorch", "Qdrant", "FastAPI", "AWS"],
            )
        ],
        education=[EducationItem(degree="M.S. in Computer Science", institution="Stanford University")],
        total_years_experience=6.0,
    )

    jd = ParsedJD(
        role_title="Senior Machine Learning Engineer",
        company_name="Synthetix AI",
        min_years_experience=5.0,
        required_skills=["Python", "PyTorch", "RAG", "AWS"],
        preferred_skills=["Docker", "FastAPI"],
        core_responsibilities=["Build production RAG pipelines", "Deploy scalable models on AWS"],
        education_requirements=["M.S. or B.S. in Computer Science"],
        ats_keywords=["PyTorch", "RAG", "AWS", "FastAPI", "Qdrant"],
    )

    resume_text = "Alex Rivera Senior ML Engineer Python PyTorch RAG AWS FastAPI Qdrant Stanford M.S. in Computer Science"
    jd_text = "Senior Machine Learning Engineer Synthetix AI Python PyTorch RAG AWS FastAPI"

    breakdown = scoring_engine.evaluate(resume, jd, resume_text, jd_text)
    assert breakdown.overall_score >= 80.0
    assert breakdown.match_tier in ("Strong Match", "Good Match")
    assert len(breakdown.strengths) > 0
    assert breakdown.skills_score >= 80.0
    assert breakdown.experience_score >= 80.0


def test_scoring_engine_low_match():
    # Junior frontend candidate vs Senior ML role
    resume = ParsedResume(
        contact=CandidateContact(name="Junior Dev"),
        summary="Junior web developer with HTML/CSS",
        technical_skills=["HTML", "CSS", "WordPress"],
        experience=[],
        education=[EducationItem(degree="B.A. in History", institution="Local College")],
        total_years_experience=0.5,
    )

    jd = ParsedJD(
        role_title="Principal Machine Learning Architect",
        company_name="AI Labs",
        min_years_experience=8.0,
        required_skills=["PyTorch", "Distributed Training", "CUDA", "C++", "Triton"],
        core_responsibilities=["Lead multi-node distributed training clusters"],
        education_requirements=["Ph.D. in Computer Science or Physics"],
        ats_keywords=["CUDA", "PyTorch", "Distributed Systems", "Triton"],
    )

    resume_text = "Junior Dev HTML CSS WordPress Local College"
    jd_text = "Principal ML Architect CUDA PyTorch Distributed Systems"

    breakdown = scoring_engine.evaluate(resume, jd, resume_text, jd_text)
    assert breakdown.overall_score < 50.0
    assert breakdown.match_tier == "Weak Match"
    assert len(breakdown.gaps_summary) > 0
