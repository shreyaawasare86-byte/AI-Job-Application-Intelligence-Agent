"""Tests for anti-hallucination guardrail engine."""
from app.core.schemas import ParsedResume, CandidateContact, ExperienceItem, BulletRewrite
from app.core.guardrails import guardrail_engine


def test_metric_extraction():
    text = "Improved throughput from 12 req/s to 55 req/s while saving $18K/month (a 358% increase)."
    metrics = guardrail_engine.extract_metrics(text)
    assert any("358%" in m for m in metrics)
    assert any("$18K" in m for m in metrics)
    assert any("55 req/s" in m or "req/s" in m for m in metrics)


def test_verify_bullet_with_no_invented_metrics():
    orig = "Architected RAG pipeline with Qdrant reducing hallucination by 38%."
    rewrite = "Engineered enterprise RAG pipeline with Qdrant vector database, decreasing hallucination by 38%."
    needs_confirm, new_metrics, score = guardrail_engine.verify_bullet(orig, rewrite)
    assert needs_confirm is False
    assert len(new_metrics) == 0
    assert score == 1.0


def test_verify_bullet_with_invented_metric():
    orig = "Built RAG pipeline with Qdrant."
    rewrite = "Engineered enterprise RAG pipeline with Qdrant, cutting latency by 45% and saving $50,000 annually."
    needs_confirm, new_metrics, score = guardrail_engine.verify_bullet(orig, rewrite)
    assert needs_confirm is True
    assert any("45%" in m for m in new_metrics)
    assert any("$50,000" in m for m in new_metrics)
    assert score < 1.0


def test_audit_optimization_report():
    resume = ParsedResume(
        contact=CandidateContact(name="Alex Rivera"),
        experience=[
            ExperienceItem(
                role_title="Senior ML Engineer",
                company="NeuralScale AI",
                bullets=["Built RAG pipeline querying 15M+ documents using Qdrant."],
            )
        ],
    )
    rewrites = [
        BulletRewrite(
            original_bullet="Built RAG pipeline querying 15M+ documents using Qdrant.",
            rewritten_bullet="Architected RAG pipeline querying 15M+ documents using Qdrant, decreasing search latency by 35%.",
            format_used="STAR",
        )
    ]
    audit_res = guardrail_engine.audit_optimization_report(resume, rewrites)
    assert audit_res.is_valid is True
    assert len(audit_res.hallucinated_metrics) > 0
    assert rewrites[0].requires_confirmation is True
    assert "Verify" in (rewrites[0].confirmation_note or "")
