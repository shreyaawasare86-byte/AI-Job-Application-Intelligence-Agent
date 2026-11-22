"""Evaluation Runner: Benchmarks multi-agent pipeline consistency across sample resume/JD pairs."""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set WindowsSelectorEventLoopPolicy for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.config import settings
from app.core.orchestrator import pipeline_orchestrator
from app.services.parser_service import DocumentParserService

logger = logging.getLogger(__name__)

SAMPLE_DIR = settings.SAMPLE_DIR

EVAL_BENCHMARKS = [
    {
        "id": "eval-1",
        "name": "Senior ML Engineer (Domain Match)",
        "resume": "resume_1_ml_engineer.txt",
        "jd": "jd_1_senior_ml_engineer.txt",
        "expected_tier": ["Strong Match", "Good Match"],
        "min_expected_score": 75.0,
    },
    {
        "id": "eval-2",
        "name": "Full Stack React/Python (Domain Match)",
        "resume": "resume_2_fullstack_dev.txt",
        "jd": "jd_2_fullstack_react_python.txt",
        "expected_tier": ["Strong Match", "Good Match"],
        "min_expected_score": 70.0,
    },
    {
        "id": "eval-3",
        "name": "Lead Data Analyst (Domain Match)",
        "resume": "resume_3_data_analyst.txt",
        "jd": "jd_3_lead_data_analyst.txt",
        "expected_tier": ["Strong Match", "Good Match"],
        "min_expected_score": 70.0,
    },
    {
        "id": "eval-4",
        "name": "Cloud DevOps & SRE (Domain Match)",
        "resume": "resume_4_devops_cloud.txt",
        "jd": "jd_4_site_reliability_engineer.txt",
        "expected_tier": ["Strong Match", "Good Match"],
        "min_expected_score": 75.0,
    },
    {
        "id": "eval-5",
        "name": "Junior Backend Developer (Domain Match)",
        "resume": "resume_5_junior_developer.txt",
        "jd": "jd_5_junior_backend_dev.txt",
        "expected_tier": ["Strong Match", "Good Match"],
        "min_expected_score": 70.0,
    },
    {
        "id": "eval-6-negative",
        "name": "Negative Control: Junior Web Dev vs Senior ML Role",
        "resume": "resume_5_junior_developer.txt",
        "jd": "jd_1_senior_ml_engineer.txt",
        "expected_tier": ["Weak Match", "Moderate Match"],
        "max_expected_score": 55.0,
    },
]


async def run_evaluation_suite() -> Dict[str, Any]:
    """Execute evaluation harness across all benchmarks."""
    print("=" * 80)
    print("🚀 STARTING AI JOB APPLICATION INTELLIGENCE AGENT EVALUATION HARNESS")
    print("=" * 80)
    print(f"Environment: {settings.ENVIRONMENT} | LLM: {settings.GEMINI_MODEL} | Provider: {settings.LLM_PROVIDER}")
    print("-" * 80)

    results: List[Dict[str, Any]] = []
    total_latency = 0.0

    for bench in EVAL_BENCHMARKS:
        b_id = bench["id"]
        b_name = bench["name"]
        r_path = SAMPLE_DIR / bench["resume"]
        j_path = SAMPLE_DIR / bench["jd"]

        if not r_path.exists() or not j_path.exists():
            print(f"❌ Skipping {b_id}: missing test file.")
            continue

        resume_text = DocumentParserService.extract_from_path(r_path)
        jd_text = DocumentParserService.extract_from_path(j_path)

        t0 = time.time()
        state = await pipeline_orchestrator.run(
            resume_text=resume_text,
            jd_text=jd_text,
            tone="confident",
        )
        latency = round(time.time() - t0, 2)
        total_latency += latency

        score = state.match_breakdown.overall_score
        tier = state.match_breakdown.match_tier
        guardrail_clean = state.guardrail_result.is_valid and len(state.guardrail_result.hallucinated_metrics) == 0

        # Assert criteria
        passed_score_check = True
        if "min_expected_score" in bench:
            passed_score_check = score >= bench["min_expected_score"]
        elif "max_expected_score" in bench:
            passed_score_check = score <= bench["max_expected_score"]

        status_icon = "✅" if passed_score_check else "⚠️"

        print(
            f"{status_icon} [{b_id}] {b_name:<45} | Score: {score:>5.1f}/100 | Tier: {tier:<14} | "
            f"Latency: {latency:>4.2f}s | Guardrail: {'Clean' if guardrail_clean else 'Flagged Metrics'}"
        )

        results.append({
            "id": b_id,
            "name": b_name,
            "match_score": score,
            "skills_score": state.match_breakdown.skills_score,
            "experience_score": state.match_breakdown.experience_score,
            "keyword_score": state.match_breakdown.keyword_score,
            "education_score": state.match_breakdown.education_score,
            "tier": tier,
            "latency_seconds": latency,
            "guardrail_clean": guardrail_clean,
            "rewritten_bullets_count": len(state.resume_optimization.bullet_rewrites),
            "interview_questions_count": len(state.interview_prep.questions),
            "cover_letter_words": len(state.cover_letter.letter_body.split()),
            "passed_expected_criteria": passed_score_check,
        })

    avg_score = round(sum(r["match_score"] for r in results) / len(results), 1) if results else 0
    avg_latency = round(total_latency / len(results), 2) if results else 0
    pass_rate = round((sum(1 for r in results if r["passed_expected_criteria"]) / len(results)) * 100, 1)

    print("-" * 80)
    print(f"📊 SUMMARY: Total Benchmarks: {len(results)} | Pass Rate: {pass_rate}% | Avg Latency: {avg_latency}s")
    print("=" * 80)

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_evaluations": len(results),
        "pass_rate_percentage": pass_rate,
        "average_score": avg_score,
        "average_latency_seconds": avg_latency,
        "benchmark_results": results,
    }

    # Save to JSON artifact
    out_path = settings.DATA_DIR / "evaluation_report.json"
    out_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"📁 Detailed evaluation report saved to: {out_path}")

    return summary_data


if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
