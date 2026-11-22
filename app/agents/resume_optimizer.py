"""Resume Optimizer Agent: Rewrites resume bullets to STAR/XYZ format with ATS keywords and honesty guardrails."""
import logging
import json
from app.core.llm_client import llm_client
from app.core.guardrails import guardrail_engine
from app.core.schemas import ParsedResume, ParsedJD, ResumeOptimizationReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Elite Executive Resume Writer and Career Coach.
Your task is to transform existing resume bullet points into high-impact STAR (Situation, Task, Action, Result)
or Google XYZ ("Accomplished [X] as measured by [Y], by doing [Z]") format, integrating target ATS keywords.

STRICT TRUTHFULNESS & ANTI-HALLUCINATION RULES:
1. NEVER invent employers, job titles, university degrees, or unperformed projects.
2. Maintain exact factual fidelity to what the candidate actually accomplished.
3. If you suggest a quantitative metric to make the bullet punchier (e.g. latency improvement, percentage increase, cost reduction),
   you MUST set requires_confirmation=True and explicitly ask the candidate to confirm the exact metric.
4. Integrate relevant ATS keywords from the JD where they truthfully apply to the work done.
"""


class ResumeOptimizerAgent:
    """Optimizes resume accomplishment bullets for ATS match and recruiter impact."""

    @classmethod
    async def run(cls, resume: ParsedResume, jd: ParsedJD) -> ResumeOptimizationReport:
        logger.info("Executing Resume Optimizer Agent...")
        bullets_to_rewrite = []
        for exp in resume.experience[:3]:  # Top 3 most recent experiences
            for b in exp.bullets[:3]:
                bullets_to_rewrite.append(b)

        prompt = (
            f"Target Role: {jd.role_title} at {jd.company_name}\n"
            f"Target ATS Keywords: {json.dumps(jd.ats_keywords)}\n"
            f"Original Resume Bullets to Enhance:\n{json.dumps(bullets_to_rewrite, indent=2)}\n\n"
            "Rewrite each bullet point using STAR or Google XYZ impact formula while seamlessly weaving in target ATS keywords. "
            "Flag any metrics that need user confirmation. Provide an ATS keyword density report and overall critique."
        )

        report = await llm_client.generate_structured(
            prompt=prompt,
            response_model=ResumeOptimizationReport,
            system_prompt=SYSTEM_PROMPT,
        )

        # Apply secondary guardrail verification audit
        verification_result = guardrail_engine.audit_optimization_report(resume, report.bullet_rewrites)
        if verification_result.hallucinated_metrics:
            for warning in verification_result.verification_notes.split(". "):
                if warning and warning not in report.ungrounded_claim_warnings:
                    report.ungrounded_claim_warnings.append(warning)

        return report
