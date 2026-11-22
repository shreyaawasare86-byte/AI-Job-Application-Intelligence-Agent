"""Skill Gap Agent: Classifies skills as matched/partial/missing and recommends learning paths and mini-projects."""
import logging
import json
from app.core.llm_client import llm_client
from app.core.schemas import ParsedResume, ParsedJD, SkillGapReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Senior Engineering Career Advisor and Technical Curriculum Designer.
Your objective is to provide a constructive, honest skill-gap analysis comparing a candidate's background to a target role.
RULES:
1. Categorize each JD requirement into matched, partial, or missing.
2. For missing or partial requirements, design practical, high-leverage weekend mini-projects that the candidate could build to prove competence.
3. Recommend specific, high-quality documentation, books, or courses with estimated study hours.
"""


class SkillGapAgent:
    """Diagnoses technical qualification gaps and builds targeted upskilling roadmaps."""

    @classmethod
    async def run(cls, resume: ParsedResume, jd: ParsedJD) -> SkillGapReport:
        logger.info("Executing Skill Gap Agent...")
        candidate_skills = list(set(resume.technical_skills + resume.tools_and_frameworks))
        prompt = (
            f"Candidate Skills: {json.dumps(candidate_skills)}\n"
            f"Candidate Years of Experience: {resume.total_years_experience}\n"
            f"Target Role: {jd.role_title} at {jd.company_name}\n"
            f"Target Required Skills: {json.dumps(jd.required_skills)}\n"
            f"Target Preferred Skills: {json.dumps(jd.preferred_skills)}\n\n"
            "Perform a rigorous skill-gap analysis. Identify matched skills, partial skills, and missing skills. "
            "For significant gaps, provide actionable mini-project blueprints and top learning resources."
        )
        return await llm_client.generate_structured(
            prompt=prompt,
            response_model=SkillGapReport,
            system_prompt=SYSTEM_PROMPT,
        )
