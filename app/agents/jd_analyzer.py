"""JD Analyzer Agent: Extracts required skills, preferred skills, responsibilities, and ATS keywords."""
import logging
from app.core.llm_client import llm_client
from app.core.schemas import ParsedJD

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Principal Talent Acquisition Architect and ATS Optimization Specialist.
Your task is to analyze a job description and extract its core requirements, qualifications, and domain keywords.
RULES:
1. Differentiate strictly between non-negotiable required_skills and nice-to-have preferred_skills.
2. Extract critical ATS keywords that an ATS scanner would filter on (e.g. tools, libraries, protocols).
3. Identify the seniority level and minimum years of experience expected.
"""


class JDAnalyzerAgent:
    """Analyzes job description text into structured requirements."""

    @classmethod
    async def run(cls, jd_text: str) -> ParsedJD:
        logger.info("Executing JD Analyzer Agent...")
        prompt = (
            "Analyze the following job description and extract the role title, company name, "
            "seniority level, minimum years of experience required, required skills, preferred skills, "
            "core responsibilities, education requirements, ATS keywords, and role summary:\n\n"
            f"--- JOB DESCRIPTION ---\n{jd_text}\n--- END JOB DESCRIPTION ---"
        )
        return await llm_client.generate_structured(
            prompt=prompt,
            response_model=ParsedJD,
            system_prompt=SYSTEM_PROMPT,
        )
