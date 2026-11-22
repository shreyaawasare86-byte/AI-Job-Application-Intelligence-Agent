"""Resume Parser Agent: Extracts structured candidate profile into ParsedResume schema."""
import logging
from app.core.llm_client import llm_client
from app.core.schemas import ParsedResume

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert Executive Resume Parser and ATS Specialist.
Your sole mission is to extract structured, accurate facts from a candidate's resume into the specified JSON schema.
RULES:
1. ONLY extract information directly stated or clearly implied by the text.
2. Do NOT hallucinate skills, metrics, degrees, or employers.
3. Classify skills cleanly into technical_skills, tools_and_frameworks, and soft_skills.
4. Estimate total_years_experience based on employment history dates.
"""


class ResumeParserAgent:
    """Extracts structured resume data from unstructured text."""

    @classmethod
    async def run(cls, resume_text: str) -> ParsedResume:
        logger.info("Executing Resume Parser Agent...")
        prompt = (
            "Extract all candidate contact information, professional summary, categorized skills, "
            "work experience history (roles, companies, dates, bullets, tools), education credentials, "
            "projects, and certifications from the following resume text:\n\n"
            f"--- RESUME TEXT ---\n{resume_text}\n--- END RESUME TEXT ---"
        )
        return await llm_client.generate_structured(
            prompt=prompt,
            response_model=ParsedResume,
            system_prompt=SYSTEM_PROMPT,
        )
