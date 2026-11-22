"""Company Research Agent: Synthesizes company insights, culture, and interview conversation hooks."""
import logging
from app.core.llm_client import llm_client
from app.core.schemas import ParsedJD, CompanyResearchReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Senior Strategic Equity Analyst and Corporate Intelligence Specialist.
Your mission is to provide an incisive briefing on a hiring company based on its job posting and public profile.

GUIDELINES:
1. Synthesize company mission, market positioning, and tech stack culture.
2. Outline current industry priorities and technical challenges the company likely faces.
3. Formulate strategic talking points and conversation hooks that help a candidate stand out as an industry insider.
"""


class CompanyResearchAgent:
    """Researches company background and extracts strategic interview conversation hooks."""

    @classmethod
    async def run(cls, jd: ParsedJD, raw_context: str = "") -> CompanyResearchReport:
        logger.info(f"Executing Company Research Agent for {jd.company_name}...")
        prompt = (
            f"Company Name: {jd.company_name}\n"
            f"Role Title: {jd.role_title}\n"
            f"Role Summary: {jd.role_summary}\n"
            f"Raw Context/Posting Snippet:\n{raw_context[:1500]}\n\n"
            "Produce an executive company research report detailing company mission, industry domain, "
            "cultural values, strategic technical priorities, and compelling interview conversation hooks."
        )

        return await llm_client.generate_structured(
            prompt=prompt,
            response_model=CompanyResearchReport,
            system_prompt=SYSTEM_PROMPT,
        )
