"""Cover Letter Agent: Generates tailored, high-converting cover letters with configurable tone."""
import logging
import json
from typing import Literal
from app.core.llm_client import llm_client
from app.core.schemas import ParsedResume, ParsedJD, CoverLetterResponse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Senior Executive Career Strategist and Professional Copywriter.
Your task is to write a compelling, tailored, non-generic cover letter that connects a candidate's verified achievements
directly to a company's highest priority problems.

GUIDELINES:
1. Avoid generic clichés like "I am writing to express my interest in... and I am a hard worker".
2. Open with a compelling narrative hook linking candidate's proven outcomes to the company's core mission.
3. Feature 2-3 specific quantifiable achievements drawn directly from the candidate's resume.
4. Strictly observe the requested tone:
   - "formal": Traditional, polished, respectful, executive.
   - "confident": Bold, results-driven, visionary, authoritative.
   - "friendly": Warm, collaborative, personable, culturally enthusiastic.
"""


class CoverLetterAgent:
    """Generates customized, compelling cover letters tailored to role and tone."""

    @classmethod
    async def run(
        cls,
        resume: ParsedResume,
        jd: ParsedJD,
        tone: Literal["formal", "confident", "friendly"] = "confident",
    ) -> CoverLetterResponse:
        logger.info(f"Executing Cover Letter Agent (Tone: {tone})...")
        candidate_highlights = []
        for exp in resume.experience[:2]:
            candidate_highlights.extend(exp.bullets[:2])

        prompt = (
            f"Candidate Name: {resume.contact.name}\n"
            f"Candidate Professional Summary: {resume.summary}\n"
            f"Selected Verified Accomplishments:\n{json.dumps(candidate_highlights, indent=2)}\n\n"
            f"Target Company: {jd.company_name}\n"
            f"Target Position: {jd.role_title}\n"
            f"Target Responsibilities & Requirements:\n{json.dumps(jd.core_responsibilities, indent=2)}\n\n"
            f"Requested Tone: {tone.upper()}\n\n"
            "Craft an executive-level cover letter that positions the candidate as the ideal solution to the company's needs. "
            "Highlight core value propositions and write an assertive, courteous call to action."
        )

        return await llm_client.generate_structured(
            prompt=prompt,
            response_model=CoverLetterResponse,
            system_prompt=SYSTEM_PROMPT,
        )
