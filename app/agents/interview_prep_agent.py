"""Interview Prep Agent: Generates role-specific questions and resume-grounded answer blueprints."""
import logging
import json
from app.core.llm_client import llm_client
from app.core.schemas import ParsedResume, ParsedJD, InterviewPrepReport

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Principal Technical Interviewer and Hiring Committee Lead.
Your mission is to simulate an intense, realistic interview loop for the candidate based on the target Job Description.

GUIDELINES:
1. Generate high-probability interview questions across technical, behavioral, and system design categories.
2. For each question, construct a concrete answer blueprint grounded specifically in the candidate's actual resume experiences.
3. Include rationale explaining why the interviewer asks this question.
4. Formulate 3 thoughtful, high-signal questions for the candidate to ask the interview panel.
"""


class InterviewPrepAgent:
    """Prepares candidates for technical and behavioral interviews with tailored answer frameworks."""

    @classmethod
    async def run(cls, resume: ParsedResume, jd: ParsedJD) -> InterviewPrepReport:
        logger.info("Executing Interview Prep Agent...")
        candidate_summary = {
            "name": resume.contact.name,
            "years_exp": resume.total_years_experience,
            "skills": resume.technical_skills[:10],
            "recent_projects": [p.title for p in resume.projects[:3]],
            "recent_roles": [f"{e.role_title} at {e.company}" for e in resume.experience[:2]],
        }

        prompt = (
            f"Candidate Profile: {json.dumps(candidate_summary, indent=2)}\n\n"
            f"Target Role: {jd.role_title} at {jd.company_name}\n"
            f"Core Responsibilities: {json.dumps(jd.core_responsibilities, indent=2)}\n"
            f"Required Skills: {json.dumps(jd.required_skills, indent=2)}\n\n"
            "Generate curated technical, system design, and behavioral interview questions tailored to this role. "
            "For each question, map an answer outline directly to the candidate's verified experience using the STAR method."
        )

        return await llm_client.generate_structured(
            prompt=prompt,
            response_model=InterviewPrepReport,
            system_prompt=SYSTEM_PROMPT,
        )
