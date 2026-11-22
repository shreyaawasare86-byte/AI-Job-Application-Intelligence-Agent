"""Matching Agent: Synthesizes hybrid semantic, keyword, and experience scoring into MatchScoreBreakdown."""
import logging
from app.core.schemas import ParsedResume, ParsedJD, MatchScoreBreakdown
from app.services.scoring_service import scoring_engine

logger = logging.getLogger(__name__)


class MatchingAgent:
    """Computes transparent multi-factor match score and synthesizes actionable reasoning."""

    @classmethod
    async def run(
        cls,
        resume: ParsedResume,
        jd: ParsedJD,
        resume_text: str,
        jd_text: str,
    ) -> MatchScoreBreakdown:
        logger.info("Executing Matching Agent...")
        # Deterministic hybrid scoring engine combines embeddings + keyword overlap + seniority
        breakdown = scoring_engine.evaluate(
            resume=resume,
            jd=jd,
            resume_text=resume_text,
            jd_text=jd_text,
        )
        return breakdown
