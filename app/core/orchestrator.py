"""Multi-Agent Orchestrator: Directed Acyclic Graph (DAG) coordinating parallel execution and state."""
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Literal

from app.core.schemas import PipelineExecutionState
from app.agents.resume_parser import ResumeParserAgent
from app.agents.jd_analyzer import JDAnalyzerAgent
from app.agents.matching_agent import MatchingAgent
from app.agents.skill_gap_agent import SkillGapAgent
from app.agents.resume_optimizer import ResumeOptimizerAgent
from app.agents.cover_letter_agent import CoverLetterAgent
from app.agents.interview_prep_agent import InterviewPrepAgent
from app.agents.research_agent import CompanyResearchAgent
from app.core.guardrails import guardrail_engine

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Coordinates execution graph for the 8 specialized AI agents with parallel fan-out."""

    @classmethod
    async def run(
        cls,
        resume_text: str,
        jd_text: str,
        tone: Literal["formal", "confident", "friendly"] = "confident",
    ) -> PipelineExecutionState:
        """Execute full multi-agent pipeline and aggregate structured intelligence state."""
        start_time = time.time()
        run_id = str(uuid.uuid4())
        logger.info(f"Starting Multi-Agent Pipeline Run [ID: {run_id}]...")

        # --- STAGE 1: PARALLEL INGESTION & EXTRACTION ---
        # Resume Parser and JD Analyzer run simultaneously
        parsed_resume, parsed_jd = await asyncio.gather(
            ResumeParserAgent.run(resume_text),
            JDAnalyzerAgent.run(jd_text),
        )
        logger.info("Stage 1 completed: Resume and Job Description parsed.")

        # --- STAGE 2: HYBRID MATCHING ENGINE ---
        # Computes semantic similarity, ATS keyword density, and seniority fit
        match_breakdown = await MatchingAgent.run(
            resume=parsed_resume,
            jd=parsed_jd,
            resume_text=resume_text,
            jd_text=jd_text,
        )
        logger.info(f"Stage 2 completed: Match score computed ({match_breakdown.overall_score}/100).")

        # --- STAGE 3: PARALLEL AGENT FAN-OUT ---
        # Skill Gap, Resume Optimizer, Cover Letter, Interview Prep, and Company Research run in parallel
        skill_gap_task = SkillGapAgent.run(parsed_resume, parsed_jd)
        resume_opt_task = ResumeOptimizerAgent.run(parsed_resume, parsed_jd)
        cover_letter_task = CoverLetterAgent.run(parsed_resume, parsed_jd, tone=tone)
        interview_prep_task = InterviewPrepAgent.run(parsed_resume, parsed_jd)
        research_task = CompanyResearchAgent.run(parsed_jd, raw_context=jd_text)

        (
            skill_gap_report,
            resume_optimization,
            cover_letter,
            interview_prep,
            company_research,
        ) = await asyncio.gather(
            skill_gap_task,
            resume_opt_task,
            cover_letter_task,
            interview_prep_task,
            research_task,
        )
        logger.info("Stage 3 completed: Downstream intelligence agents synthesized outputs.")

        # --- STAGE 4: GUARDRAIL & VERIFICATION AUDIT ---
        guardrail_result = guardrail_engine.audit_optimization_report(
            resume=parsed_resume,
            rewritten_bullets=resume_optimization.bullet_rewrites,
        )
        logger.info(f"Stage 4 completed: Guardrail audit passed (Clean: {guardrail_result.is_valid}).")

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"Pipeline Run [ID: {run_id}] completed in {elapsed}s.")

        # --- STAGE 5: CONSOLIDATE PIPELINE STATE ---
        return PipelineExecutionState(
            run_id=run_id,
            run_timestamp=datetime.now().isoformat(),
            resume_text=resume_text,
            jd_text=jd_text,
            parsed_resume=parsed_resume,
            parsed_jd=parsed_jd,
            match_breakdown=match_breakdown,
            skill_gap_report=skill_gap_report,
            resume_optimization=resume_optimization,
            cover_letter=cover_letter,
            interview_prep=interview_prep,
            company_research=company_research,
            guardrail_result=guardrail_result,
            execution_time_seconds=elapsed,
        )


pipeline_orchestrator = MultiAgentOrchestrator()
