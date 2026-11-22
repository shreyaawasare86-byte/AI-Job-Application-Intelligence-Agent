"""FastAPI REST API for AI Job Application Intelligence Agent."""
import logging
from typing import Optional, List, Dict, Any, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.schemas import PipelineExecutionState, ApplicationRecord
from app.core.orchestrator import pipeline_orchestrator
from app.services.parser_service import DocumentParserService
from app.services.tracker_service import tracker_service

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Job Application Intelligence Agent API",
    description="Multi-agent orchestration backend for resume parsing, scoring, ATS optimization, and application tracking.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REQUEST & RESPONSE SCHEMAS ---

class AnalyzeTextRequest(BaseModel):
    resume_text: str = Field(description="Raw text of candidate resume")
    jd_text: str = Field(description="Raw text of target job description")
    tone: Literal["formal", "confident", "friendly"] = Field(default="confident")


class UpdateApplicationRequest(BaseModel):
    status: Optional[Literal["Wishlist", "Applied", "Screening", "Interviewing", "Offer", "Rejected"]] = None
    notes: Optional[str] = None
    match_score: Optional[float] = None


# --- HEALTH & STATUS ENDPOINTS ---

@app.get("/api/health", tags=["System"])
async def health_check():
    """System health check and runtime configuration status."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "llm_provider": settings.LLM_PROVIDER,
        "model": settings.GEMINI_MODEL,
        "has_gemini_key": settings.has_valid_gemini_key,
        "cache_enabled": settings.ENABLE_LLM_CACHE,
    }


# --- PIPELINE ANALYSIS ENDPOINTS ---

@app.post("/api/analyze/text", response_model=PipelineExecutionState, tags=["Intelligence Pipeline"])
async def analyze_text(request: AnalyzeTextRequest):
    """Run full multi-agent intelligence pipeline on raw resume and JD text."""
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")
    if not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    try:
        return await pipeline_orchestrator.run(
            resume_text=request.resume_text,
            jd_text=request.jd_text,
            tone=request.tone,
        )
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/api/analyze/file", response_model=PipelineExecutionState, tags=["Intelligence Pipeline"])
async def analyze_file(
    resume_file: UploadFile = File(..., description="PDF, DOCX, or TXT resume document"),
    jd_text: Optional[str] = Form(None, description="Pasted Job Description text"),
    jd_url: Optional[str] = Form(None, description="URL of target job description"),
    tone: Literal["formal", "confident", "friendly"] = Form("confident"),
):
    """Upload resume document (PDF/DOCX/TXT) and analyze against JD text or URL."""
    # 1. Parse Resume Document
    try:
        content_bytes = await resume_file.read()
        extracted_resume_text = DocumentParserService.extract_from_bytes(
            file_bytes=content_bytes,
            filename=resume_file.filename or "resume.txt",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing resume file: {str(e)}")

    if not extracted_resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded resume.")

    # 2. Extract or Fetch Job Description
    target_jd_text = jd_text or ""
    if not target_jd_text.strip() and jd_url:
        try:
            target_jd_text = await DocumentParserService.extract_from_url(jd_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch job description from URL: {str(e)}")

    if not target_jd_text.strip():
        raise HTTPException(status_code=400, detail="Please provide either JD text or a valid JD URL.")

    # 3. Execute Pipeline
    try:
        result = await pipeline_orchestrator.run(
            resume_text=extracted_resume_text,
            jd_text=target_jd_text,
            tone=tone,
        )
        return result
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# --- APPLICATION TRACKER ENDPOINTS ---

@app.get("/api/applications", response_model=List[ApplicationRecord], tags=["Application Tracker"])
async def list_applications(status: Optional[str] = Query(None, description="Filter by application status")):
    """List all tracked job applications with optional status filter."""
    return tracker_service.list_applications(status_filter=status)


@app.post("/api/applications", response_model=ApplicationRecord, status_code=status.HTTP_201_CREATED, tags=["Application Tracker"])
async def create_application(record: ApplicationRecord):
    """Add a new job application record to SQLite storage."""
    try:
        return tracker_service.add_application(record)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create application: {str(e)}")


@app.get("/api/applications/{app_id}", response_model=ApplicationRecord, tags=["Application Tracker"])
async def get_application(app_id: int):
    """Get single tracked application by ID."""
    record = tracker_service.get_application(app_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Application with ID {app_id} not found.")
    return record


@app.patch("/api/applications/{app_id}", response_model=ApplicationRecord, tags=["Application Tracker"])
async def update_application(app_id: int, request: UpdateApplicationRequest):
    """Update status, notes, or match score of an application."""
    updated = tracker_service.update_application(
        app_id=app_id,
        status=request.status,
        notes=request.notes,
        match_score=request.match_score,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Application with ID {app_id} not found.")
    return updated


@app.delete("/api/applications/{app_id}", tags=["Application Tracker"])
async def delete_application(app_id: int):
    """Delete an application record by ID."""
    success = tracker_service.delete_application(app_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Application with ID {app_id} not found.")
    return {"message": f"Application {app_id} successfully deleted."}


@app.get("/api/applications/stats/summary", tags=["Application Tracker"])
async def get_application_analytics():
    """Retrieve aggregate application metrics and pipeline funnels."""
    return tracker_service.get_analytics_summary()
