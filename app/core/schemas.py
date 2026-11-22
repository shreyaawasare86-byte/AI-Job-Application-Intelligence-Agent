"""Pydantic v2 schemas for all multi-agent inputs, outputs, and intermediate states."""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# --- RESUME PARSER SCHEMAS ---

class CandidateContact(BaseModel):
    name: str = Field(default="Candidate", description="Full name of candidate")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="City, State / Country")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    github: Optional[str] = Field(default=None, description="GitHub profile URL")


class ExperienceItem(BaseModel):
    role_title: str = Field(description="Job title or role")
    company: str = Field(description="Company or organization name")
    location: Optional[str] = Field(default=None, description="Job location")
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date or 'Present'")
    bullets: List[str] = Field(default_factory=list, description="Original accomplishment bullet points")
    technologies_used: List[str] = Field(default_factory=list, description="Technologies or tools mentioned")


class EducationItem(BaseModel):
    degree: str = Field(description="Degree name (e.g. B.S. Computer Science)")
    institution: str = Field(description="University or college name")
    field_of_study: Optional[str] = Field(default=None, description="Major or area of specialization")
    graduation_year: Optional[str] = Field(default=None, description="Year graduated or expected")
    gpa: Optional[str] = Field(default=None, description="GPA if specified")


class ProjectItem(BaseModel):
    title: str = Field(description="Project name")
    description: str = Field(description="Short summary of project and purpose")
    technologies: List[str] = Field(default_factory=list, description="Technologies / frameworks used")
    url: Optional[str] = Field(default=None, description="Live link or repository URL")


class ParsedResume(BaseModel):
    contact: CandidateContact = Field(default_factory=CandidateContact)
    summary: str = Field(default="", description="Professional summary or headline")
    technical_skills: List[str] = Field(default_factory=list, description="Languages, frameworks, databases, infrastructure")
    soft_skills: List[str] = Field(default_factory=list, description="Communication, leadership, teamwork")
    tools_and_frameworks: List[str] = Field(default_factory=list, description="Developer tools, libraries, platforms")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work history entries")
    education: List[EducationItem] = Field(default_factory=list, description="Education credentials")
    projects: List[ProjectItem] = Field(default_factory=list, description="Notable projects")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    total_years_experience: float = Field(default=0.0, description="Estimated total professional experience in years")


# --- JOB DESCRIPTION SCHEMAS ---

class ParsedJD(BaseModel):
    role_title: str = Field(description="Official job title")
    company_name: str = Field(description="Hiring company name")
    seniority_level: str = Field(default="Mid", description="Seniority level (Entry, Mid, Senior, Lead, Principal)")
    min_years_experience: float = Field(default=0.0, description="Minimum years of experience required")
    required_skills: List[str] = Field(default_factory=list, description="Non-negotiable required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills")
    core_responsibilities: List[str] = Field(default_factory=list, description="Key duties and deliverables")
    education_requirements: List[str] = Field(default_factory=list, description="Degrees or certifications specified")
    ats_keywords: List[str] = Field(default_factory=list, description="Key ATS terms, domain terminology, toolnames")
    role_summary: str = Field(default="", description="High level synopsis of the opportunity")


# --- MATCHING SCHEMAS ---

class MatchScoreBreakdown(BaseModel):
    overall_score: float = Field(ge=0.0, le=100.0, description="Aggregated composite match score 0-100")
    skills_score: float = Field(ge=0.0, le=100.0, description="Technical and soft skills alignment score 0-100")
    experience_score: float = Field(ge=0.0, le=100.0, description="Years of experience and seniority tier match 0-100")
    education_score: float = Field(ge=0.0, le=100.0, description="Education degree and credential match 0-100")
    keyword_score: float = Field(ge=0.0, le=100.0, description="ATS keyword exact & semantic match 0-100")
    match_tier: str = Field(default="Moderate Match", description="Strong Match / Good Match / Moderate Match / Weak Match")
    reasoning: str = Field(description="Transparent natural language explanation of the composite score")
    strengths: List[str] = Field(default_factory=list, description="Top candidate advantages for this role")
    gaps_summary: List[str] = Field(default_factory=list, description="Primary qualification gaps identified")


# --- SKILL GAP SCHEMAS ---

class LearningResource(BaseModel):
    title: str = Field(description="Course, book, or documentation title")
    resource_type: str = Field(description="Course / Documentation / Book / Hands-on Project")
    url_or_platform: str = Field(description="Platform name or URL")
    estimated_hours: int = Field(default=10, description="Estimated time investment in hours")


class SkillGapItem(BaseModel):
    skill_name: str = Field(description="Name of the skill")
    status: Literal["matched", "partial", "missing"] = Field(description="Classification status")
    importance: Literal["required", "preferred"] = Field(description="JD priority")
    candidate_evidence: Optional[str] = Field(default=None, description="What the resume shows or lacks")
    gap_analysis: str = Field(description="Explanation of the gap depth")
    mini_project_idea: Optional[str] = Field(default=None, description="A weekend mini-project to build evidence")
    learning_resources: List[LearningResource] = Field(default_factory=list, description="Targeted learning materials")


class SkillGapReport(BaseModel):
    matched_skills: List[str] = Field(default_factory=list, description="Fully demonstrated skills")
    partial_skills: List[str] = Field(default_factory=list, description="Adjacent or partially evidenced skills")
    missing_skills: List[str] = Field(default_factory=list, description="Unaddressed requirements")
    detailed_gaps: List[SkillGapItem] = Field(default_factory=list, description="Detailed item-by-item breakdown")
    summary: str = Field(default="", description="Executive summary of the skill landscape")


# --- RESUME OPTIMIZER SCHEMAS ---

class BulletRewrite(BaseModel):
    original_bullet: str = Field(description="Exact original bullet from resume")
    rewritten_bullet: str = Field(description="Enhanced STAR/XYZ bullet incorporating keywords without fabricating")
    format_used: str = Field(default="STAR", description="Framework applied: STAR or XYZ")
    ats_keywords_incorporated: List[str] = Field(default_factory=list, description="Target keywords woven in")
    requires_confirmation: bool = Field(default=False, description="True if a metric was generalized and needs user verification")
    confirmation_note: Optional[str] = Field(default=None, description="Guidance on what exact number or fact to verify")
    groundedness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence that facts stay 100% faithful to resume")


class ResumeOptimizationReport(BaseModel):
    bullet_rewrites: List[BulletRewrite] = Field(default_factory=list, description="Rewritten bullet suggestions")
    ats_keyword_density_report: Dict[str, int] = Field(default_factory=dict, description="Keyword appearance counts")
    ungrounded_claim_warnings: List[str] = Field(default_factory=list, description="Warnings if any claims risk exceeding original facts")
    general_critique: str = Field(default="", description="High level advice on structure and presentation")


# --- COVER LETTER SCHEMAS ---

class CoverLetterResponse(BaseModel):
    recipient_role: str = Field(default="Hiring Manager")
    tone: Literal["formal", "confident", "friendly"] = Field(default="confident")
    letter_body: str = Field(description="Full text of the tailored cover letter")
    key_value_props_highlighted: List[str] = Field(default_factory=list, description="Core candidate strengths featured")
    call_to_action: str = Field(default="Looking forward to discussing how my experience aligns with your team's goals.")


# --- INTERVIEW PREP SCHEMAS ---

class InterviewQuestion(BaseModel):
    category: Literal["technical", "behavioral", "situational", "system_design"] = Field(description="Question type")
    question: str = Field(description="Interview question tailored to the JD")
    rationale: str = Field(description="Why the hiring team will ask this based on the JD")
    answer_outline: str = Field(description="STAR-formatted answer outline referencing candidate's actual projects")
    key_talking_points: List[str] = Field(default_factory=list, description="Crucial bullet points to articulate")


class InterviewPrepReport(BaseModel):
    questions: List[InterviewQuestion] = Field(default_factory=list, description="Curated interview questions")
    questions_for_candidate_to_ask: List[str] = Field(default_factory=list, description="Insightful reverse-interview questions")
    behavioral_framework_tips: str = Field(default="", description="Coaching tips for candidate delivery")


# --- COMPANY RESEARCH SCHEMAS ---

class CompanyResearchReport(BaseModel):
    company_name: str = Field(description="Name of company")
    industry: str = Field(description="Industry vertical")
    overview: str = Field(description="Executive overview of company mission and products")
    values_and_culture: List[str] = Field(default_factory=list, description="Key cultural values and working style")
    recent_priorities: List[str] = Field(default_factory=list, description="Known strategic initiatives or technical direction")
    strategic_talking_points_for_interview: List[str] = Field(default_factory=list, description="Conversation starters to stand out")


# --- GUARDRAIL & VERIFICATION SCHEMAS ---

class GuardrailVerificationResult(BaseModel):
    is_valid: bool = Field(default=True, description="True if output adheres to anti-hallucination rules")
    ungrounded_entities: List[str] = Field(default_factory=list, description="Disallowed or ungrounded companies/technologies")
    hallucinated_metrics: List[str] = Field(default_factory=list, description="Fabricated quantitative metrics detected")
    verification_notes: str = Field(default="All generated content verified against original resume.", description="Audit notes")


# --- AGGREGATED PIPELINE STATE ---

class PipelineExecutionState(BaseModel):
    run_id: str = Field(description="Unique UUID for this pipeline run")
    run_timestamp: str = Field(description="ISO timestamp of run")
    resume_text: str = Field(description="Raw text of candidate resume")
    jd_text: str = Field(description="Raw text of target job description")
    parsed_resume: Optional[ParsedResume] = None
    parsed_jd: Optional[ParsedJD] = None
    match_breakdown: Optional[MatchScoreBreakdown] = None
    skill_gap_report: Optional[SkillGapReport] = None
    resume_optimization: Optional[ResumeOptimizationReport] = None
    cover_letter: Optional[CoverLetterResponse] = None
    interview_prep: Optional[InterviewPrepReport] = None
    company_research: Optional[CompanyResearchReport] = None
    guardrail_result: Optional[GuardrailVerificationResult] = None
    execution_time_seconds: float = 0.0


# --- TRACKER SCHEMA ---

class ApplicationRecord(BaseModel):
    id: Optional[int] = None
    company: str
    role: str
    match_score: float = 0.0
    status: Literal["Wishlist", "Applied", "Screening", "Interviewing", "Offer", "Rejected"] = "Applied"
    date_applied: str
    notes: Optional[str] = ""
    resume_filename: Optional[str] = None
    job_url: Optional[str] = None
