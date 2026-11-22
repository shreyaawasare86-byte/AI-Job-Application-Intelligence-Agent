"""Streamlit interactive user interface for AI Job Application Intelligence Agent."""
import asyncio
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure event loop for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.core.schemas import PipelineExecutionState, ApplicationRecord
from app.core.orchestrator import pipeline_orchestrator
from app.services.parser_service import DocumentParserService
from app.services.tracker_service import tracker_service

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Job Application Intelligence Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS STYLING ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .tag-matched {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .tag-partial {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .tag-missing {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 3px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        border-radius: 8px 8px 0px 0px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE INITIALIZATION ---
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "resume_text_input" not in st.session_state:
    st.session_state.resume_text_input = ""
if "jd_text_input" not in st.session_state:
    st.session_state.jd_text_input = ""
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "Select sample..."


# --- HELPER: SAMPLE LOADER ---
SAMPLE_DIR = settings.SAMPLE_DIR
SAMPLE_PROFILES = {
    "Senior ML Engineer (LLM/RAG)": ("resume_1_ml_engineer.txt", "jd_1_senior_ml_engineer.txt"),
    "Full Stack React/Python Dev": ("resume_2_fullstack_dev.txt", "jd_2_fullstack_react_python.txt"),
    "Lead Data Analyst (BI/SQL)": ("resume_3_data_analyst.txt", "jd_3_lead_data_analyst.txt"),
    "Cloud DevOps & SRE": ("resume_4_devops_cloud.txt", "jd_4_site_reliability_engineer.txt"),
    "Junior Backend Developer": ("resume_5_junior_developer.txt", "jd_5_junior_backend_dev.txt"),
}


# --- SIDEBAR: CONTROLS & INPUTS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8644/8644445.png", width=64)
    st.title("Job Application AI")
    st.caption("Multi-Agent Career Intelligence System")

    st.markdown("---")
    st.subheader("⚙️ Settings & Model")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=settings.GEMINI_API_KEY,
        type="password",
        help="Optional: Enter your Gemini API Key or configure via .env. If left blank, agent utilizes intelligent mock mode.",
    )
    if api_key_input:
        settings.GEMINI_API_KEY = api_key_input

    model_choice = st.selectbox(
        "LLM Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        index=0,
    )
    settings.GEMINI_MODEL = model_choice

    status_pill = "🟢 Active (Gemini API)" if settings.has_valid_gemini_key else "🟡 Offline / Demo Mode"
    st.caption(f"Status: **{status_pill}**")

    st.markdown("---")
    st.subheader("📁 Load Pre-built Profile")
    sample_select = st.selectbox(
        "Quick-fill sample career pair:",
        options=["Select sample..."] + list(SAMPLE_PROFILES.keys()),
        index=0,
    )

    if sample_select != "Select sample..." and sample_select != st.session_state.selected_sample:
        resume_file, jd_file = SAMPLE_PROFILES[sample_select]
        r_path = SAMPLE_DIR / resume_file
        j_path = SAMPLE_DIR / jd_file
        if r_path.exists() and j_path.exists():
            st.session_state.resume_text_input = r_path.read_text(encoding="utf-8")
            st.session_state.jd_text_input = j_path.read_text(encoding="utf-8")
            st.session_state.selected_sample = sample_select
            st.success(f"Loaded {sample_select}!")

    st.markdown("---")
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt", "md"],
        help="Parse text directly from your resume file.",
    )

    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            extracted = DocumentParserService.extract_from_bytes(bytes_data, uploaded_file.name)
            st.session_state.resume_text_input = extracted
            st.success(f"Parsed {uploaded_file.name} ({len(extracted)} chars)")
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

    cover_tone = st.radio(
        "Cover Letter Tone",
        options=["confident", "formal", "friendly"],
        index=0,
        horizontal=True,
    )

    analyze_button = st.button("🚀 Analyze Application", type="primary", use_container_width=True)


# --- MAIN HEADER ---
st.markdown('<div class="main-header">🎯 AI Job Application Intelligence Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-agent resume analysis, truthful STAR optimizations, ATS keyword scoring, cover letter generator & interview prep.</div>',
    unsafe_allow_html=True,
)

# --- INPUT SECTION (IF NO ANALYSIS RUN YET) ---
if not st.session_state.pipeline_result:
    col_r, col_j = st.columns(2)
    with col_r:
        st.subheader("1. Candidate Resume")
        resume_text_area = st.text_area(
            "Paste or inspect resume text:",
            value=st.session_state.resume_text_input,
            height=320,
            placeholder="Paste complete resume text here or upload in the sidebar...",
        )
        st.session_state.resume_text_input = resume_text_area

    with col_j:
        st.subheader("2. Target Job Description")
        jd_text_area = st.text_area(
            "Paste job description text:",
            value=st.session_state.jd_text_input,
            height=320,
            placeholder="Paste target job posting here or select a sample profile in the sidebar...",
        )
        st.session_state.jd_text_input = jd_text_area


# --- RUN PIPELINE LOGIC ---
if analyze_button:
    r_text = st.session_state.resume_text_input.strip()
    j_text = st.session_state.jd_text_input.strip()

    if not r_text:
        st.error("Please provide your resume text or upload a document in the sidebar.")
    elif not j_text:
        st.error("Please provide the target job description text.")
    else:
        with st.spinner("🤖 Multi-agent team executing (Parsing, Matching, Skill-Gap, STAR Optimizer, Cover Letter, Q&A)..."):
            try:
                # Run async orchestrator
                pipeline_state = asyncio.run(
                    pipeline_orchestrator.run(
                        resume_text=r_text,
                        jd_text=j_text,
                        tone=cover_tone,
                    )
                )
                st.session_state.pipeline_result = pipeline_state
                st.toast("Intelligence pipeline successfully completed!", icon="✅")
            except Exception as e:
                st.error(f"Error during pipeline execution: {e}")


# --- RESULTS DASHBOARD ---
if st.session_state.pipeline_result:
    state: PipelineExecutionState = st.session_state.pipeline_result
    mb = state.match_breakdown

    # Quick reset / re-analyze bar
    c_top_info, c_top_btn = st.columns([4, 1])
    with c_top_info:
        st.caption(
            f"Run ID: `{state.run_id[:8]}...` | Executed in **{state.execution_time_seconds}s** | "
            f"Role: **{state.parsed_jd.role_title if state.parsed_jd else 'Target Role'}** at **{state.parsed_jd.company_name if state.parsed_jd else 'Target Company'}**"
        )
    with c_top_btn:
        if st.button("🔄 Analyze New Pair", use_container_width=True):
            st.session_state.pipeline_result = None
            st.rerun()

    # TABS FOR INTELLIGENCE SUITE
    tab_dash, tab_gaps, tab_opt, tab_letter, tab_interview, tab_research, tab_tracker = st.tabs([
        "📊 Match Score",
        "🧩 Skill Gaps",
        "✨ Resume Optimizer",
        "✉️ Cover Letter",
        "🎙️ Interview Coach",
        "🏢 Company Intel",
        "📋 Application Tracker",
    ])

    # --- TAB 1: EXECUTIVE DASHBOARD ---
    with tab_dash:
        st.subheader("Match Score & Alignment Breakdown")

        # Top Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            tier_color = "#10B981" if mb.overall_score >= 80 else ("#F59E0B" if mb.overall_score >= 60 else "#EF4444")
            st.markdown(
                f"""
                <div class="metric-card" style="border-top: 4px solid {tier_color};">
                    <div class="metric-value" style="color: {tier_color};">{mb.overall_score}</div>
                    <div class="metric-label">{mb.match_tier}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{mb.skills_score}</div>
                    <div class="metric-label">Skills (40%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{mb.experience_score}</div>
                    <div class="metric-label">Experience (30%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{mb.keyword_score}</div>
                    <div class="metric-label">ATS Keywords (20%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{mb.education_score}</div>
                    <div class="metric-label">Education (10%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Reasoning Summary Callout
        st.info(f"**Evaluation Rationale:** {mb.reasoning}")

        # Strengths & Gaps Two Columns
        c_str, c_gap = st.columns(2)
        with c_str:
            st.markdown("#### ✅ Top Candidate Strengths")
            if mb.strengths:
                for s in mb.strengths:
                    st.markdown(f"- **{s}**")
            else:
                st.markdown("_No major advantages identified._")

        with c_gap:
            st.markdown("#### ⚠️ Key Gaps to Address")
            if mb.gaps_summary:
                for g in mb.gaps_summary:
                    st.markdown(f"- {g}")
            else:
                st.markdown("_No critical qualification deficits detected._")

        # Save to Tracker Action
        st.markdown("---")
        c_tr_btn, c_tr_msg = st.columns([1, 3])
        with c_tr_btn:
            if st.button("📌 Save to Application Tracker", type="secondary"):
                new_app = ApplicationRecord(
                    company=state.parsed_jd.company_name if state.parsed_jd else "Company",
                    role=state.parsed_jd.role_title if state.parsed_jd else "Role",
                    match_score=mb.overall_score,
                    status="Applied",
                    date_applied=datetime.now().strftime("%Y-%m-%d"),
                    notes=f"Match Tier: {mb.match_tier}. Top strength: {mb.strengths[0] if mb.strengths else ''}",
                )
                tracker_service.add_application(new_app)
                st.toast("Added to SQLite Application Tracker!", icon="📋")

    # --- TAB 2: SKILL GAP & LEARNING BLUEPRINTS ---
    with tab_gaps:
        sg = state.skill_gap_report
        st.subheader("Skill Classification & Targeted Upskilling")
        if sg:
            st.markdown(f"**Executive Summary:** {sg.summary}")

            # Skill Badge Visuals
            st.markdown("##### Demonstrated & Matched Skills")
            if sg.matched_skills:
                st.markdown(
                    " ".join([f'<span class="tag-matched">✓ {s}</span>' for s in sg.matched_skills]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("_None_")

            st.markdown("##### Partial or Adjacent Skills")
            if sg.partial_skills:
                st.markdown(
                    " ".join([f'<span class="tag-partial">~ {s}</span>' for s in sg.partial_skills]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("_None_")

            st.markdown("##### Missing Requirements")
            if sg.missing_skills:
                st.markdown(
                    " ".join([f'<span class="tag-missing">✗ {s}</span>' for s in sg.missing_skills]),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("_No missing requirements!_")

            st.markdown("---")
            st.subheader("🛠️ High-Impact Weekend Mini-Projects & Learning Paths")
            for gap in sg.detailed_gaps:
                with st.expander(f"**{gap.skill_name}** ({gap.status.upper()} - {gap.importance.upper()})"):
                    st.markdown(f"**Analysis:** {gap.gap_analysis}")
                    if gap.candidate_evidence:
                        st.markdown(f"**Current Resume Signal:** _{gap.candidate_evidence}_")
                    if gap.mini_project_idea:
                        st.markdown(f"**💡 Weekend Portfolio Project:** {gap.mini_project_idea}")
                    if gap.learning_resources:
                        st.markdown("**📚 Recommended Study Materials:**")
                        for lr in gap.learning_resources:
                            st.markdown(f"- [{lr.title}]({lr.url_or_platform}) ({lr.resource_type} • ~{lr.estimated_hours} hrs)")

    # --- TAB 3: RESUME OPTIMIZER & GUARDRAIL AUDIT ---
    with tab_opt:
        ro = state.resume_optimization
        gr = state.guardrail_result
        st.subheader("STAR & XYZ Impact Bullet Enhancements")

        # Guardrail Audit Banner
        if gr and gr.is_valid and not gr.hallucinated_metrics:
            st.success("🛡️ **Anti-Hallucination Guardrail: Verified.** All rewritten bullets strictly adhere to facts stated in your resume.")
        else:
            st.warning(f"⚠️ **Candidate Metric Audit:** {gr.verification_notes if gr else 'Verify highlighted metrics.'}")

        if ro and ro.bullet_rewrites:
            st.markdown("#### Side-by-Side Bullet Comparisons")
            for i, rewrite in enumerate(ro.bullet_rewrites, 1):
                with st.container():
                    st.markdown(f"**Bullet #{i}** (Format: `{rewrite.format_used}` • Groundedness: `{rewrite.groundedness_score*100:.0f}%`)")
                    c_orig, c_new = st.columns(2)
                    with c_orig:
                        st.markdown("**Original:**")
                        st.caption(rewrite.original_bullet)
                    with c_new:
                        st.markdown("**Optimized (STAR/XYZ):**")
                        st.markdown(f"**{rewrite.rewritten_bullet}**")

                    if rewrite.requires_confirmation:
                        st.warning(f"🔍 **Confirm Metric:** {rewrite.confirmation_note}")

                    if rewrite.ats_keywords_incorporated:
                        kw_tags = " ".join([f"`{k}`" for k in rewrite.ats_keywords_incorporated])
                        st.markdown(f"ATS Keywords Woven: {kw_tags}")
                    st.markdown("---")

            # General Critique
            if ro.general_critique:
                st.info(f"**Resume Strategy Critique:** {ro.general_critique}")

            # Download Button
            suggestions_text = "# Optimized Resume Bullets\n\n"
            for i, rewrite in enumerate(ro.bullet_rewrites, 1):
                suggestions_text += f"## Bullet {i}\n"
                suggestions_text += f"- Original: {rewrite.original_bullet}\n"
                suggestions_text += f"- Enhanced: {rewrite.rewritten_bullet}\n"
                if rewrite.confirmation_note:
                    suggestions_text += f"- Note: {rewrite.confirmation_note}\n\n"

            st.download_button(
                label="📥 Download Optimized Bullets (.md)",
                data=suggestions_text,
                file_name="optimized_resume_bullets.md",
                mime="text/markdown",
            )

    # --- TAB 4: TAILORED COVER LETTER ---
    with tab_letter:
        cl = state.cover_letter
        st.subheader("Personalized, High-Converting Cover Letter")
        if cl:
            st.caption(f"Recipient: **{cl.recipient_role}** | Tone: **{cl.tone.upper()}**")

            letter_body_area = st.text_area(
                "Cover Letter Text (Editable):",
                value=cl.letter_body,
                height=340,
            )

            st.markdown("##### Core Value Propositions Featured:")
            for prop in cl.key_value_props_highlighted:
                st.markdown(f"- {prop}")

            st.download_button(
                label="📥 Download Cover Letter (.txt)",
                data=letter_body_area,
                file_name=f"cover_letter_{state.parsed_jd.company_name if state.parsed_jd else 'job'}.txt",
                mime="text/plain",
            )

    # --- TAB 5: INTERVIEW PREP COACH ---
    with tab_interview:
        ip = state.interview_prep
        st.subheader("Role-Specific Interview Simulator")
        if ip:
            if ip.behavioral_framework_tips:
                st.info(f"💡 **Coaching Tip:** {ip.behavioral_framework_tips}")

            st.markdown("#### High-Probability Questions & Resume Answer Blueprints")
            for i, q in enumerate(ip.questions, 1):
                badge_type = q.category.replace("_", " ").title()
                with st.expander(f"**Q{i}: [{badge_type}]** {q.question}"):
                    st.markdown(f"**🎯 Why they ask this:** _{q.rationale}_")
                    st.markdown("**📝 Your STAR Answer Framework:**")
                    st.markdown(f"> {q.answer_outline}")
                    st.markdown("**Key Points to Emphasize:**")
                    for pt in q.key_talking_points:
                        st.markdown(f"- {pt}")

            st.markdown("---")
            st.markdown("#### 🙋 Smart Questions to Ask the Interview Panel")
            for rev_q in ip.questions_for_candidate_to_ask:
                st.markdown(f"- **{rev_q}**")

    # --- TAB 6: COMPANY RESEARCH ---
    with tab_research:
        cr = state.company_research
        st.subheader("Company Strategic Briefing")
        if cr:
            st.markdown(f"### {cr.company_name} • _{cr.industry}_")
            st.markdown(f"**Mission & Overview:** {cr.overview}")

            c_val, c_pri = st.columns(2)
            with c_val:
                st.markdown("##### 🤝 Culture & Engineering Values")
                for val in cr.values_and_culture:
                    st.markdown(f"- {val}")

            with c_pri:
                st.markdown("##### 🚀 Current Strategic Initiatives")
                for pri in cr.recent_priorities:
                    st.markdown(f"- {pri}")

            st.markdown("---")
            st.markdown("##### 💡 Strategic Conversation Hooks for Interviews:")
            for hook in cr.strategic_talking_points_for_interview:
                st.markdown(f"- {hook}")

    # --- TAB 7: APPLICATION TRACKER ---
    with tab_tracker:
        st.subheader("Job Application Tracker (SQLite)")
        apps = tracker_service.list_applications()

        # Metrics
        analytics = tracker_service.get_analytics_summary()
        c_k1, c_k2, c_k3 = st.columns(3)
        c_k1.metric("Total Tracked", analytics["total_applications"])
        c_k2.metric("Avg Match Score", f"{analytics['average_match_score']}/100")
        c_k3.metric("Offers", analytics["status_breakdown"].get("Offer", 0))

        st.markdown("---")
        # Filter & Table
        filter_status = st.selectbox(
            "Filter by Status:",
            options=["All", "Wishlist", "Applied", "Screening", "Interviewing", "Offer", "Rejected"],
            index=0,
        )

        filtered_apps = apps if filter_status == "All" else [a for a in apps if a.status == filter_status]

        if filtered_apps:
            table_data = []
            for a in filtered_apps:
                table_data.append({
                    "ID": a.id,
                    "Company": a.company,
                    "Role": a.role,
                    "Score": a.match_score,
                    "Status": a.status,
                    "Date Applied": a.date_applied,
                    "Notes": a.notes,
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

            # CSV Download
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Export Applications (.csv)",
                data=csv_data,
                file_name="job_applications.csv",
                mime="text/csv",
            )
        else:
            st.info("No applications match the current filter.")
