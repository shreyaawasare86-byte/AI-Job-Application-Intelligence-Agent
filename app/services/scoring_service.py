"""Hybrid deterministic scoring engine for resume-to-job matching."""
import logging
import re
from typing import Dict, Any, List
from app.core.schemas import ParsedResume, ParsedJD, MatchScoreBreakdown
from app.services.embeddings_service import embedding_service

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Calculates granular match scores across skills, experience, education, and ATS keywords."""

    WEIGHT_SKILLS = 0.40
    WEIGHT_EXPERIENCE = 0.30
    WEIGHT_KEYWORDS = 0.20
    WEIGHT_EDUCATION = 0.10

    @classmethod
    def calculate_skills_score(cls, resume: ParsedResume, jd: ParsedJD) -> Dict[str, Any]:
        """Evaluate hard and soft skill alignment against required and preferred skills."""
        all_candidate_skills = list(set(
            resume.technical_skills
            + resume.tools_and_frameworks
            + resume.soft_skills
        ))

        # Required skills coverage
        req_matched, req_partial, req_missing, req_ratio = embedding_service.match_skills(
            all_candidate_skills, jd.required_skills
        )

        # Preferred skills coverage
        pref_matched, pref_partial, pref_missing, pref_ratio = embedding_service.match_skills(
            all_candidate_skills, jd.preferred_skills
        )

        if jd.required_skills and jd.preferred_skills:
            final_skills_ratio = (req_ratio * 0.75) + (pref_ratio * 0.25)
        elif jd.required_skills:
            final_skills_ratio = req_ratio
        elif jd.preferred_skills:
            final_skills_ratio = pref_ratio
        else:
            final_skills_ratio = 0.80  # Default baseline if JD specifies no skills

        skills_score = round(final_skills_ratio * 100.0, 1)

        return {
            "score": max(0.0, min(100.0, skills_score)),
            "req_matched": req_matched,
            "req_partial": req_partial,
            "req_missing": req_missing,
            "pref_matched": pref_matched,
            "pref_partial": pref_partial,
            "pref_missing": pref_missing,
        }

    @classmethod
    def calculate_experience_score(cls, resume: ParsedResume, jd: ParsedJD) -> Dict[str, Any]:
        """Evaluate total experience years and seniority fit."""
        cand_years = resume.total_years_experience
        req_years = jd.min_years_experience

        # Years of experience ratio
        if req_years <= 0:
            years_score = 100.0
        elif cand_years >= req_years:
            years_score = 100.0
        elif cand_years >= req_years * 0.75:
            years_score = 85.0
        elif cand_years >= req_years * 0.50:
            years_score = 65.0
        else:
            years_score = max(20.0, (cand_years / req_years) * 60.0)

        # Seniority title match check
        title_boost = 0.0
        cand_titles = " ".join([exp.role_title for exp in resume.experience]).lower()
        jd_title = jd.role_title.lower()
        if any(term in cand_titles and term in jd_title for term in ["senior", "lead", "staff", "principal", "junior", "intern"]):
            title_boost = 15.0

        # Responsibility semantic alignment
        exp_bullets = []
        for exp in resume.experience:
            exp_bullets.extend(exp.bullets)
        
        bullet_text = " ".join(exp_bullets)
        resp_text = " ".join(jd.core_responsibilities)
        if bullet_text and resp_text:
            semantic_sim = embedding_service.compute_similarity(bullet_text, resp_text)
            responsibility_score = min(100.0, max(60.0, (semantic_sim * 100.0) + title_boost))
        else:
            responsibility_score = years_score

        # Weighted combination: 70% years fit + 30% responsibility alignment
        exp_score = round((years_score * 0.70) + (responsibility_score * 0.30), 1)

        return {
            "score": max(0.0, min(100.0, exp_score)),
            "cand_years": cand_years,
            "req_years": req_years,
            "responsibility_alignment": round(responsibility_score, 1),
        }

    @classmethod
    def calculate_keyword_score(cls, resume_text: str, jd: ParsedJD) -> Dict[str, Any]:
        """Compute exact and normalized keyword frequency overlap."""
        if not jd.ats_keywords:
            return {"score": 85.0, "found_keywords": [], "missing_keywords": []}

        resume_lower = resume_text.lower()
        found_keywords = []
        missing_keywords = []

        for kw in jd.ats_keywords:
            kw_clean = kw.strip().lower()
            if not kw_clean:
                continue
            # Word boundary regex check
            pattern = r"\b" + re.escape(kw_clean) + r"\b"
            if re.search(pattern, resume_lower):
                found_keywords.append(kw)
            else:
                missing_keywords.append(kw)

        total_kw = len(jd.ats_keywords)
        coverage = len(found_keywords) / total_kw if total_kw > 0 else 1.0
        kw_score = round(coverage * 100.0, 1)

        return {
            "score": max(0.0, min(100.0, kw_score)),
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
        }

    @classmethod
    def calculate_education_score(cls, resume: ParsedResume, jd: ParsedJD) -> Dict[str, Any]:
        """Evaluate education credentials and certifications against job requirements."""
        if not jd.education_requirements:
            return {"score": 90.0, "details": "No specific education requirement enforced."}

        edu_text = " ".join([f"{e.degree} {e.field_of_study or ''} {e.institution}" for e in resume.education]).lower()
        cert_text = " ".join(resume.certifications).lower()
        full_edu_cred = f"{edu_text} {cert_text}"

        score = 70.0  # Base score for having documented higher education
        req_joined = " ".join(jd.education_requirements).lower()

        # Check degree levels
        has_phd = "ph.d" in full_edu_cred or "doctorate" in full_edu_cred
        has_ms = "m.s." in full_edu_cred or "master" in full_edu_cred or "mba" in full_edu_cred
        has_bs = "b.s." in full_edu_cred or "bachelor" in full_edu_cred or "b.a." in full_edu_cred or "b.e." in full_edu_cred

        if "ph.d" in req_joined or "doctorate" in req_joined:
            score = 100.0 if has_phd else (80.0 if has_ms else 60.0)
        elif "master" in req_joined or "m.s." in req_joined:
            score = 100.0 if (has_ms or has_phd) else (85.0 if has_bs else 60.0)
        elif "bachelor" in req_joined or "b.s." in req_joined or "degree" in req_joined:
            score = 100.0 if (has_bs or has_ms or has_phd) else 75.0
        else:
            score = 90.0

        # Bonus for relevant certifications
        if resume.certifications:
            score = min(100.0, score + 5.0)

        return {"score": round(score, 1), "details": "Degree level matched."}

    @classmethod
    def evaluate(
        cls,
        resume: ParsedResume,
        jd: ParsedJD,
        resume_text: str,
        jd_text: str,
    ) -> MatchScoreBreakdown:
        """Run full hybrid scoring engine and synthesize transparent breakdown."""
        skills_res = cls.calculate_skills_score(resume, jd)
        exp_res = cls.calculate_experience_score(resume, jd)
        kw_res = cls.calculate_keyword_score(resume_text, jd)
        edu_res = cls.calculate_education_score(resume, jd)

        s_score = skills_res["score"]
        e_score = exp_res["score"]
        k_score = kw_res["score"]
        ed_score = edu_res["score"]

        # Composite score
        overall = round(
            (s_score * cls.WEIGHT_SKILLS)
            + (e_score * cls.WEIGHT_EXPERIENCE)
            + (k_score * cls.WEIGHT_KEYWORDS)
            + (ed_score * cls.WEIGHT_EDUCATION),
            1,
        )
        overall = max(0.0, min(100.0, overall))

        # Determine Match Tier
        if overall >= 85.0:
            match_tier = "Strong Match"
        elif overall >= 70.0:
            match_tier = "Good Match"
        elif overall >= 50.0:
            match_tier = "Moderate Match"
        else:
            match_tier = "Weak Match"

        # Synthesize top strengths
        strengths = []
        if skills_res["req_matched"]:
            strengths.append(f"Matched core required skills: {', '.join(skills_res['req_matched'][:4])}")
        if exp_res["cand_years"] >= exp_res["req_years"] and exp_res["req_years"] > 0:
            strengths.append(f"Exceeds minimum experience threshold ({resume.total_years_experience:.1f} yrs vs {jd.min_years_experience:.1f} yrs req)")
        if len(kw_res["found_keywords"]) >= 3:
            strengths.append(f"High ATS keyword density ({len(kw_res['found_keywords'])} target terms detected)")

        # Synthesize top gaps
        gaps = []
        if skills_res["req_missing"]:
            gaps.append(f"Missing required skills: {', '.join(skills_res['req_missing'][:4])}")
        if skills_res["pref_missing"]:
            gaps.append(f"Missing preferred skills: {', '.join(skills_res['pref_missing'][:3])}")
        if exp_res["cand_years"] < exp_res["req_years"]:
            gaps.append(f"Experience deficit: {exp_res['cand_years']:.1f} yrs vs {exp_res['req_years']:.1f} yrs required")
        if kw_res["missing_keywords"]:
            gaps.append(f"Unmatched ATS keywords: {', '.join(kw_res['missing_keywords'][:4])}")

        reasoning = (
            f"Overall score of {overall}/100 ({match_tier}). "
            f"Candidate achieved {s_score}/100 on skills alignment, {e_score}/100 on experience/seniority fit, "
            f"{k_score}/100 on ATS keyword density, and {ed_score}/100 on academic/certification credentials. "
            f"Key strength: {strengths[0] if strengths else 'Broad profile'}. "
            f"Primary area for enhancement: {gaps[0] if gaps else 'None detected'}."
        )

        return MatchScoreBreakdown(
            overall_score=overall,
            skills_score=s_score,
            experience_score=e_score,
            education_score=ed_score,
            keyword_score=k_score,
            match_tier=match_tier,
            reasoning=reasoning,
            strengths=strengths,
            gaps_summary=gaps,
        )


scoring_engine = ScoringEngine()
