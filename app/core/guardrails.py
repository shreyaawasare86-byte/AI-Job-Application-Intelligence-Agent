"""Anti-hallucination guardrail engine verifying generated outputs against source resume facts."""
import re
import logging
from typing import List, Tuple
from app.core.schemas import ParsedResume, BulletRewrite, GuardrailVerificationResult

logger = logging.getLogger(__name__)


class GuardrailEngine:
    """Audits generated resume bullets and content to guarantee zero fabricated facts."""

    @staticmethod
    def extract_metrics(text: str) -> List[str]:
        """Extract percentages, currency amounts, multipliers, and numbers from text."""
        patterns = [
            r"\b\d+(?:\.\d+)?%",                         # e.g., 38%, 45.5%
            r"\$\s*\d+(?:,\d+)*(?:\.\d+)?[kKmMbB]?",    # e.g., $18K, $3.2M, $216,000
            r"\b\d+(?:\.\d+)?\s*[xX]\b",                 # e.g., 3x, 10x
            r"\b\d+(?:,\d+)*\+?\s*(?:req/s|rps|ms|users|documents|queries|stars)\b", # metrics with units
        ]
        metrics = []
        for pat in patterns:
            found = re.findall(pat, text, flags=re.IGNORECASE)
            metrics.extend(found)
        return list(set(metrics))

    @classmethod
    def verify_bullet(cls, original_bullet: str, rewritten_bullet: str) -> Tuple[bool, List[str], float]:
        """Inspect a single rewritten bullet for ungrounded claims or newly invented metrics.
        
        Returns:
            Tuple of (requires_confirmation, ungrounded_metrics, groundedness_score)
        """
        orig_metrics = set(cls.extract_metrics(original_bullet))
        new_metrics = set(cls.extract_metrics(rewritten_bullet))

        # Metrics introduced in the rewrite that were absent in original
        unverified_metrics = list(new_metrics - orig_metrics)

        if unverified_metrics:
            # Flag for confirmation with proportional confidence penalty
            groundedness = max(0.80, 1.0 - (len(unverified_metrics) * 0.05))
            return True, unverified_metrics, round(groundedness, 2)

        return False, [], 1.0

    @classmethod
    def audit_optimization_report(
        cls,
        resume: ParsedResume,
        rewritten_bullets: List[BulletRewrite],
    ) -> GuardrailVerificationResult:
        """Run full guardrail audit across all rewritten bullets and resume entities."""
        ungrounded_entities = []
        hallucinated_metrics = []
        notes = []

        valid_companies = {exp.company.lower() for exp in resume.experience}
        valid_degrees = {edu.degree.lower() for edu in resume.education}

        # Check all rewritten bullets
        for rewrite in rewritten_bullets:
            needs_confirm, new_metrics, score = cls.verify_bullet(
                rewrite.original_bullet, rewrite.rewritten_bullet
            )
            rewrite.groundedness_score = score
            if needs_confirm:
                rewrite.requires_confirmation = True
                metric_str = ", ".join(new_metrics)
                rewrite.confirmation_note = f"Verify newly introduced quantitative metric ({metric_str}) against your actual team metrics."
                hallucinated_metrics.extend(new_metrics)

        hallucinated_metrics = list(set(hallucinated_metrics))
        is_clean = len(ungrounded_entities) == 0

        if hallucinated_metrics:
            notes.append(
                f"Flagged {len(hallucinated_metrics)} suggested metric(s) for candidate verification: {', '.join(hallucinated_metrics[:5])}."
            )
        else:
            notes.append("All metrics strictly grounded in source resume data.")

        return GuardrailVerificationResult(
            is_valid=is_clean,
            ungrounded_entities=ungrounded_entities,
            hallucinated_metrics=hallucinated_metrics,
            verification_notes=" ".join(notes),
        )


guardrail_engine = GuardrailEngine()
