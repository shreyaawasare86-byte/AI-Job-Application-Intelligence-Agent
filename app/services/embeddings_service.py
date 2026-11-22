"""Semantic similarity and text embedding engine with sentence-transformers and TF-IDF fallback."""
import logging
import re
from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Computes semantic embeddings and cosine similarities with resilient fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._st_model = None
        self._has_st = False

        # Attempt to load sentence-transformers if available
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name)
            self._has_st = True
            logger.info(f"Loaded sentence-transformers model: {model_name}")
        except Exception as e:
            logger.warning(
                f"sentence-transformers not available or failed to load ({e}). "
                "Using high-performance TF-IDF n-gram semantic similarity fallback."
            )
            self._has_st = False

    @property
    def is_using_sentence_transformers(self) -> bool:
        return self._has_st

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two text snippets (0.0 to 1.0)."""
        if not text_a or not text_b:
            return 0.0
        
        # If texts are exact matches (case-insensitive)
        if text_a.strip().lower() == text_b.strip().lower():
            return 1.0

        if self._has_st and self._st_model is not None:
            try:
                embeddings = self._st_model.encode([text_a, text_b], convert_to_numpy=True)
                sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
                return max(0.0, min(1.0, sim))
            except Exception as e:
                logger.warning(f"SentenceTransformer encoding failed: {e}. Falling back to TF-IDF.")

        # Fallback: TF-IDF with character and word n-grams
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                analyzer="word",
                lowercase=True,
                token_pattern=r"(?u)\b[\w\-\+\#\.]+\b",
            )
            tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
            sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            return max(0.0, min(1.0, sim))
        except Exception:
            # Token set intersection as ultimate fallback
            tokens_a = set(re.findall(r"\w+", text_a.lower()))
            tokens_b = set(re.findall(r"\w+", text_b.lower()))
            if not tokens_a or not tokens_b:
                return 0.0
            return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))

    def batch_similarity(self, target_query: str, candidates: List[str]) -> List[float]:
        """Compute cosine similarity between one target query and a list of candidates."""
        if not candidates or not target_query:
            return [0.0] * len(candidates)

        return [self.compute_similarity(target_query, cand) for cand in candidates]

    def match_skills(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        similarity_threshold: float = 0.65,
        partial_threshold: float = 0.45,
    ) -> Tuple[List[str], List[str], List[str], float]:
        """Classify skills into matched, partial, missing and compute overall match ratio.
        
        Returns:
            Tuple of (matched_skills, partial_skills, missing_skills, coverage_score)
        """
        if not required_skills:
            return candidate_skills, [], [], 1.0

        matched = []
        partial = []
        missing = []

        norm_candidate = {s.lower().strip(): s for s in candidate_skills}

        for req in required_skills:
            req_clean = req.lower().strip()
            # Exact match check
            if req_clean in norm_candidate:
                matched.append(req)
                continue

            # Substring match (e.g. "React" in "React.js" or "PyTorch" in "PyTorch Lightning")
            sub_match = any(req_clean in c or c in req_clean for c in norm_candidate)
            if sub_match:
                matched.append(req)
                continue

            # Semantic similarity check against all candidate skills
            best_sim = 0.0
            for cand_skill in candidate_skills:
                sim = self.compute_similarity(req, cand_skill)
                if sim > best_sim:
                    best_sim = sim

            if best_sim >= similarity_threshold:
                matched.append(req)
            elif best_sim >= partial_threshold:
                partial.append(req)
            else:
                missing.append(req)

        # Weighted score: matched = 1.0, partial = 0.5
        total_req = len(required_skills)
        score = (len(matched) * 1.0 + len(partial) * 0.5) / total_req if total_req > 0 else 1.0
        return matched, partial, missing, max(0.0, min(1.0, score))


# Singleton instance
embedding_service = EmbeddingService()
