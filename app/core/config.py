"""Configuration management for AI Job Application Intelligence Agent."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file from project root if present
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Application settings with environment variable support."""

    # Project paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    SAMPLE_DIR: Path = BASE_DIR / "data" / "sample"
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", str(BASE_DIR / "data" / "cache")))
    DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", str(BASE_DIR / "data" / "applications.db")))

    # LLM Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    ENABLE_LLM_CACHE: bool = os.getenv("ENABLE_LLM_CACHE", "true").lower() in ("true", "1", "yes")

    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # API Server Settings
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    def __init__(self):
        # Ensure data directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if self.DATABASE_PATH.parent:
            self.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    @property
    def has_valid_gemini_key(self) -> bool:
        """Check if a Gemini API key is configured and not a placeholder."""
        return bool(
            self.GEMINI_API_KEY
            and not self.GEMINI_API_KEY.startswith("your_")
            and len(self.GEMINI_API_KEY) > 10
        )


settings = Settings()
