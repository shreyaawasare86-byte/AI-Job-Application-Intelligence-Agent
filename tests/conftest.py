"""Pytest configuration and test environment setup."""
import os
import pytest

# Force mock LLM provider and test environment during test runs
os.environ["TESTING"] = "1"
os.environ["ENVIRONMENT"] = "test"
os.environ["LLM_PROVIDER"] = "mock"

# Ensure settings picks up test environment
from app.core.config import settings
settings.ENVIRONMENT = "test"
settings.LLM_PROVIDER = "mock"
