"""Tests for document and file parser service."""
import pytest
from pathlib import Path
from app.services.parser_service import DocumentParserService

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


def test_clean_text_normalization():
    dirty_text = "Line 1   with    extra    spaces.\r\n\r\n\r\n\r\nLine 2.\xa0Another word."
    cleaned = DocumentParserService.clean_text(dirty_text)
    assert "\r" not in cleaned
    assert "\xa0" not in cleaned
    assert "with extra spaces." in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_from_plain_text_file():
    resume_path = SAMPLE_DIR / "resume_1_ml_engineer.txt"
    assert resume_path.exists(), f"Sample file missing: {resume_path}"

    text = DocumentParserService.extract_from_path(resume_path)
    assert "ALEX RIVERA" in text
    assert "NeuralScale AI" in text
    assert "PyTorch" in text


def test_extract_from_bytes_txt():
    sample_content = b"Candidate Name\nSoftware Engineer\nPython, Docker, FastAPI"
    text = DocumentParserService.extract_from_bytes(sample_content, "resume.txt")
    assert "Candidate Name" in text
    assert "FastAPI" in text


def test_unsupported_file_extension():
    with pytest.raises(ValueError, match="Unsupported file"):
        DocumentParserService.extract_from_bytes(b"data", "resume.xyz123")


def test_extract_from_docx_file():
    docx_path = SAMPLE_DIR / "sample_resume.docx"
    if docx_path.exists():
        text = DocumentParserService.extract_from_path(docx_path)
        assert "ALEX RIVERA" in text
        assert "NeuralScale AI" in text
        assert "Qdrant" in text

