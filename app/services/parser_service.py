"""Document and URL parsing service for resumes and job descriptions."""
import io
import re
from pathlib import Path
from typing import Union, Optional
import httpx


class DocumentParserService:
    """Extracts raw text from PDF, DOCX, TXT, and Web URLs."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize line breaks and excessive whitespace while preserving structure."""
        if not text:
            return ""
        # Replace non-breaking spaces and carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
        # Collapse 3+ consecutive newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse multiple inline spaces to a single space
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @classmethod
    def extract_from_pdf(cls, file_source: Union[str, Path, bytes, io.BytesIO]) -> str:
        """Extract text from PDF using pdfplumber with fallback."""
        try:
            import pdfplumber
            extracted_pages = []

            if isinstance(file_source, (str, Path)):
                with pdfplumber.open(file_source) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text(layout=True) or page.extract_text() or ""
                        extracted_pages.append(page_text)
            else:
                stream = file_source if isinstance(file_source, io.BytesIO) else io.BytesIO(file_source)
                with pdfplumber.open(stream) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text(layout=True) or page.extract_text() or ""
                        extracted_pages.append(page_text)

            full_text = "\n\n".join(extracted_pages)
            return cls.clean_text(full_text)
        except Exception as e:
            # Fallback using pypdf or pdfminer if pdfplumber fails
            try:
                from pdfminer.high_level import extract_text as pdfminer_extract
                if isinstance(file_source, (str, Path)):
                    return cls.clean_text(pdfminer_extract(str(file_source)))
                else:
                    stream = file_source if isinstance(file_source, io.BytesIO) else io.BytesIO(file_source)
                    return cls.clean_text(pdfminer_extract(stream))
            except Exception as e2:
                raise ValueError(f"Failed to parse PDF document: {e} (fallback error: {e2})")

    @classmethod
    def extract_from_docx(cls, file_source: Union[str, Path, bytes, io.BytesIO]) -> str:
        """Extract text from DOCX files including paragraphs and tables."""
        try:
            import docx

            if isinstance(file_source, (str, Path)):
                doc = docx.Document(file_source)
            else:
                stream = file_source if isinstance(file_source, io.BytesIO) else io.BytesIO(file_source)
                doc = docx.Document(stream)

            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())

            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        full_text.append(" | ".join(row_cells))

            return cls.clean_text("\n".join(full_text))
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX document: {e}")

    @classmethod
    def extract_from_bytes(cls, file_bytes: bytes, filename: str) -> str:
        """Extract text from bytes based on file extension."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return cls.extract_from_pdf(file_bytes)
        elif ext in (".docx", ".doc"):
            return cls.extract_from_docx(file_bytes)
        elif ext in (".txt", ".md"):
            try:
                return cls.clean_text(file_bytes.decode("utf-8"))
            except UnicodeDecodeError:
                return cls.clean_text(file_bytes.decode("latin-1", errors="ignore"))
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @classmethod
    def extract_from_path(cls, file_path: Union[str, Path]) -> str:
        """Extract text from a file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return cls.extract_from_pdf(path)
        elif ext in (".docx", ".doc"):
            return cls.extract_from_docx(path)
        elif ext in (".txt", ".md"):
            return cls.clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @classmethod
    async def extract_from_url(cls, url: str) -> str:
        """Fetch and extract text from a job description URL."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text

        # Strip scripts, styles, and tags for clean text
        cleaned_html = re.sub(r"<(script|style).*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Convert paragraph breaks and list items to newlines
        cleaned_html = re.sub(r"<(p|br|div|li|h[1-6])[^>]*>", "\n", cleaned_html, flags=re.IGNORECASE)
        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", cleaned_html)
        return cls.clean_text(text)
