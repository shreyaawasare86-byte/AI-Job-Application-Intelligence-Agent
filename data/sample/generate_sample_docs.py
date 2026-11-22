"""Helper script to generate sample .docx and .pdf files for testing."""
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent

def create_docx():
    try:
        import docx
        doc = docx.Document()
        doc.add_heading("ALEX RIVERA", level=0)
        doc.add_paragraph("San Francisco, CA | (555) 234-5678 | alex.rivera@email.com")
        doc.add_heading("Professional Summary", level=1)
        doc.add_paragraph(
            "Senior Machine Learning Engineer with 6+ years of experience designing and deploying "
            "scalable deep learning and LLM architectures in production. Proven track record leading "
            "LLM fine-tuning, RAG pipelines, and vector database integrations."
        )
        doc.add_heading("Technical Skills", level=1)
        doc.add_paragraph("Python, PyTorch, LangChain, Qdrant, Docker, AWS, FastAPI, MLflow")
        doc.add_heading("Experience", level=1)
        p = doc.add_paragraph()
        p.add_run("Senior Machine Learning Engineer - NeuralScale AI (2022 - Present)\n").bold = True
        doc.add_paragraph("• Architected enterprise RAG pipeline querying 15M+ documents using Qdrant vector store.")
        doc.add_paragraph("• Optimized LLM inference serving using vLLM on AWS EC2 G5 clusters, improving throughput to 55 req/s.")
        
        doc_path = SAMPLE_DIR / "sample_resume.docx"
        doc.save(str(doc_path))
        print(f"Generated DOCX at {doc_path}")
    except ImportError:
        print("python-docx not yet installed, skipping docx generation.")

if __name__ == "__main__":
    create_docx()
