"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_analyze_text_endpoint():
    payload = {
        "resume_text": "Alex Rivera\nSenior Machine Learning Engineer\nPython, PyTorch, RAG",
        "jd_text": "Senior ML Engineer at Synthetix AI. Requirements: Python, PyTorch, RAG, AWS.",
        "tone": "confident",
    }
    response = client.post("/api/analyze/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["parsed_resume"] is not None
    assert data["parsed_jd"] is not None
    assert data["match_breakdown"] is not None
    assert data["match_breakdown"]["overall_score"] >= 0
    assert data["cover_letter"] is not None
    assert data["interview_prep"] is not None


def test_analyze_file_endpoint():
    file_content = b"Candidate Name\nSenior Software Engineer\nPython, FastAPI, Docker, AWS"
    files = {"resume_file": ("resume.txt", file_content, "text/plain")}
    data = {"jd_text": "Senior Python Backend Developer with FastAPI and Docker", "tone": "formal"}

    response = client.post("/api/analyze/file", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["parsed_resume"] is not None
    assert res_json["match_breakdown"] is not None


def test_application_tracker_api_crud():
    # 1. Create application
    app_data = {
        "company": "API Test Corp",
        "role": "Staff AI Engineer",
        "match_score": 91.5,
        "status": "Applied",
        "date_applied": "2026-11-21",
        "notes": "Submitted via internal referral",
    }
    create_resp = client.post("/api/applications", json=app_data)
    assert create_resp.status_code == 201
    created = create_resp.json()
    app_id = created["id"]
    assert app_id is not None
    assert created["company"] == "API Test Corp"

    # 2. Get single application
    get_resp = client.get(f"/api/applications/{app_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["role"] == "Staff AI Engineer"

    # 3. Update application
    patch_resp = client.patch(
        f"/api/applications/{app_id}",
        json={"status": "Interviewing", "notes": "Completed recruiter screen"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "Interviewing"

    # 4. List applications
    list_resp = client.get("/api/applications")
    assert list_resp.status_code == 200
    apps = list_resp.json()
    assert any(a["id"] == app_id for a in apps)

    # 5. Stats endpoint
    stats_resp = client.get("/api/applications/stats/summary")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_applications"] >= 1

    # 6. Delete application
    del_resp = client.delete(f"/api/applications/{app_id}")
    assert del_resp.status_code == 200

    # 7. Confirm 404 after deletion
    get_after_del = client.get(f"/api/applications/{app_id}")
    assert get_after_del.status_code == 404
