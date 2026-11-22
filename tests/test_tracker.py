"""Tests for SQLite Job Application Tracker service."""
import pytest
import tempfile
from pathlib import Path
from app.core.schemas import ApplicationRecord
from app.services.tracker_service import ApplicationTrackerService


@pytest.fixture
def temp_tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_applications.db"
        service = ApplicationTrackerService(db_path=db_path)
        yield service


def test_tracker_create_and_get(temp_tracker):
    record = ApplicationRecord(
        company="Synthetix AI",
        role="Senior ML Engineer",
        match_score=89.2,
        status="Applied",
        date_applied="2026-11-21",
        notes="First round interview next week",
        resume_filename="alex_rivera_resume.pdf",
    )
    created = temp_tracker.add_application(record)
    assert created.id is not None
    assert created.id > 0

    fetched = temp_tracker.get_application(created.id)
    assert fetched is not None
    assert fetched.company == "Synthetix AI"
    assert fetched.match_score == 89.2
    assert fetched.status == "Applied"


def test_tracker_list_and_filter(temp_tracker):
    app1 = ApplicationRecord(
        company="Company A",
        role="ML Engineer",
        match_score=90.0,
        status="Applied",
        date_applied="2026-11-20",
    )
    app2 = ApplicationRecord(
        company="Company B",
        role="Fullstack Dev",
        match_score=75.0,
        status="Interviewing",
        date_applied="2026-11-19",
    )
    temp_tracker.add_application(app1)
    temp_tracker.add_application(app2)

    all_apps = temp_tracker.list_applications()
    assert len(all_apps) == 2

    interviewing_apps = temp_tracker.list_applications(status_filter="Interviewing")
    assert len(interviewing_apps) == 1
    assert interviewing_apps[0].company == "Company B"


def test_tracker_update_and_delete(temp_tracker):
    record = ApplicationRecord(
        company="Stripe",
        role="Staff Engineer",
        match_score=85.0,
        status="Applied",
        date_applied="2026-11-21",
    )
    created = temp_tracker.add_application(record)

    updated = temp_tracker.update_application(
        created.id, status="Offer", notes="Received competitive offer!"
    )
    assert updated.status == "Offer"
    assert "offer" in updated.notes

    deleted = temp_tracker.delete_application(created.id)
    assert deleted is True
    assert temp_tracker.get_application(created.id) is None


def test_tracker_analytics_summary(temp_tracker):
    temp_tracker.add_application(
        ApplicationRecord(
            company="Google",
            role="Research Scientist",
            match_score=95.0,
            status="Offer",
            date_applied="2026-11-01",
        )
    )
    temp_tracker.add_application(
        ApplicationRecord(
            company="Meta",
            role="AI Engineer",
            match_score=85.0,
            status="Interviewing",
            date_applied="2026-11-05",
        )
    )

    analytics = temp_tracker.get_analytics_summary()
    assert analytics["total_applications"] == 2
    assert analytics["average_match_score"] == 90.0
    assert analytics["status_breakdown"].get("Offer") == 1
