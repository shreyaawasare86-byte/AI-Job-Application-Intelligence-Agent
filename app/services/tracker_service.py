"""SQLite storage service for tracking job applications."""
import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.core.config import settings
from app.core.schemas import ApplicationRecord

logger = logging.getLogger(__name__)


class ApplicationTrackerService:
    """Manages persistent SQLite storage for job application lifecycle."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Create database tables and indices if not already present."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company TEXT NOT NULL,
                    role TEXT NOT NULL,
                    match_score REAL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'Applied',
                    date_applied TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    resume_filename TEXT,
                    job_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_apps_status ON applications(status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_apps_company ON applications(company);"
            )
            conn.commit()

    def add_application(self, record: ApplicationRecord) -> ApplicationRecord:
        """Insert a new job application record."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO applications (company, role, match_score, status, date_applied, notes, resume_filename, job_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.company,
                    record.role,
                    record.match_score,
                    record.status,
                    record.date_applied,
                    record.notes or "",
                    record.resume_filename,
                    record.job_url,
                ),
            )
            conn.commit()
            record.id = cursor.lastrowid
            return record

    def get_application(self, app_id: int) -> Optional[ApplicationRecord]:
        """Fetch an application by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM applications WHERE id = ?", (app_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return ApplicationRecord(**dict(row))

    def list_applications(self, status_filter: Optional[str] = None) -> List[ApplicationRecord]:
        """List all applications, optionally filtered by status."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if status_filter:
                cursor.execute(
                    "SELECT * FROM applications WHERE status = ? ORDER BY id DESC",
                    (status_filter,),
                )
            else:
                cursor.execute("SELECT * FROM applications ORDER BY id DESC")
            rows = cursor.fetchall()
            return [ApplicationRecord(**dict(r)) for r in rows]

    def update_application(
        self,
        app_id: int,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        match_score: Optional[float] = None,
    ) -> Optional[ApplicationRecord]:
        """Update status, notes, or score of an existing application."""
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        if match_score is not None:
            updates.append("match_score = ?")
            params.append(match_score)

        if not updates:
            return self.get_application(app_id)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(app_id)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE applications SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

        return self.get_application(app_id)

    def delete_application(self, app_id: int) -> bool:
        """Delete an application by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM applications WHERE id = ?", (app_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Compute aggregate stats for dashboard metrics."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(match_score) FROM applications")
            total_apps, avg_score = cursor.fetchone()

            cursor.execute("SELECT status, COUNT(*) FROM applications GROUP BY status")
            status_counts = dict(cursor.fetchall())

            return {
                "total_applications": total_apps or 0,
                "average_match_score": round(avg_score, 1) if avg_score else 0.0,
                "status_breakdown": status_counts,
            }


tracker_service = ApplicationTrackerService()
