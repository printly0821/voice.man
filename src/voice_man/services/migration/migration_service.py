"""
Migration Service for Forensic Evidence Data

Handles data migration from JSON files to SQLite database.
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional, Callable, Any
import numpy as np

from voice_man.services.migration.schema_manager import SchemaManager


class MigrationService:
    """Service for migrating forensic data to SQLite database"""

    # Required fields for forensic data
    REQUIRED_FIELDS = [
        "audio_file_id",
        "file_path",
        "duration",
        "speakers",
        "transcript",
        "forensic_analysis",
    ]

    def __init__(self, db_path: str):
        """Initialize migration service

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.schema_manager = SchemaManager(db_path)

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection

        Returns:
            SQLite connection
        """
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
        self.schema_manager.close()

    def initialize_schema(self):
        """Initialize database schema"""
        self.schema_manager.create_tables()
        self.schema_manager.create_fts5_index()
        self.schema_manager.create_vector_tables()

    def _validate_forensic_data(self, data: Dict[str, Any]) -> None:
        """Validate forensic data has required fields

        Args:
            data: Forensic data dictionary

        Raises:
            ValueError: If required fields are missing
        """
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required field(s): {', '.join(missing_fields)}")

    def migrate_file(self, data: Dict[str, Any]) -> str:
        """Migrate single file data to database

        Args:
            data: Forensic data dictionary

        Returns:
            Audio file ID

        Raises:
            ValueError: If required fields are missing
        """
        self._validate_forensic_data(data)

        conn = self.get_connection()
        cursor = conn.cursor()

        audio_file_id = data["audio_file_id"]

        # Insert or update audio file
        cursor.execute(
            """
            INSERT INTO audio_files (id, file_path, duration, speakers)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                file_path = excluded.file_path,
                duration = excluded.duration,
                speakers = excluded.speakers,
                updated_at = CURRENT_TIMESTAMP
        """,
            (
                audio_file_id,
                data["file_path"],
                data["duration"],
                json.dumps(data["speakers"]),
            ),
        )

        # Delete existing transcripts for this file
        cursor.execute("DELETE FROM transcripts WHERE audio_file_id = ?", (audio_file_id,))

        # Insert transcripts
        for segment in data["transcript"]:
            cursor.execute(
                """
                INSERT INTO transcripts (audio_file_id, start_time, end_time, text, speaker)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    audio_file_id,
                    segment["start"],
                    segment["end"],
                    segment["text"],
                    segment["speaker"],
                ),
            )

        # Insert or update forensic results
        forensic = data["forensic_analysis"]
        cursor.execute(
            """
            INSERT INTO forensic_results (
                audio_file_id, gaslighting_score, threat_score,
                coercion_score, deception_score, stress_level, emotion_profile
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(audio_file_id) DO UPDATE SET
                gaslighting_score = excluded.gaslighting_score,
                threat_score = excluded.threat_score,
                coercion_score = excluded.coercion_score,
                deception_score = excluded.deception_score,
                stress_level = excluded.stress_level,
                emotion_profile = excluded.emotion_profile
        """,
            (
                audio_file_id,
                forensic["gaslighting_score"],
                forensic["threat_score"],
                forensic["coercion_score"],
                forensic["deception_score"],
                forensic["stress_level"],
                json.dumps(forensic["emotion_profile"]),
            ),
        )

        # Delete existing events
        cursor.execute("DELETE FROM gaslighting_events WHERE audio_file_id = ?", (audio_file_id,))
        cursor.execute("DELETE FROM emotion_events WHERE audio_file_id = ?", (audio_file_id,))

        # Insert gaslighting events
        for event in data.get("gaslighting_events", []):
            cursor.execute(
                """
                INSERT INTO gaslighting_events (
                    audio_file_id, start_time, end_time, pattern, confidence
                )
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    audio_file_id,
                    event["start"],
                    event["end"],
                    event["pattern"],
                    event["confidence"],
                ),
            )

        # Insert emotion events
        for event in data.get("emotion_events", []):
            cursor.execute(
                """
                INSERT INTO emotion_events (
                    audio_file_id, start_time, end_time, emotion, confidence
                )
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    audio_file_id,
                    event["start"],
                    event["end"],
                    event["emotion"],
                    event["confidence"],
                ),
            )

        conn.commit()
        return audio_file_id

    def migrate_batch(
        self,
        files_data: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Migrate batch of files to database

        Args:
            files_data: List of forensic data dictionaries
            progress_callback: Optional callback function(current, total, file_id)

        Returns:
            List of migration results with status
        """
        results = []
        total = len(files_data)

        for i, data in enumerate(files_data, 1):
            try:
                file_id = self.migrate_file(data)
                results.append({"file_id": file_id, "success": True})

                if progress_callback:
                    progress_callback(i, total, file_id)

            except Exception as e:
                results.append(
                    {
                        "file_id": data.get("audio_file_id", "unknown"),
                        "success": False,
                        "error": str(e),
                    }
                )

                if progress_callback:
                    progress_callback(i, total, data.get("audio_file_id", "unknown"))

        return results

    def rebuild_fts5_index(self):
        """Rebuild FTS5 full-text search index"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Rebuild index using INSERT with automatic replace
        cursor.execute(
            """
            INSERT INTO transcripts_fts(rowid, text, speaker, audio_file_id)
            SELECT rowid, text, speaker, audio_file_id FROM transcripts
        """
        )

        conn.commit()

    def search_transcripts(self, query: str) -> List[Dict[str, Any]]:
        """Search transcripts using FTS5

        Args:
            query: Search query

        Returns:
            List of matching transcript segments
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Add wildcards for partial matching
        search_query = f"{query}*"

        cursor.execute(
            """
            SELECT t.text, t.speaker, t.audio_file_id, t.start_time, t.end_time
            FROM transcripts_fts fts
            JOIN transcripts t ON t.rowid = fts.rowid
            WHERE transcripts_fts MATCH ?
            ORDER BY t.audio_file_id, t.start_time
        """,
            (search_query,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "text": row["text"],
                    "speaker": row["speaker"],
                    "audio_file_id": row["audio_file_id"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                }
            )

        return results

    def save_embedding(self, embedding_data: Dict[str, Any]) -> None:
        """Save vector embedding to database

        Args:
            embedding_data: Dictionary with audio_file_id, embedding (numpy array), and model
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        audio_file_id = embedding_data["audio_file_id"]
        embedding = embedding_data["embedding"]

        # Convert numpy array to bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()

        cursor.execute(
            """
            INSERT INTO embeddings (audio_file_id, embedding, model)
            VALUES (?, ?, ?)
            ON CONFLICT(audio_file_id) DO UPDATE SET
                embedding = excluded.embedding,
                model = excluded.model
        """,
            (audio_file_id, embedding_bytes, embedding_data["model"]),
        )

        conn.commit()

    def get_statistics(self) -> Dict[str, int]:
        """Get migration statistics

        Returns:
            Dictionary with various counts
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        stats = {}

        # Count audio files
        cursor.execute("SELECT COUNT(*) FROM audio_files")
        stats["total_files"] = cursor.fetchone()[0]

        # Count transcripts
        cursor.execute("SELECT COUNT(*) FROM transcripts")
        stats["total_transcripts"] = cursor.fetchone()[0]

        # Count gaslighting events
        cursor.execute("SELECT COUNT(*) FROM gaslighting_events")
        stats["total_gaslighting_events"] = cursor.fetchone()[0]

        # Count emotion events
        cursor.execute("SELECT COUNT(*) FROM emotion_events")
        stats["total_emotion_events"] = cursor.fetchone()[0]

        # Count embeddings
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        stats["total_embeddings"] = cursor.fetchone()[0]

        return stats

    def cleanup_and_vacuum(self):
        """Cleanup database and vacuum to reclaim space"""
        conn = self.get_connection()
        conn.execute("VACUUM")
        conn.execute("ANALYZE")

    def export_to_json(self, export_path: str) -> None:
        """Export database to JSON file

        Args:
            export_path: Path to output JSON file
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        export_data = {}

        # Export audio files
        cursor.execute("SELECT * FROM audio_files")
        export_data["audio_files"] = [dict(row) for row in cursor.fetchall()]

        # Export transcripts
        cursor.execute("SELECT * FROM transcripts")
        export_data["transcripts"] = [dict(row) for row in cursor.fetchall()]

        # Export forensic results
        cursor.execute("SELECT * FROM forensic_results")
        export_data["forensic_results"] = [dict(row) for row in cursor.fetchall()]

        # Export gaslighting events
        cursor.execute("SELECT * FROM gaslighting_events")
        export_data["gaslighting_events"] = [dict(row) for row in cursor.fetchall()]

        # Export emotion events
        cursor.execute("SELECT * FROM emotion_events")
        export_data["emotion_events"] = [dict(row) for row in cursor.fetchall()]

        # Write to file
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def import_from_json(self, import_path: str) -> None:
        """Import data from JSON file to database

        Args:
            import_path: Path to input JSON file
        """
        with open(import_path, "r", encoding="utf-8") as f:
            import_data = json.load(f)

        conn = self.get_connection()
        cursor = conn.cursor()

        # Import audio files
        for row in import_data.get("audio_files", []):
            cursor.execute(
                """
                INSERT OR REPLACE INTO audio_files
                (id, file_path, duration, speakers, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    row["id"],
                    row["file_path"],
                    row["duration"],
                    row["speakers"],
                    row["created_at"],
                    row["updated_at"],
                ),
            )

        # Import transcripts
        for row in import_data.get("transcripts", []):
            cursor.execute(
                """
                INSERT OR REPLACE INTO transcripts
                (id, audio_file_id, start_time, end_time, text, speaker)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    row["id"],
                    row["audio_file_id"],
                    row["start_time"],
                    row["end_time"],
                    row["text"],
                    row["speaker"],
                ),
            )

        # Import forensic results
        for row in import_data.get("forensic_results", []):
            cursor.execute(
                """
                INSERT OR REPLACE INTO forensic_results
                (id, audio_file_id, gaslighting_score, threat_score,
                 coercion_score, deception_score, stress_level, emotion_profile, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    row["id"],
                    row["audio_file_id"],
                    row["gaslighting_score"],
                    row["threat_score"],
                    row["coercion_score"],
                    row["deception_score"],
                    row["stress_level"],
                    row["emotion_profile"],
                    row["created_at"],
                ),
            )

        # Import gaslighting events
        for row in import_data.get("gaslighting_events", []):
            cursor.execute(
                """
                INSERT OR REPLACE INTO gaslighting_events
                (id, audio_file_id, start_time, end_time, pattern, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    row["id"],
                    row["audio_file_id"],
                    row["start_time"],
                    row["end_time"],
                    row["pattern"],
                    row["confidence"],
                ),
            )

        # Import emotion events
        for row in import_data.get("emotion_events", []):
            cursor.execute(
                """
                INSERT OR REPLACE INTO emotion_events
                (id, audio_file_id, start_time, end_time, emotion, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    row["id"],
                    row["audio_file_id"],
                    row["start_time"],
                    row["end_time"],
                    row["emotion"],
                    row["confidence"],
                ),
            )

        conn.commit()

        # Rebuild FTS5 index
        self.rebuild_fts5_index()
