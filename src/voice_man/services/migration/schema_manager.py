"""
Schema Manager for Forensic Database

Manages database schema creation and initialization.
"""

import sqlite3
from typing import List


class SchemaManager:
    """Manages database schema for forensic evidence data"""

    def __init__(self, db_path: str):
        """Initialize schema manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None

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

    def get_tables(self) -> List[str]:
        """Get list of all tables

        Returns:
            List of table names
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def create_tables(self):
        """Create all forensic data tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Audio files table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_files (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                duration REAL NOT NULL,
                speakers TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Transcripts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                text TEXT NOT NULL,
                speaker TEXT NOT NULL,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """
        )

        # Forensic results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forensic_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id TEXT NOT NULL UNIQUE,
                gaslighting_score REAL NOT NULL,
                threat_score REAL NOT NULL,
                coercion_score REAL NOT NULL,
                deception_score REAL NOT NULL,
                stress_level INTEGER NOT NULL,
                emotion_profile TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """
        )

        # Gaslighting events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS gaslighting_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                pattern TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """
        )

        # Emotion events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emotion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """
        )

        conn.commit()

    def create_fts5_index(self):
        """Create FTS5 full-text search index for transcripts"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Drop existing FTS5 table if exists
        cursor.execute("DROP TABLE IF EXISTS transcripts_fts")

        # Create FTS5 virtual table
        cursor.execute(
            """
            CREATE VIRTUAL TABLE transcripts_fts USING fts5(
                text,
                speaker,
                audio_file_id,
                content=transcripts,
                content_rowid=rowid,
                tokenize='unicode61 remove_diacritics 0'
            )
        """
        )

        # Populate FTS5 index
        cursor.execute(
            """
            INSERT INTO transcripts_fts(rowid, text, speaker, audio_file_id)
            SELECT rowid, text, speaker, audio_file_id FROM transcripts
        """
        )

        conn.commit()

    def create_vector_tables(self):
        """Create tables for vector embeddings"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Embeddings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id TEXT NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """
        )

        # Vector index table (metadata for FAISS)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_file_id TEXT NOT NULL UNIQUE,
                vector_id INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
            )
        """
        )

        conn.commit()
