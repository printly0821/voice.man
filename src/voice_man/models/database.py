"""
데이터베이스 ORM 모델

SQLAlchemy를 사용한 데이터베이스 모델 정의입니다.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator

from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
    sessionmaker,
)


class Base(DeclarativeBase):
    """SQLAlchemy Base 클래스"""

    pass


# 데이터베이스 파일 경로
DB_PATH = Path(__file__).parent.parent.parent / "voice_man.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# 엔진 생성
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite에서 필요
    echo=False,  # 개발 시 True로 설정하여 SQL 로그 확인 가능
)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    데이터베이스 세션 의존성 주입 함수

    FastAPI Depends와 함께 사용됩니다.

    Yields:
        Session: SQLAlchemy 세션
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AudioFile(Base):
    """
    오디오 파일 모델

    업로드된 오디오 파일의 메타데이터를 저장합니다.
    """

    __tablename__ = "audio_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    file_path: Mapped[str] = mapped_column(String(512))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(50), default="uploaded")

    # 관계
    transcripts: Mapped[list["Transcript"]] = relationship(
        "Transcript", back_populates="audio_file", cascade="all, delete-orphan"
    )


class Transcript(Base):
    """
    텍스트 변환 결과 모델

    STT 변환 결과를 저장합니다.
    """

    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_id: Mapped[str] = mapped_column(String(36), ForeignKey("audio_files.id"))
    version: Mapped[str] = mapped_column(String(50), default="v1")
    content: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # 관계
    audio_file: Mapped["AudioFile"] = relationship("AudioFile", back_populates="transcripts")
    segments: Mapped[list["TranscriptSegment"]] = relationship(
        "TranscriptSegment", back_populates="transcript", cascade="all, delete-orphan"
    )


class TranscriptSegment(Base):
    """
    텍스트 변환 세그먼트 모델

    개별 텍스트 세그먼트와 타임스탬프를 저장합니다.
    """

    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transcript_id: Mapped[int] = mapped_column(Integer, ForeignKey("transcripts.id"))
    speaker_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)

    # 관계
    transcript: Mapped["Transcript"] = relationship("Transcript", back_populates="segments")
