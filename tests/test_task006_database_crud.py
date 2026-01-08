"""
TASK-006: 데이터베이스 스키마 및 모델 CRUD 테스트

테스트 목적:
- SQLAlchemy async 설정 검증
- 데이터 모델 관계 정의 검증
- CRUD 작업 기능 검증
- SQLite 및 PostgreSQL 호환성 검증
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from voice_man.models.database import AudioFile, Base, Transcript, TranscriptSegment


class TestDatabaseSetup:
    """데이터베이스 설정 테스트"""

    @pytest.mark.asyncio
    async def test_create_sqlite_engine(self):
        """
        Given: SQLite 데이터베이스 URL
        When: async engine 생성
        Then: engine이 정상적으로 생성됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        # When
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(db_url, echo=False)

        # Then
        assert engine is not None
        await engine.dispose()

    @pytest.mark.asyncio
    async def test_create_tables(self):
        """
        Given: 빈 데이터베이스
        When: 테이블 생성
        Then: 모든 테이블이 정상적으로 생성됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)

        # When
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Then
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            )
            tables = [row[0] for row in result.fetchall()]

            assert "audio_files" in tables
            assert "transcripts" in tables
            assert "transcript_segments" in tables

        await engine.dispose()


class TestAudioFileModel:
    """AudioFile 모델 테스트"""

    @pytest.mark.asyncio
    async def test_create_audio_file(self):
        """
        Given: 유효한 AudioFile 데이터
        When: 데이터베이스에 저장
        Then: AudioFile이 정상적으로 저장됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # When
        async with async_session() as session:
            audio_file = AudioFile(
                id="test-id-123",
                original_filename="test_audio.mp3",
                file_hash="a" * 64,  # SHA-256 hash
                file_path="/uploads/test-id-123.mp3",
                duration_seconds=120.5,
                status="uploaded",
            )
            session.add(audio_file)
            await session.commit()
            await session.refresh(audio_file)

        # Then
        async with async_session() as session:
            result = await session.get(AudioFile, "test-id-123")
            assert result is not None
            assert result.original_filename == "test_audio.mp3"
            assert result.file_hash == "a" * 64
            assert result.duration_seconds == 120.5
            assert result.status == "uploaded"

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_read_audio_file(self):
        """
        Given: 저장된 AudioFile
        When: 조회 요청
        Then: AudioFile이 정상적으로 반환됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        audio_file = AudioFile(
            id="read-test-id",
            original_filename="read_test.mp3",
            file_hash="b" * 64,
            file_path="/uploads/read-test-id.mp3",
            status="uploaded",
        )

        async with async_session() as session:
            session.add(audio_file)
            await session.commit()

        # When
        async with async_session() as session:
            result = await session.get(AudioFile, "read-test-id")

        # Then
        assert result is not None
        assert result.id == "read-test-id"
        assert result.original_filename == "read_test.mp3"

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_update_audio_file(self):
        """
        Given: 저장된 AudioFile
        When: 상태 업데이트
        Then: 상태가 정상적으로 변경됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        audio_file = AudioFile(
            id="update-test-id",
            original_filename="update_test.mp3",
            file_hash="c" * 64,
            file_path="/uploads/update-test-id.mp3",
            status="uploaded",
        )

        async with async_session() as session:
            session.add(audio_file)
            await session.commit()

        # When
        async with async_session() as session:
            result = await session.get(AudioFile, "update-test-id")
            result.status = "processing"
            await session.commit()

        # Then
        async with async_session() as session:
            result = await session.get(AudioFile, "update-test-id")
            assert result.status == "processing"

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_delete_audio_file(self):
        """
        Given: 저장된 AudioFile
        When: 삭제 요청
        Then: AudioFile이 정상적으로 삭제됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        audio_file = AudioFile(
            id="delete-test-id",
            original_filename="delete_test.mp3",
            file_hash="d" * 64,
            file_path="/uploads/delete-test-id.mp3",
            status="uploaded",
        )

        async with async_session() as session:
            session.add(audio_file)
            await session.commit()

        # When
        async with async_session() as session:
            result = await session.get(AudioFile, "delete-test-id")
            await session.delete(result)
            await session.commit()

        # Then
        async with async_session() as session:
            result = await session.get(AudioFile, "delete-test-id")
            assert result is None

        await engine.dispose()


class TestTranscriptModel:
    """Transcript 모델 테스트"""

    @pytest.mark.asyncio
    async def test_create_transcript_with_segments(self):
        """
        Given: AudioFile과 트랜스크립트 데이터
        When: 관계형 데이터 저장
        Then: Transcript와 TranscriptSegment가 정상적으로 저장됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # When
        transcript_id = None
        async with async_session() as session:
            # AudioFile 생성
            audio_file = AudioFile(
                id="transcript-test-id",
                original_filename="transcript_test.mp3",
                file_hash="e" * 64,
                file_path="/uploads/transcript-test-id.mp3",
                status="transcribed",
            )
            session.add(audio_file)
            await session.flush()

            # Transcript 생성
            transcript = Transcript(
                audio_id="transcript-test-id",
                version="v1",
                content="전체 텍스트 내용",
                language="ko",
            )
            session.add(transcript)
            await session.flush()
            transcript_id = transcript.id

            # TranscriptSegments 생성
            segment1 = TranscriptSegment(
                transcript_id=transcript.id,
                speaker_id="speaker_0",
                start_time=0.0,
                end_time=2.5,
                text="안녕하세요",
                confidence=0.95,
            )
            segment2 = TranscriptSegment(
                transcript_id=transcript.id,
                speaker_id="speaker_1",
                start_time=2.5,
                end_time=5.0,
                text="반갑습니다",
                confidence=0.92,
            )
            session.add(segment1)
            session.add(segment2)
            await session.commit()

        # Then
        async with async_session() as session:
            from sqlalchemy import select

            # AudioFile 확인
            result = await session.get(AudioFile, "transcript-test-id")
            assert result is not None

            # Transcript 확인 (명시적 쿼리)
            stmt = select(Transcript).where(Transcript.audio_id == "transcript-test-id")
            transcript_result = await session.execute(stmt)
            transcript = transcript_result.scalar_one()

            assert transcript.content == "전체 텍스트 내용"

            # Segments 확인
            stmt = select(TranscriptSegment).where(TranscriptSegment.transcript_id == transcript_id)
            segments_result = await session.execute(stmt)
            segments = segments_result.scalars().all()

            assert len(segments) == 2
            assert segments[0].text == "안녕하세요"
            assert segments[1].text == "반갑습니다"

        await engine.dispose()

    @pytest.mark.asyncio
    async def test_foreign_key_relationship(self):
        """
        Given: 관계형 데이터
        When: 외래키 관계 조회
        Then: 정확한 관계가 설정됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # When
        transcript_pk = None
        async with async_session() as session:
            audio_file = AudioFile(
                id="fk-test-id",
                original_filename="fk_test.mp3",
                file_hash="f" * 64,
                file_path="/uploads/fk-test-id.mp3",
                status="transcribed",
            )
            session.add(audio_file)
            await session.flush()

            transcript = Transcript(
                audio_id="fk-test-id",
                version="v1",
                content="FK 테스트",
                language="ko",
            )
            session.add(transcript)
            await session.flush()
            transcript_pk = transcript.id
            await session.commit()

        # Then
        async with async_session() as session:
            from sqlalchemy import select

            # AudioFile → Transcript 관계 확인
            audio = await session.get(AudioFile, "fk-test-id")
            assert audio is not None

            stmt = select(Transcript).where(Transcript.audio_id == "fk-test-id")
            transcript_result = await session.execute(stmt)
            transcript = transcript_result.scalar_one()

            assert transcript.audio_id == "fk-test-id"

            # Transcript → AudioFile 역관계 확인
            transcript2 = await session.get(Transcript, transcript_pk)
            assert transcript2.audio_id == "fk-test-id"

        await engine.dispose()


class TestDatabaseCompatibility:
    """데이터베이스 호환성 테스트"""

    @pytest.mark.asyncio
    async def test_sqlite_compatibility(self):
        """
        Given: SQLite 데이터베이스
        When: 모든 CRUD 작업 수행
        Then: 모든 작업이 정상적으로 완료됨
        """
        # Given
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_url = f"sqlite+aiosqlite:///{tmp.name}"

        engine = create_async_engine(db_url, echo=False)
        async_session = async_sessionmaker(engine, expire_on_commit=False)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # When - Create
        async with async_session() as session:
            audio = AudioFile(
                id="sqlite-test",
                original_filename="sqlite.mp3",
                file_hash="g" * 64,
                file_path="/uploads/sqlite.mp3",
                status="uploaded",
            )
            session.add(audio)
            await session.commit()

        # Then - Read
        async with async_session() as session:
            result = await session.get(AudioFile, "sqlite-test")
            assert result is not None

        # Update
        async with async_session() as session:
            result = await session.get(AudioFile, "sqlite-test")
            result.status = "completed"
            await session.commit()

        # Delete
        async with async_session() as session:
            result = await session.get(AudioFile, "sqlite-test")
            await session.delete(result)
            await session.commit()

        await engine.dispose()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="PostgreSQL 드라이버(asyncpg)는 선택적 의존성")
    async def test_postgresql_url_format(self):
        """
        Given: PostgreSQL 형식의 URL
        When: engine 생성 시도
        Then: PostgreSQL URL이 정상적으로 처리됨
        """
        # Given
        postgres_url = "postgresql+asyncpg://user:pass@localhost/testdb"

        # When
        engine = create_async_engine(postgres_url, echo=False)

        # Then
        assert engine is not None
        await engine.dispose()
