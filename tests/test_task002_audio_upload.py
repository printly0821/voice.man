"""
TASK-002: 오디오 파일 업로드 엔드포인트 테스트

Acceptance Criteria:
- 정상 업로드: 유효한 mp3 파일 → 200 상태, 파일 ID 반환, SHA-256 해시 생성
- 잘못된 형식 거부: txt 파일 → 400 상태, "지원하지 않는 파일 형식입니다"
"""

import hashlib
from io import BytesIO
from fastapi.testclient import TestClient
from voice_man.main import app


class TestAudioUploadEndpoint:
    """오디오 파일 업로드 엔드포인트 테스트"""

    def test_upload_endpoint_exists(self):
        """POST /api/v1/audio/upload 엔드포인트가 존재해야 함"""
        routes = [route.path for route in app.routes]
        assert "/api/v1/audio/upload" in routes, "업로드 엔드포인트가 존재하지 않습니다"

    def test_upload_valid_mp3_file_returns_200(self):
        """유효한 MP3 파일 업로드 시 200 상태와 파일 ID를 반환해야 함"""
        client = TestClient(app)

        # Create a fake MP3 file (just use some binary data)
        mp3_content = b"ID3" + b"\x00" * 100  # Minimal MP3 header

        files = {"file": ("test_audio.mp3", BytesIO(mp3_content), "audio/mpeg")}
        response = client.post("/api/v1/audio/upload", files=files)

        assert response.status_code == 200, (
            f"예상 상태코드 200, 받은 상태코드: {response.status_code}"
        )

        data = response.json()
        assert "file_id" in data, "응답에 file_id 필드가 없습니다"
        assert isinstance(data["file_id"], str), "file_id는 문자열이어야 합니다"
        assert len(data["file_id"]) > 0, "file_id가 비어있습니다"

    def test_upload_valid_wav_file_returns_200(self):
        """유효한 WAV 파일 업로드 시 200 상태를 반환해야 함"""
        client = TestClient(app)

        # Create a fake WAV file
        wav_content = b"RIFF" + b"\x00" * 100  # Minimal WAV header

        files = {"file": ("test_audio.wav", BytesIO(wav_content), "audio/wav")}
        response = client.post("/api/v1/audio/upload", files=files)

        assert response.status_code == 200, (
            f"예상 상태코드 200, 받은 상태코드: {response.status_code}"
        )

        data = response.json()
        assert "file_id" in data, "응답에 file_id 필드가 없습니다"

    def test_upload_txt_file_returns_400(self):
        """TXT 파일 업로드 시 400 상태와 에러 메시지를 반환해야 함"""
        client = TestClient(app)

        txt_content = b"This is not an audio file"

        files = {"file": ("test.txt", BytesIO(txt_content), "text/plain")}
        response = client.post("/api/v1/audio/upload", files=files)

        assert response.status_code == 400, (
            f"예상 상태코드 400, 받은 상태코드: {response.status_code}"
        )

        data = response.json()
        assert "detail" in data, "응답에 detail 필드가 없습니다"
        assert "지원하지 않는 파일 형식입니다" in data["detail"], (
            f"예상 에러 메시지가 아닙니다: {data['detail']}"
        )

    def test_upload_jpg_file_returns_400(self):
        """JPG 이미지 파일 업로드 시 400 상태를 반환해야 함"""
        client = TestClient(app)

        jpg_content = b"\xff\xd8\xff\xe0"  # JPEG header

        files = {"file": ("test.jpg", BytesIO(jpg_content), "image/jpeg")}
        response = client.post("/api/v1/audio/upload", files=files)

        assert response.status_code == 400, (
            f"예상 상태코드 400, 받은 상태코드: {response.status_code}"
        )

    def test_upload_creates_sha256_hash(self):
        """파일 업로드 시 SHA-256 해시를 생성해야 함"""
        client = TestClient(app)

        test_content = b"Test audio content for hash generation"
        expected_hash = hashlib.sha256(test_content).hexdigest()

        files = {"file": ("test.mp3", BytesIO(test_content), "audio/mpeg")}
        response = client.post("/api/v1/audio/upload", files=files)

        data = response.json()
        assert "sha256_hash" in data, "응답에 sha256_hash 필드가 없습니다"
        assert data["sha256_hash"] == expected_hash, (
            f"예상 해시: {expected_hash}, 받은 해시: {data['sha256_hash']}"
        )

    def test_upload_includes_file_metadata(self):
        """파일 업로드 응답에 파일 메타데이터가 포함되어야 함"""
        client = TestClient(app)

        mp3_content = b"ID3" + b"\x00" * 100

        files = {"file": ("test_audio.mp3", BytesIO(mp3_content), "audio/mpeg")}
        response = client.post("/api/v1/audio/upload", files=files)

        data = response.json()
        assert "filename" in data, "응답에 filename 필드가 없습니다"
        assert data["filename"] == "test_audio.mp3", (
            f"예상 파일명: test_audio.mp3, 받은 파일명: {data['filename']}"
        )
        assert "content_type" in data, "응답에 content_type 필드가 없습니다"
        assert "file_size" in data, "응답에 file_size 필드가 없습니다"

    def test_upload_without_file_returns_422(self):
        """파일 없이 업로드 요청 시 422 상태를 반환해야 함"""
        client = TestClient(app)

        response = client.post("/api/v1/audio/upload")

        assert response.status_code == 422, (
            f"예상 상태코드 422, 받은 상태코드: {response.status_code}"
        )
