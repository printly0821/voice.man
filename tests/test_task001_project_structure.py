"""
TASK-001: 프로젝트 구조 초기화 테스트

Acceptance Criteria:
- pyproject.toml이 올바른 의존성으로 생성됨
- pip install -e . 또는 uv pip install -e . 실행 성공
- FastAPI 앱이 정상적으로 로드됨
- /health 엔드포인트 응답 성공
"""

import pytest
from pathlib import Path
import subprocess
import sys


class TestProjectStructure:
    """프로젝트 구조 초기화 테스트"""

    def test_pyproject_toml_exists(self):
        """pyproject.toml 파일이 존재해야 함"""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        assert pyproject_path.exists(), "pyproject.toml 파일이 존재하지 않습니다"

    def test_pyproject_toml_has_required_dependencies(self):
        """pyproject.toml에 필요한 의존성이 포함되어야 함"""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        import tomllib

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        dependencies = config.get("project", {}).get("dependencies", [])
        dep_names = [
            dep.split("==")[0].split(">=")[0].split(">")[0].split("<")[0] for dep in dependencies
        ]

        required_deps = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "sqlalchemy",
            "alembic",
        ]

        for dep in required_deps:
            assert any(dep in d for d in dep_names), f"필수 의존성 {dep}가 누락되었습니다"

    def test_src_directory_exists(self):
        """src 디렉토리가 존재해야 함"""
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"

        assert src_dir.exists(), "src 디렉토리가 존재하지 않습니다"
        assert src_dir.is_dir(), "src가 디렉토리가 아닙니다"

    def test_main_module_exists(self):
        """메인 모듈 파일이 존재해야 함"""
        project_root = Path(__file__).parent.parent
        main_module = project_root / "src" / "voice_man" / "__init__.py"
        app_module = project_root / "src" / "voice_man" / "main.py"

        assert main_module.exists(), "src/voice_man/__init__.py가 존재하지 않습니다"
        assert app_module.exists(), "src/voice_man/main.py가 존재하지 않습니다"

    def test_tests_directory_exists(self):
        """tests 디렉토리가 존재해야 함"""
        project_root = Path(__file__).parent.parent
        tests_dir = project_root / "tests"

        assert tests_dir.exists(), "tests 디렉토리가 존재하지 않합니다"
        assert tests_dir.is_dir(), "tests가 디렉토리가 아닙니다"

    def test_fastapi_app_can_be_imported(self):
        """FastAPI 앱을 임포트할 수 있어야 함"""
        try:
            from voice_man.main import app

            assert app is not None, "FastAPI 앱이 None입니다"
        except ImportError as e:
            pytest.fail(f"FastAPI 앱을 임포트할 수 없습니다: {e}")

    def test_fastapi_app_is_fastapi_instance(self):
        """앱이 FastAPI 인스턴스여야 함"""
        from fastapi import FastAPI
        from voice_man.main import app

        assert isinstance(app, FastAPI), "앱이 FastAPI 인스턴스가 아닙니다"

    def test_health_endpoint_exists(self):
        """/health 엔드포인트가 존재해야 함"""
        from voice_man.main import app

        routes = [route.path for route in app.routes]
        assert "/health" in routes, "/health 엔드포인트가 존재하지 않습니다"

    def test_health_endpoint_returns_200(self):
        """/health 엔드포인트가 200 상태를 반환해야 함"""
        from fastapi.testclient import TestClient
        from voice_man.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200, (
            f"/health 엔드포인트가 {response.status_code}를 반환했습니다"
        )

    def test_health_endpoint_returns_correct_response(self):
        """/health 엔드포인트가 올바른 응답을 반환해야 함"""
        from fastapi.testclient import TestClient
        from voice_man.main import app

        client = TestClient(app)
        response = client.get("/health")

        data = response.json()
        assert "status" in data, "응답에 status 필드가 없습니다"
        assert data["status"] == "healthy", f"status가 'healthy'가 아닙니다: {data.get('status')}"

    def test_package_installable(self):
        """패키지가 설치 가능해야 함 (설치 검증)"""
        project_root = Path(__file__).parent.parent

        # 이미 설치된 경우 건너뜀
        try:
            import voice_man

            # 이미 설치됨
            return
        except ImportError:
            pass

        # 설치 시도
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(project_root)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # 설치 실패는 테스트 실패로 간주하지만, 이미 설치된 경우는 OK
        if result.returncode != 0 and "Successfully installed" not in result.stdout:
            if "Requirement already satisfied" not in result.stdout:
                pytest.fail(f"패키지 설치 실패: {result.stderr}")
