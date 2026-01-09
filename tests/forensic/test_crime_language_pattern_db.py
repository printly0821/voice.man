"""
범죄 언어 패턴 데이터베이스 테스트
SPEC-FORENSIC-001 Phase 2-A: Crime Language Pattern Database

TDD RED Phase: CrimeLanguagePatternDB 클래스 테스트
"""

import pytest
from pathlib import Path


class TestCrimeLanguagePatternDB:
    """범죄 언어 패턴 데이터베이스 테스트"""

    def test_pattern_db_initialization(self):
        """패턴 DB 초기화 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()
        assert db is not None

    def test_load_gaslighting_patterns(self):
        """가스라이팅 패턴 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import GaslightingType

        db = CrimeLanguagePatternDB()
        patterns = db.get_gaslighting_patterns()

        assert len(patterns) == 7  # 7 types of gaslighting patterns
        assert GaslightingType.DENIAL in [p.type for p in patterns]
        assert GaslightingType.BLAME_SHIFTING in [p.type for p in patterns]

    def test_load_threat_patterns(self):
        """협박 패턴 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import ThreatType

        db = CrimeLanguagePatternDB()
        patterns = db.get_threat_patterns()

        assert len(patterns) == 5  # 5 types of threat patterns
        assert ThreatType.DIRECT in [p.type for p in patterns]
        assert ThreatType.ECONOMIC in [p.type for p in patterns]

    def test_load_coercion_patterns(self):
        """강압 패턴 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import CoercionType

        db = CrimeLanguagePatternDB()
        patterns = db.get_coercion_patterns()

        assert len(patterns) == 3  # 3 types of coercion patterns
        assert CoercionType.EMOTIONAL in [p.type for p in patterns]
        assert CoercionType.ISOLATION in [p.type for p in patterns]

    def test_load_deception_markers(self):
        """기만 지표 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import DeceptionCategory

        db = CrimeLanguagePatternDB()
        markers = db.get_deception_markers()

        assert len(markers) == 5  # 5 categories of deception markers
        assert DeceptionCategory.HEDGING in [m.category for m in markers]
        assert DeceptionCategory.DISTANCING in [m.category for m in markers]

    def test_get_patterns_by_gaslighting_type(self):
        """가스라이팅 유형별 패턴 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import GaslightingType

        db = CrimeLanguagePatternDB()
        denial_patterns = db.get_patterns_by_gaslighting_type(GaslightingType.DENIAL)

        assert len(denial_patterns) >= 5
        assert "그런 적 없어" in denial_patterns

    def test_get_patterns_by_threat_type(self):
        """협박 유형별 패턴 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import ThreatType

        db = CrimeLanguagePatternDB()
        direct_patterns = db.get_patterns_by_threat_type(ThreatType.DIRECT)

        assert len(direct_patterns) >= 5
        assert "가만 안 둬" in direct_patterns

    def test_get_patterns_by_coercion_type(self):
        """강압 유형별 패턴 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import CoercionType

        db = CrimeLanguagePatternDB()
        isolation_patterns = db.get_patterns_by_coercion_type(CoercionType.ISOLATION)

        assert len(isolation_patterns) >= 5
        assert "그 친구 만나지 마" in isolation_patterns

    def test_get_markers_by_deception_category(self):
        """기만 카테고리별 지표 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import DeceptionCategory

        db = CrimeLanguagePatternDB()
        hedging_markers = db.get_markers_by_deception_category(DeceptionCategory.HEDGING)

        assert len(hedging_markers) >= 5
        assert "아마" in hedging_markers

    def test_get_all_gaslighting_pattern_strings(self):
        """모든 가스라이팅 패턴 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()
        all_patterns = db.get_all_gaslighting_pattern_strings()

        # At least 7 types * 5 patterns = 35 patterns
        assert len(all_patterns) >= 35

    def test_get_all_threat_pattern_strings(self):
        """모든 협박 패턴 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()
        all_patterns = db.get_all_threat_pattern_strings()

        # At least 5 types * 5 patterns = 25 patterns
        assert len(all_patterns) >= 25

    def test_get_severity_weight(self):
        """패턴 유형별 심각도 가중치 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import (
            GaslightingType,
            ThreatType,
        )

        db = CrimeLanguagePatternDB()

        # Gaslighting severity weights
        denial_weight = db.get_gaslighting_severity_weight(GaslightingType.DENIAL)
        assert 0.0 <= denial_weight <= 1.0
        assert denial_weight == 0.8

        # Threat severity weights
        direct_weight = db.get_threat_severity_weight(ThreatType.DIRECT)
        assert direct_weight == 1.0

    def test_singleton_pattern(self):
        """싱글톤 패턴 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db1 = CrimeLanguagePatternDB()
        db2 = CrimeLanguagePatternDB()

        assert db1 is db2

    def test_pattern_db_version_info(self):
        """패턴 DB 버전 정보 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()
        version_info = db.get_version_info()

        assert "gaslighting" in version_info
        assert "threat" in version_info
        assert "coercion" in version_info
        assert "deception" in version_info
        assert version_info["gaslighting"]["version"] == "1.0.0"

    def test_reload_patterns(self):
        """패턴 리로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()
        initial_count = len(db.get_all_gaslighting_pattern_strings())

        # Reload should not raise errors
        db.reload_patterns()

        reloaded_count = len(db.get_all_gaslighting_pattern_strings())
        assert initial_count == reloaded_count

    def test_get_all_coercion_pattern_strings(self):
        """모든 강압 패턴 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()
        all_patterns = db.get_all_coercion_pattern_strings()

        # At least 3 types * 5 patterns = 15 patterns
        assert len(all_patterns) >= 15

    def test_get_all_deception_marker_strings(self):
        """모든 기만 지표 문자열 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()
        all_markers = db.get_all_deception_marker_strings()

        # At least 5 categories * 5 markers = 25 markers
        assert len(all_markers) >= 25

    def test_get_coercion_severity_weight(self):
        """강압 유형별 심각도 가중치 조회 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import CoercionType

        db = CrimeLanguagePatternDB()
        isolation_weight = db.get_coercion_severity_weight(CoercionType.ISOLATION)

        assert 0.0 <= isolation_weight <= 1.0
        assert isolation_weight == 0.85

    def test_get_patterns_by_nonexistent_type(self):
        """존재하지 않는 유형 조회 시 빈 리스트 반환 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import GaslightingType

        db = CrimeLanguagePatternDB()

        # Test that getting a valid type works
        patterns = db.get_patterns_by_gaslighting_type(GaslightingType.DENIAL)
        assert len(patterns) > 0


class TestCrimeLanguagePatternDBWithMissingFiles:
    """패턴 DB 파일 누락 시 테스트"""

    def test_graceful_handling_of_missing_file(self, tmp_path):
        """파일 누락 시 우아한 처리 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        # Reset singleton for testing
        CrimeLanguagePatternDB._instance = None

        # This should not raise an exception even if files are missing
        # The DB should use default patterns
        try:
            db = CrimeLanguagePatternDB()
            patterns = db.get_gaslighting_patterns()
            assert patterns is not None
        finally:
            # Reset singleton after test
            CrimeLanguagePatternDB._instance = None


class TestCrimeLanguagePatternDBDefaultPatterns:
    """기본 패턴 로드 테스트"""

    def test_default_gaslighting_patterns_loaded_when_file_missing(self, tmp_path, monkeypatch):
        """파일이 없을 때 기본 가스라이팅 패턴 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import GaslightingType

        # Reset singleton
        CrimeLanguagePatternDB._instance = None

        try:
            # Patch the data directory to a non-existent path
            db = CrimeLanguagePatternDB()
            original_dir = db._data_dir
            db._data_dir = tmp_path / "nonexistent"
            db._gaslighting_patterns = []
            db._version_info = {}

            # Trigger default loading
            db._load_default_gaslighting_patterns()

            # Verify default patterns are loaded
            patterns = db._gaslighting_patterns
            assert len(patterns) == 7  # 7 default gaslighting types

            # Check that all types are present
            types_in_patterns = [p.type for p in patterns]
            assert GaslightingType.DENIAL in types_in_patterns
            assert GaslightingType.BLAME_SHIFTING in types_in_patterns

            # Restore original
            db._data_dir = original_dir
        finally:
            CrimeLanguagePatternDB._instance = None

    def test_default_threat_patterns_loaded_when_file_missing(self, tmp_path):
        """파일이 없을 때 기본 협박 패턴 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import ThreatType

        # Reset singleton
        CrimeLanguagePatternDB._instance = None

        try:
            db = CrimeLanguagePatternDB()
            db._threat_patterns = []
            db._version_info = {}

            # Trigger default loading
            db._load_default_threat_patterns()

            # Verify default patterns are loaded
            patterns = db._threat_patterns
            assert len(patterns) == 5  # 5 default threat types

            types_in_patterns = [p.type for p in patterns]
            assert ThreatType.DIRECT in types_in_patterns
            assert ThreatType.ECONOMIC in types_in_patterns
        finally:
            CrimeLanguagePatternDB._instance = None

    def test_default_coercion_patterns_loaded_when_file_missing(self, tmp_path):
        """파일이 없을 때 기본 강압 패턴 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import CoercionType

        # Reset singleton
        CrimeLanguagePatternDB._instance = None

        try:
            db = CrimeLanguagePatternDB()
            db._coercion_patterns = []
            db._version_info = {}

            # Trigger default loading
            db._load_default_coercion_patterns()

            # Verify default patterns are loaded
            patterns = db._coercion_patterns
            assert len(patterns) == 3  # 3 default coercion types

            types_in_patterns = [p.type for p in patterns]
            assert CoercionType.EMOTIONAL in types_in_patterns
            assert CoercionType.ISOLATION in types_in_patterns
        finally:
            CrimeLanguagePatternDB._instance = None

    def test_default_deception_markers_loaded_when_file_missing(self, tmp_path):
        """파일이 없을 때 기본 기만 지표 로드 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )
        from voice_man.models.forensic.crime_language import DeceptionCategory

        # Reset singleton
        CrimeLanguagePatternDB._instance = None

        try:
            db = CrimeLanguagePatternDB()
            db._deception_markers = []
            db._version_info = {}

            # Trigger default loading
            db._load_default_deception_markers()

            # Verify default markers are loaded
            markers = db._deception_markers
            assert len(markers) == 5  # 5 default deception categories

            categories_in_markers = [m.category for m in markers]
            assert DeceptionCategory.HEDGING in categories_in_markers
            assert DeceptionCategory.DISTANCING in categories_in_markers
        finally:
            CrimeLanguagePatternDB._instance = None

    def test_default_severity_weight_for_unknown_type(self):
        """알 수 없는 유형의 기본 심각도 가중치 테스트"""
        from voice_man.services.forensic.crime_language_pattern_db import (
            CrimeLanguagePatternDB,
        )

        db = CrimeLanguagePatternDB()

        # Clear patterns to test default return
        original_patterns = db._gaslighting_patterns
        db._gaslighting_patterns = []

        try:
            # Should return default weight (0.5) when no pattern found
            from voice_man.models.forensic.crime_language import GaslightingType

            weight = db.get_gaslighting_severity_weight(GaslightingType.DENIAL)
            assert weight == 0.5
        finally:
            db._gaslighting_patterns = original_patterns
