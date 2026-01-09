"""
범죄 언어 패턴 데이터베이스
SPEC-FORENSIC-001 Phase 2-A: Crime Language Pattern Database

JSON 패턴 파일을 로드하고 관리하는 싱글톤 클래스
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from voice_man.models.forensic.crime_language import (
    GaslightingPattern,
    GaslightingType,
    ThreatPattern,
    ThreatType,
    CoercionPattern,
    CoercionType,
    DeceptionMarker,
    DeceptionCategory,
)


class CrimeLanguagePatternDB:
    """
    범죄 언어 패턴 데이터베이스

    JSON 파일에서 패턴을 로드하고 제공하는 싱글톤 클래스
    """

    _instance: Optional["CrimeLanguagePatternDB"] = None

    def __new__(cls) -> "CrimeLanguagePatternDB":
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """패턴 데이터베이스 초기화"""
        if self._initialized:
            return

        self._initialized = True
        self._data_dir = Path(__file__).parent.parent.parent / "data" / "forensic"

        # Pattern storage
        self._gaslighting_patterns: List[GaslightingPattern] = []
        self._threat_patterns: List[ThreatPattern] = []
        self._coercion_patterns: List[CoercionPattern] = []
        self._deception_markers: List[DeceptionMarker] = []

        # Version info storage
        self._version_info: Dict[str, Dict] = {}

        # Load patterns
        self._load_all_patterns()

    def _load_all_patterns(self) -> None:
        """모든 패턴 파일 로드"""
        self._load_gaslighting_patterns()
        self._load_threat_patterns()
        self._load_coercion_patterns()
        self._load_deception_markers()

    def _load_gaslighting_patterns(self) -> None:
        """가스라이팅 패턴 로드"""
        file_path = self._data_dir / "gaslighting_patterns_ko.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._version_info["gaslighting"] = {
                "version": data.get("version", "unknown"),
                "description": data.get("description", ""),
                "source": data.get("source", ""),
                "last_updated": data.get("last_updated", ""),
            }

            for pattern_data in data.get("patterns", []):
                pattern = GaslightingPattern(
                    type=GaslightingType(pattern_data["type"]),
                    patterns_ko=pattern_data["patterns_ko"],
                    patterns_en=pattern_data.get("patterns_en", []),
                    severity_weight=pattern_data["severity_weight"],
                    description_ko=pattern_data["description_ko"],
                    description_en=pattern_data.get("description_en", ""),
                )
                self._gaslighting_patterns.append(pattern)
        except FileNotFoundError:
            # Use default patterns if file not found
            self._load_default_gaslighting_patterns()

    def _load_default_gaslighting_patterns(self) -> None:
        """기본 가스라이팅 패턴 로드 (파일 없을 경우)"""
        self._version_info["gaslighting"] = {
            "version": "1.0.0",
            "description": "Default gaslighting patterns",
            "source": "Built-in",
            "last_updated": "2026-01-09",
        }
        # Add minimal default patterns
        default_types = [
            (GaslightingType.DENIAL, ["그런 적 없어"], 0.8, "부정 패턴"),
            (GaslightingType.COUNTERING, ["네 기억이 틀린 거야"], 0.7, "반박 패턴"),
            (GaslightingType.TRIVIALIZING, ["별거 아닌데"], 0.6, "최소화 패턴"),
            (GaslightingType.DIVERTING, ["그게 지금 중요해?"], 0.5, "화제전환 패턴"),
            (GaslightingType.BLOCKING, ["더 이상 말하지 마"], 0.7, "차단 패턴"),
            (GaslightingType.BLAME_SHIFTING, ["네 탓이야"], 0.9, "책임전가 패턴"),
            (GaslightingType.REALITY_DISTORTION, ["다 널 위해서야"], 0.85, "현실왜곡 패턴"),
        ]
        for type_, patterns, weight, desc in default_types:
            self._gaslighting_patterns.append(
                GaslightingPattern(
                    type=type_,
                    patterns_ko=patterns,
                    severity_weight=weight,
                    description_ko=desc,
                )
            )

    def _load_threat_patterns(self) -> None:
        """협박 패턴 로드"""
        file_path = self._data_dir / "threat_patterns_ko.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._version_info["threat"] = {
                "version": data.get("version", "unknown"),
                "description": data.get("description", ""),
                "source": data.get("source", ""),
                "last_updated": data.get("last_updated", ""),
            }

            for pattern_data in data.get("patterns", []):
                pattern = ThreatPattern(
                    type=ThreatType(pattern_data["type"]),
                    patterns_ko=pattern_data["patterns_ko"],
                    patterns_en=pattern_data.get("patterns_en", []),
                    severity_weight=pattern_data["severity_weight"],
                    description_ko=pattern_data["description_ko"],
                    description_en=pattern_data.get("description_en", ""),
                )
                self._threat_patterns.append(pattern)
        except FileNotFoundError:
            self._load_default_threat_patterns()

    def _load_default_threat_patterns(self) -> None:
        """기본 협박 패턴 로드"""
        self._version_info["threat"] = {
            "version": "1.0.0",
            "description": "Default threat patterns",
            "source": "Built-in",
            "last_updated": "2026-01-09",
        }
        default_types = [
            (ThreatType.DIRECT, ["가만 안 둬"], 1.0, "직접적 협박"),
            (ThreatType.CONDITIONAL, ["만약 ~하면"], 0.9, "조건부 협박"),
            (ThreatType.VEILED, ["알지?"], 0.7, "암시적 협박"),
            (ThreatType.ECONOMIC, ["돈 한 푼 못 받아"], 0.8, "경제적 협박"),
            (ThreatType.SOCIAL, ["다 소문낼 거야"], 0.75, "사회적 협박"),
        ]
        for type_, patterns, weight, desc in default_types:
            self._threat_patterns.append(
                ThreatPattern(
                    type=type_,
                    patterns_ko=patterns,
                    severity_weight=weight,
                    description_ko=desc,
                )
            )

    def _load_coercion_patterns(self) -> None:
        """강압 패턴 로드"""
        file_path = self._data_dir / "coercion_patterns_ko.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._version_info["coercion"] = {
                "version": data.get("version", "unknown"),
                "description": data.get("description", ""),
                "source": data.get("source", ""),
                "last_updated": data.get("last_updated", ""),
            }

            for pattern_data in data.get("patterns", []):
                pattern = CoercionPattern(
                    type=CoercionType(pattern_data["type"]),
                    patterns_ko=pattern_data["patterns_ko"],
                    patterns_en=pattern_data.get("patterns_en", []),
                    severity_weight=pattern_data["severity_weight"],
                    description_ko=pattern_data["description_ko"],
                    description_en=pattern_data.get("description_en", ""),
                )
                self._coercion_patterns.append(pattern)
        except FileNotFoundError:
            self._load_default_coercion_patterns()

    def _load_default_coercion_patterns(self) -> None:
        """기본 강압 패턴 로드"""
        self._version_info["coercion"] = {
            "version": "1.0.0",
            "description": "Default coercion patterns",
            "source": "Built-in",
            "last_updated": "2026-01-09",
        }
        default_types = [
            (CoercionType.EMOTIONAL, ["나를 사랑하면"], 0.8, "감정적 강압"),
            (CoercionType.GUILT_INDUCTION, ["너 때문에 내가 이 꼴이야"], 0.75, "죄책감 유발"),
            (CoercionType.ISOLATION, ["그 친구 만나지 마"], 0.85, "고립화"),
        ]
        for type_, patterns, weight, desc in default_types:
            self._coercion_patterns.append(
                CoercionPattern(
                    type=type_,
                    patterns_ko=patterns,
                    severity_weight=weight,
                    description_ko=desc,
                )
            )

    def _load_deception_markers(self) -> None:
        """기만 지표 로드"""
        file_path = self._data_dir / "deception_markers_ko.json"
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._version_info["deception"] = {
                "version": data.get("version", "unknown"),
                "description": data.get("description", ""),
                "source": data.get("source", ""),
                "last_updated": data.get("last_updated", ""),
            }

            for marker_data in data.get("markers", []):
                marker = DeceptionMarker(
                    category=DeceptionCategory(marker_data["category"]),
                    markers_ko=marker_data["markers_ko"],
                    markers_en=marker_data.get("markers_en", []),
                    description_ko=marker_data["description_ko"],
                    description_en=marker_data.get("description_en", ""),
                )
                self._deception_markers.append(marker)
        except FileNotFoundError:
            self._load_default_deception_markers()

    def _load_default_deception_markers(self) -> None:
        """기본 기만 지표 로드"""
        self._version_info["deception"] = {
            "version": "1.0.0",
            "description": "Default deception markers",
            "source": "Built-in",
            "last_updated": "2026-01-09",
        }
        default_markers = [
            (DeceptionCategory.HEDGING, ["아마"], "회피어"),
            (DeceptionCategory.DISTANCING, ["그 사람"], "거리두기"),
            (DeceptionCategory.NEGATIVE_EMOTION, ["싫어"], "부정 감정어"),
            (DeceptionCategory.EXCLUSIVE, ["절대"], "배타적 표현"),
            (DeceptionCategory.COGNITIVE_COMPLEXITY, ["왜냐하면"], "인지 복잡성"),
        ]
        for category, markers, desc in default_markers:
            self._deception_markers.append(
                DeceptionMarker(
                    category=category,
                    markers_ko=markers,
                    description_ko=desc,
                )
            )

    # ==========================================================================
    # Public API - Pattern Access
    # ==========================================================================

    def get_gaslighting_patterns(self) -> List[GaslightingPattern]:
        """모든 가스라이팅 패턴 반환"""
        return self._gaslighting_patterns

    def get_threat_patterns(self) -> List[ThreatPattern]:
        """모든 협박 패턴 반환"""
        return self._threat_patterns

    def get_coercion_patterns(self) -> List[CoercionPattern]:
        """모든 강압 패턴 반환"""
        return self._coercion_patterns

    def get_deception_markers(self) -> List[DeceptionMarker]:
        """모든 기만 지표 반환"""
        return self._deception_markers

    # ==========================================================================
    # Public API - Pattern Strings by Type
    # ==========================================================================

    def get_patterns_by_gaslighting_type(self, gaslighting_type: GaslightingType) -> List[str]:
        """가스라이팅 유형별 패턴 문자열 반환"""
        for pattern in self._gaslighting_patterns:
            if pattern.type == gaslighting_type:
                return pattern.patterns_ko
        return []

    def get_patterns_by_threat_type(self, threat_type: ThreatType) -> List[str]:
        """협박 유형별 패턴 문자열 반환"""
        for pattern in self._threat_patterns:
            if pattern.type == threat_type:
                return pattern.patterns_ko
        return []

    def get_patterns_by_coercion_type(self, coercion_type: CoercionType) -> List[str]:
        """강압 유형별 패턴 문자열 반환"""
        for pattern in self._coercion_patterns:
            if pattern.type == coercion_type:
                return pattern.patterns_ko
        return []

    def get_markers_by_deception_category(self, category: DeceptionCategory) -> List[str]:
        """기만 카테고리별 지표 문자열 반환"""
        for marker in self._deception_markers:
            if marker.category == category:
                return marker.markers_ko
        return []

    # ==========================================================================
    # Public API - All Pattern Strings
    # ==========================================================================

    def get_all_gaslighting_pattern_strings(self) -> List[str]:
        """모든 가스라이팅 패턴 문자열 반환"""
        all_patterns = []
        for pattern in self._gaslighting_patterns:
            all_patterns.extend(pattern.patterns_ko)
        return all_patterns

    def get_all_threat_pattern_strings(self) -> List[str]:
        """모든 협박 패턴 문자열 반환"""
        all_patterns = []
        for pattern in self._threat_patterns:
            all_patterns.extend(pattern.patterns_ko)
        return all_patterns

    def get_all_coercion_pattern_strings(self) -> List[str]:
        """모든 강압 패턴 문자열 반환"""
        all_patterns = []
        for pattern in self._coercion_patterns:
            all_patterns.extend(pattern.patterns_ko)
        return all_patterns

    def get_all_deception_marker_strings(self) -> List[str]:
        """모든 기만 지표 문자열 반환"""
        all_markers = []
        for marker in self._deception_markers:
            all_markers.extend(marker.markers_ko)
        return all_markers

    # ==========================================================================
    # Public API - Severity Weights
    # ==========================================================================

    def get_gaslighting_severity_weight(self, gaslighting_type: GaslightingType) -> float:
        """가스라이팅 유형별 심각도 가중치 반환"""
        for pattern in self._gaslighting_patterns:
            if pattern.type == gaslighting_type:
                return pattern.severity_weight
        return 0.5  # Default weight

    def get_threat_severity_weight(self, threat_type: ThreatType) -> float:
        """협박 유형별 심각도 가중치 반환"""
        for pattern in self._threat_patterns:
            if pattern.type == threat_type:
                return pattern.severity_weight
        return 0.5

    def get_coercion_severity_weight(self, coercion_type: CoercionType) -> float:
        """강압 유형별 심각도 가중치 반환"""
        for pattern in self._coercion_patterns:
            if pattern.type == coercion_type:
                return pattern.severity_weight
        return 0.5

    # ==========================================================================
    # Public API - Metadata
    # ==========================================================================

    def get_version_info(self) -> Dict[str, Dict]:
        """버전 정보 반환"""
        return self._version_info

    def reload_patterns(self) -> None:
        """패턴 다시 로드"""
        self._gaslighting_patterns = []
        self._threat_patterns = []
        self._coercion_patterns = []
        self._deception_markers = []
        self._version_info = {}
        self._load_all_patterns()
