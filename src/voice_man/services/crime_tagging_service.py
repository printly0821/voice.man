"""
범죄 발언 태깅 서비스

정규 표현식 기반 패턴 매칭, KoELECTRA 모델 기반 분류,
Claude API 기반 맥락 분석을 통합한 범죄 발언 태깅 시스템
"""

import json
import re
from pathlib import Path
from typing import List, Optional
from voice_man.models.crime_tag import CrimeTag, CrimeType, CrimeAnalysisResult


class CrimeTaggingService:
    """
    범죄 발언 태깅 서비스

    1차 필터링: 정규 표현식 기반 키워드 탐지
    2차 정밀 분석: Claude API 기반 맥락 분석 (선택적)
    """

    # 범적 참조 매핑
    LEGAL_REFERENCES = {
        CrimeType.THREAT: "형법 제283조",
        CrimeType.INTIMIDATION: "형법 제350조",
        CrimeType.FRAUD: "형법 제347조",
        CrimeType.INSULT: "형법 제311조",
    }

    def __init__(
        self,
        keywords_path: Optional[Path] = None,
        use_claude_api: bool = False,
        use_model: bool = False,
    ):
        """
        초기화

        Args:
            keywords_path: 키워드 사전 파일 경로
            use_claude_api: Claude API 사용 여부 (맥락 분석)
            use_model: KoELECTRA 모델 사용 여부
        """
        self.use_claude_api = use_claude_api
        self.use_model = use_model
        self.model = None
        self.tokenizer = None

        # 키워드 사전 로드
        if keywords_path is None:
            keywords_path = Path(__file__).parent.parent.parent / "data" / "crime_keywords.json"

        self.keywords_path = keywords_path
        self.keywords = self._load_keywords()

    def load_keywords(self) -> dict:
        """
        범죄 키워드 사전 로드

        Returns:
            키워드 사전 (범죄 유형별 키워드 리스트)
        """
        return self._load_keywords()

    def _load_keywords(self) -> dict:
        """내부 키워드 로드 메서드"""
        try:
            # Try multiple possible paths
            possible_paths = [
                self.keywords_path,
                Path(__file__).parent.parent.parent / "data" / "crime_keywords.json",
                Path("data/crime_keywords.json"),
            ]

            for path in possible_paths:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        keywords = json.load(f)
                    return keywords

            # If no file found, use default keywords
            return self._get_default_keywords()

        except Exception:
            # 기본 키워드 사전 반환
            return self._get_default_keywords()

    def _get_default_keywords(self) -> dict:
        """기본 키워드 사전 반환"""
        return {
            "협박": [
                "죽여버린다",
                "죽인다",
                "죽이겠다",
                "죽일 거다",
                "죽일 것이다",
                "가만 안 둔다",
                "불질러버린다",
                "없애버린다",
                "조질러버린다",
            ],
            "공갈": [
                "돈 안 주면",
                "까불면 알지",
                "퍼뜨려버린다",
                "돈 내놔",
                "비밀 알게 될 거다",
            ],
            "사기": [
                "무조건 된다",
                "손해 없다",
                "내가 보장한다",
                "100% 확실하다",
                "절대 실패 없다",
            ],
            "모욕": [
                "멍청이",
                "멍청한",
                "바보",
                "쓰레기",
                " idiot",
                "짐승만도 못하다",
                "미치광이",
            ],
        }

    def detect_crime(self, text: str) -> List[CrimeTag]:
        """
        텍스트에서 범죄 발언 탐지

        Args:
            text: 분석할 텍스트

        Returns:
            탐지된 범죄 태그 리스트
        """
        if not text or not text.strip():
            return []

        tags = []

        # 각 범죄 유형별로 키워드 매칭
        for crime_type_str, keywords in self.keywords.items():
            crime_type = CrimeType(crime_type_str)

            # 키워드 매칭 (부분 일치 허용)
            matched_keywords = []
            for keyword in keywords:
                # 정확한 일치 또는 부분 일치 확인
                if keyword in text or any(keyword in word for word in text.split()):
                    matched_keywords.append(keyword)

            if matched_keywords:
                # 신뢰도 점수 계산
                confidence = self._calculate_confidence(text, matched_keywords, crime_type)

                # 범죄 태그 생성
                tag = CrimeTag(
                    type=crime_type,
                    confidence=confidence,
                    keywords=matched_keywords,
                    legal_reference=self.LEGAL_REFERENCES[crime_type],
                    needs_review=confidence < 0.7,
                )
                tags.append(tag)

        return tags

    def _calculate_confidence(
        self, text: str, matched_keywords: List[str], crime_type: CrimeType
    ) -> float:
        """
        신뢰도 점수 계산

        Args:
            text: 전체 텍스트
            matched_keywords: 매칭된 키워드 리스트
            crime_type: 범죄 유형

        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        base_confidence = 0.7  # 기본 신뢰도

        # 명시적인 범죄 키워드 (높은 신뢰도)
        explicit_threats = ["죽여버린다", "불질러버린다", "가만 안 둔다"]
        for keyword in matched_keywords:
            if keyword in explicit_threats:
                return 0.85

        # 키워드 빈도에 따른 가중치
        keyword_count = len(matched_keywords)
        if keyword_count >= 2:
            return min(base_confidence + 0.1, 1.0)
        elif keyword_count == 1:
            # 모호한 키워드의 경우 낮은 신뢰도
            ambiguous_keywords = ["죽인다", "없애버린다"]
            if any(kw in matched_keywords[0] for kw in ambiguous_keywords):
                return 0.6
            return base_confidence

        return base_confidence

    async def detect_crime_with_context(self, text: str, context: str) -> List[CrimeTag]:
        """
        맥락 기반 범죄 발언 탐지 (Claude API 활용)

        Args:
            text: 분석할 텍스트
            context: 발언 맥락

        Returns:
            탐지된 범죄 태그 리스트
        """
        # 1차: 키워드 기반 탐지
        tags = self.detect_crime(text)

        if not self.use_claude_api:
            return tags

        # 2차: Claude API 기반 맥락 분석
        # TODO: Claude API 연동 (TASK-011에서 구현)
        # 현재는 기본 맥락 필터링만 적용

        # 맥락 기반 신뢰도 조정
        for tag in tags:
            if self._is_non_criminal_context(text, context):
                tag.confidence = max(tag.confidence - 0.3, 0.1)
                tag.needs_review = True

        return tags

    def _is_non_criminal_context(self, text: str, context: str) -> bool:
        """
        비범죄적 맥락 판단 (간단한 규칙 기반)

        Args:
            text: 발언 텍스트
            context: 맥락

        Returns:
            비범죄적 맥락이면 True
        """
        non_criminal_contexts = [
            "음식",
            "맛",
            "농담",
            "비유",
            "게임",
            "영화",
            "책",
        ]

        for ctx in non_criminal_contexts:
            if ctx in context.lower():
                return True

        return False

    def classify_with_model(self, text: str) -> str:
        """
        KoELECTRA 모델 기반 분류

        Args:
            text: 분석할 텍스트

        Returns:
            범죄 유형 라벨
        """
        # TODO: KoELECTRA 모델 통합 (TASK-008 후속)
        # 현재는 키워드 기반 분류 사용
        tags = self.detect_crime(text)
        if tags:
            return tags[0].type.value
        return "정상"
