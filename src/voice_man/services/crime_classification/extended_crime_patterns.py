"""
Extended Crime Patterns Service
SPEC-CRIME-CLASS-001 Phase 2: Extended Korean Language Patterns (11 Crime Types)

Defines text patterns for all 11 crime types:
사기(Fraud), 공갈(Extortion), 강요(Coercion), 협박(Threat), 모욕(Insult),
횡령(Embezzlement), 배임(Breach of Trust), 조세포탈(Tax Evasion),
가스라이팅(Gaslighting), 스토킹(Stalking), 가정폭력(Domestic Violence)
"""

import re
from typing import Dict, List, Tuple


class ExtendedCrimePatterns:
    """
    확장된 범죄 언어 패턴 데이터베이스

    11개 범죄 유형에 대한 한국어 패턴 매칭 및 점수 계산
    """

    # Multimodal weights for each crime type
    CRIME_WEIGHTS = {
        "가정폭력": {"text": 0.35, "audio": 0.40, "psych": 0.25},
        "스토킹": {"text": 0.40, "audio": 0.30, "psych": 0.30},
        "협박": {"text": 0.35, "audio": 0.40, "psych": 0.25},
        "가스라이팅": {"text": 0.40, "audio": 0.30, "psych": 0.30},
        "사기": {"text": 0.40, "audio": 0.35, "psych": 0.25},
        "공갈": {"text": 0.35, "audio": 0.40, "psych": 0.25},
        "강요": {"text": 0.35, "audio": 0.40, "psych": 0.25},
        "모욕": {"text": 0.45, "audio": 0.35, "psych": 0.20},
        "횡령": {"text": 0.50, "audio": 0.25, "psych": 0.25},
        "배임": {"text": 0.50, "audio": 0.25, "psych": 0.25},
        "조세포탈": {"text": 0.55, "audio": 0.20, "psych": 0.25},
    }

    def __init__(self) -> None:
        """Initialize extended crime patterns database"""
        self._patterns = self._load_patterns()

    def _load_patterns(self) -> Dict[str, Dict]:
        """
        Load all crime patterns

        Returns:
            Dictionary mapping crime type to pattern data
        """
        return {
            "사기": {
                "text_patterns": self._get_fraud_patterns(),
                "audio_indicators": ["fast_speech_rate", "persuasive_tone"],
                "psych_indicators": ["machiavellianism", "deception"],
            },
            "공갈": {
                "text_patterns": self._get_extortion_patterns(),
                "audio_indicators": ["aggressive_tone", "threatening_pitch"],
                "psych_indicators": ["aggression", "dominance"],
            },
            "강요": {
                "text_patterns": self._get_coercion_patterns(),
                "audio_indicators": ["commanding_tone", "high_intensity"],
                "psych_indicators": ["control", "dominance"],
            },
            "협박": {
                "text_patterns": self._get_threat_patterns(),
                "audio_indicators": ["low_pitch", "threatening_tone"],
                "psych_indicators": ["aggression", "intimidation"],
            },
            "모욕": {
                "text_patterns": self._get_insult_patterns(),
                "audio_indicators": ["mocking_tone", "contempt"],
                "psych_indicators": ["narcissism", "superiority"],
            },
            "횡령": {
                "text_patterns": self._get_embezzlement_patterns(),
                "audio_indicators": ["evasive_tone", "uncertain_speech"],
                "psych_indicators": ["greed", "rationalization"],
            },
            "배임": {
                "text_patterns": self._get_breach_of_trust_patterns(),
                "audio_indicators": ["defensive_tone", "justifying"],
                "psych_indicators": ["opportunism", "self_rationalization"],
            },
            "조세포탈": {
                "text_patterns": self._get_tax_evasion_patterns(),
                "audio_indicators": ["low_volume", "hesitant_speech"],
                "psych_indicators": ["avoidance", "rationalization"],
            },
            "가정폭력": {
                "text_patterns": self._get_domestic_violence_patterns(),
                "audio_indicators": ["high_pitch", "aggressive_tone", "shouting"],
                "psych_indicators": ["control", "dominance", "anger"],
            },
            "스토킹": {
                "text_patterns": self._get_stalking_patterns(),
                "audio_indicators": ["obsessive_tone", "repetitive_patterns"],
                "psych_indicators": ["obsession", "paranoid_traits"],
            },
            "가스라이팅": {
                "text_patterns": self._get_gaslighting_patterns(),
                "audio_indicators": ["calm_tone", "persuasive_speech"],
                "psych_indicators": ["narcissism", "manipulation", "lack_of_empathy"],
            },
        }

    # ==========================================================================
    # Fraud Patterns (사기) - 30+ patterns
    # ==========================================================================

    def _get_fraud_patterns(self) -> List[str]:
        """Get fraud/deception patterns (사기죄)"""
        return [
            # Financial promises
            "돈 빌려주면 갚을게",
            "확실히 갚아줄게",
            "이번만 믿어줘",
            "원금 보장",
            "확실히 수익이 나",
            "투자하면 두 배로",
            "절대 손해 안 봐",
            "100% 수익 보장",
            "이자 확실히 드려",
            "돈만 있으면 해결",
            # Urgency and pressure
            "지금 투자해야",
            "마감 임박",
            "지금 결제해야",
            "오늘만 특가",
            "기회 한 번뿐",
            "지금 안 하면 후회",
            "빨리 결정해",
            # False authority/expertise
            "전문가라서",
            "내가 책임질게",
            "확실한 정보야",
            "내가 잘 알아",
            "전문적으로",
            # Vague promises
            "좋은 기회야",
            "잘 될 거야",
            "걱정 마",
            "내가 알아서",
            "믿어달라고",
            # Additional patterns
            "약속할게",
            "절대 아니야",
            "검증됐어",
            "확인해봐",
            "보증할게",
            "안심해",
            "문제없어",
            "100%확실",
            "내 말 맞아",
            "믿음 가져",
        ]

    # ==========================================================================
    # Extortion Patterns (공갈) - 25+ patterns
    # ==========================================================================

    def _get_extortion_patterns(self) -> List[str]:
        """Get extortion patterns (공갈죄)"""
        return [
            # Direct threats for money
            "돈 내놔",
            "금액 요구",
            "돈 안 주면",
            "돈 주면 안",
            "돈 필요해",
            "현금 가져와",
            # Threats combined with demands
            "안 주면 알지",
            "폭로할 거야",
            "알리면 안 되",
            "비밀 유지해",
            "돈 주면 말해줘",
            # Blackmail
            "약점 있잖아",
            "쓴소리 하지 마",
            "잘 알지",
            "네 잘못",
            "증거 있어",
            # Pressure tactics
            "지금 가져와",
            "빨리 처리",
            "마지막 기회",
            # Additional patterns
            "필요해",
            "내놔",
            "주세요",
            "금전 요구",
            "압박",
            "협박해서",
            "재산 요구",
            "금품 갈취",
            "공갈해서",
            "너 책임",
        ]

    # ==========================================================================
    # Coercion Patterns (강요) - 25+ patterns
    # ==========================================================================

    def _get_coercion_patterns(self) -> List[str]:
        """Get coercion patterns (강요죄)"""
        return [
            # Commands
            "해",
            "당장 해",
            "무조건 해",
            "거절 안 해",
            "필수야",
            "의무야",
            # Threats for compliance
            "안 하면 후회",
            "필셀 거야",
            "어쩔 수 없",
            "강요하는 거야",
            # Power display
            "내 말 안 들어",
            "무시해",
            "말 안 통해",
            "안 들려",
            # Rights violation
            "권리 없어",
            "할 수 있어",
            "네 탓",
            # Additional patterns
            "명령해",
            "지시해",
            "강제로",
            "억지로",
            "거부 못해",
            "무조건",
            "반드시",
            "당장",
            "필수",
            "의무",
            "약속해",
            "계약해",
            "서명해",
            "승인해",
        ]

    # ==========================================================================
    # Threat Patterns (협박) - 25+ patterns
    # ==========================================================================

    def _get_threat_patterns(self) -> List[str]:
        """Get threat patterns (협박죄)"""
        return [
            # Death threats
            "죽일 거야",
            "죽여버려",
            "죽일래",
            "목숨",
            "생명",
            # Physical harm threats
            "때려 죽이",
            "아프게 할",
            "다치게",
            "해칠 거야",
            # Future harm
            "나중에 알",
            "기회 있을 거야",
            "두고 보",
            "기억해",
            # Indirect threats
            "알지",
            "잘 생각",
            "후회",
            "조심",
            # Additional patterns
            "위협",
            "공포",
            "해악",
            "고지",
            "협박",
            "무섭지",
            "두려워",
            "신고하지 마",
            "말하지 마",
            "비밀",
            "알리면",
            "폭로",
            "피해",
            "해치",
        ]

    # ==========================================================================
    # Insult Patterns (모욕) - 30+ patterns
    # ==========================================================================

    def _get_insult_patterns(self) -> List[str]:
        """Get insult patterns (모욕죄)"""
        return [
            # Direct insults
            "멍청이",
            "바보",
            "등신",
            "傻子",  # Chinese (bachi)
            "백치",
            # Character attacks
            "성격 나빠",
            "인성 없어",
            "수준 낮아",
            "질이 나빠",
            # Capability attacks
            "못해",
            "무능",
            "능력 없",
            # Appearance attacks
            "못생겼",
            "추해",
            "보기 싫",
            # Social status attacks
            "신분 낮아",
            "수준 미달",
            # Additional patterns
            "쓰레기",
            "쓰레기 같",
            "짐승",
            "동물",
            "개같",
            "돼지",
            "벌레",
            "하등",
            "혼자",
            "불쌍",
            "가엾",
            "비웃",
            "조롱",
            "경멸",
            "무시",
            "하대",
            "비하",
            "천대",
            "무시",
        ]

    # ==========================================================================
    # Embezzlement Patterns (횡령) - 20+ patterns
    # ==========================================================================

    def _get_embezzlement_patterns(self) -> List[str]:
        """Get embezzlement patterns (횡령죄)"""
        return [
            # Misuse of funds
            "회사 돈으로",
            "법인카드로",
            "경비 처리",
            "세금계산서 없이",
            "현금으로 받아",
            "장부에 안 올려",
            # Personal use
            "개인적으로",
            "내가 쓸",
            "필요해",
            # Hiding transactions
            "기록 안 해",
            "비밀",
            "알리지 마",
            # Additional patterns
            "착복",
            "횡령",
            "유용",
            "전용",
            "용도 외",
            "금품 착복",
            "공금 횡령",
            "회삼",
            "유용금",
            "개인 사용",
            "처분",
            "보관",
        ]

    # ==========================================================================
    # Breach of Trust Patterns (배임) - 20+ patterns
    # ==========================================================================

    def _get_breach_of_trust_patterns(self) -> List[str]:
        """Get breach of trust patterns (배임죄)"""
        return [
            # Trust violation
            "신임 위반",
            "믿음 저버려",
            "약속 어김",
            # Self-interest
            "내 이익",
            "내가 득",
            "나에게 유리",
            # Duty violation
            "의무 어겨",
            "책임 안 해",
            "할 일 안 해",
            # Additional patterns
            "배임",
            "위배",
            "의무 위반",
            "신의 위반",
            "직무 유기",
            "이익 얻",
            "제3자 이익",
            "본인 이익",
            "임무 위반",
            "직책 위반",
            "위법",
            "불법",
            "특이 이익",
        ]

    # ==========================================================================
    # Tax Evasion Patterns (조세포탈) - 15+ patterns
    # ==========================================================================

    def _get_tax_evasion_patterns(self) -> List[str]:
        """Get tax evasion patterns (조세포탈죄)"""
        return [
            # Tax avoidance language
            "세금 회피",
            "세금 안 내",
            "소득 숨겨",
            "신고 안 해",
            "허위 신고",
            "장부 조작",
            # Hiding income
            "수입 안 보여",
            "현금만",
            "기록 없",
            # Additional patterns
            "탈세",
            "포탈",
            "조세 회피",
            "소득 은닉",
            "허위 세금",
            "세금 신고",
            "과세 표준",
            "부가세",
            "소득세",
            "법인세",
            "세무",
            "세무사",
            "신고 omitted",
        ]

    # ==========================================================================
    # Domestic Violence Patterns (가정폭력) - 25+ patterns
    # ==========================================================================

    def _get_domestic_violence_patterns(self) -> List[str]:
        """Get domestic violence patterns (가정폭력)"""
        return [
            # Control
            "내 말 안 들어",
            "내가 시켜",
            "통제",
            "관리",
            # Isolation
            "만나지 마",
            "연락하지 마",
            "나가지 마",
            # Economic control
            "돈 줄게",
            "용돈",
            "경제적",
            # Behavior restriction
            "하지 마",
            "금지",
            "허용 없",
            # Aggression
            "때릴",
            "아프게",
            "해치",
            # Additional patterns
            "가정폭력",
            "구타",
            "상해",
            "감금",
            "협박",
            "명예훼손",
            "강요",
            "권리침해",
            "신체폭력",
            "언어폭력",
            "정서폭력",
            "경제폭력",
            "성폭력",
            "방임",
        ]

    # ==========================================================================
    # Stalking Patterns (스토킹) - 25+ patterns
    # ==========================================================================

    def _get_stalking_patterns(self) -> List[str]:
        """Get stalking patterns (스토킹죄)"""
        return [
            # Obsession
            "계속 생각",
            "잊을 수 없",
            "계속 신경",
            # Surveillance
            "어디",
            "누구 만나",
            "뭐 해",
            "어디 있",
            # Repeated contact
            "계속 연락",
            "계속 불러",
            "매일",
            "자꾸",
            # Following
            "따라가",
            "뒤따라",
            "미행",
            # Additional patterns
            "스토킹",
            "집착",
            "감시",
            "추적",
            "연락",
            "만남",
            "접근",
            "骚扰",
            "괴롭혀",
            "쫓아다녀",
            "지켜봐",
            "신경",
            "관심",
            "사랑",
            "애정",
            "호의",
            "집요",
        ]

    # ==========================================================================
    # Gaslighting Patterns (가스라이팅) - Uses existing patterns
    # ==========================================================================

    def _get_gaslighting_patterns(self) -> List[str]:
        """Get gaslighting patterns (가스라이팅)"""
        return [
            # Reality distortion
            "기억 못",
            "상상해",
            "잘못",
            "착각",
            # Memory denial
            "안 그랬",
            "못",
            "기억",
            # Emotional invalidation
            "예민해",
            "과민",
            "너무",
            # Self-esteem attacks
            "잘못",
            "문제",
            "네 탓",
            # Additional patterns
            "현실 왜곡",
            "기억 부정",
            "감정 무효",
            "자존감 공격",
            "이상해",
            "이상",
            "문제 있어",
            "네가 이상",
            "정신",
            "상담",
            "치료",
        ]

    # ==========================================================================
    # Public API Methods
    # ==========================================================================

    def get_patterns(self, crime_type: str) -> Dict:
        """
        Get patterns for a specific crime type

        Args:
            crime_type: Korean crime type name (e.g., "사기")

        Returns:
            Dictionary containing text_patterns, audio_indicators, psych_indicators
        """
        return self._patterns.get(crime_type, {})

    def match_text(self, crime_type: str, text: str) -> List[str]:
        """
        Match text against crime type patterns

        Args:
            crime_type: Korean crime type name
            text: Text to analyze

        Returns:
            List of matched patterns
        """
        pattern_data = self.get_patterns(crime_type)
        if not pattern_data:
            return []

        text_patterns = pattern_data.get("text_patterns", [])
        matches = []

        for pattern in text_patterns:
            if pattern in text:
                matches.append(pattern)

        return matches

    def calculate_crime_score(self, crime_type: str, text: str) -> float:
        """
        Calculate crime type score based on pattern matching

        Args:
            crime_type: Korean crime type name
            text: Text to analyze

        Returns:
            Score between 0.0 and 1.0
        """
        pattern_data = self.get_patterns(crime_type)
        if not pattern_data:
            return 0.0

        text_patterns = pattern_data.get("text_patterns", [])
        match_count = 0

        for pattern in text_patterns:
            if pattern in text:
                match_count += 1

        # Calculate score: matches / sqrt(total_patterns)
        # Using sqrt prevents saturation with many patterns
        import math

        max_score = 1.0
        if len(text_patterns) > 0:
            score = min(match_count / math.sqrt(len(text_patterns)), max_score)
        else:
            score = 0.0

        return min(score, 1.0)

    def get_crime_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Get multimodal weights for all crime types

        Returns:
            Dictionary mapping crime type to {text, audio, psych} weights
        """
        return self.CRIME_WEIGHTS.copy()
