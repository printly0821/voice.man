"""
Legal Evidence Mapper Service
SPEC-CRIME-CLASS-001 Phase 5: Legal Evidence Mapping (Korean Criminal Code)
"""

import json
from pathlib import Path
from typing import Dict, List

from voice_man.models.crime_classification.legal_requirements import (
    LegalEvidenceMapping,
    LegalRequirement,
)


class LegalEvidenceMapper:
    """
    법적 증거 매핑 서비스

    Maps forensic analysis results to Korean Criminal Code requirements
    """

    def __init__(self) -> None:
        """Initialize legal evidence mapper"""
        self._legal_requirements = self._load_legal_requirements()

    def _load_legal_requirements(self) -> Dict[str, Dict]:
        """
        Load Korean Criminal Code requirements

        Returns:
            Dictionary mapping crime types to legal requirements
        """
        return {
            "사기": {
                "legal_code": "형법 제347조",
                "requirements": [
                    {
                        "name": "기망행위",
                        "description": "허위 사실을 진술하거나 사실을 은폐",
                        "indicators": ["과장", "허위진술", "사실왜곡", "거짓말"],
                        "required": True,
                    },
                    {
                        "name": "착오유발",
                        "description": "상대방의 착오를 유발",
                        "indicators": ["오해유도", "정보은닉", "기만"],
                        "required": True,
                    },
                    {
                        "name": "처분행위",
                        "description": "재산상 처분행위 유발",
                        "indicators": ["금전요구", "투자권유", "계약체결"],
                        "required": True,
                    },
                    {
                        "name": "재산상이익",
                        "description": "재산상 이익 취득",
                        "indicators": ["금전수령", "이익실현", "재산취득"],
                        "required": True,
                    },
                    {
                        "name": "인과관계",
                        "description": "기망-착오-처분-이익 간 인과관계",
                        "indicators": ["연속성", "시간적근접", "인과성"],
                        "required": True,
                    },
                ],
            },
            "협박": {
                "legal_code": "형법 제283조",
                "requirements": [
                    {
                        "name": "해악고지",
                        "description": "상대방에게 해악을 고지",
                        "indicators": ["위호", "해악", "피해고지"],
                        "required": True,
                    },
                    {
                        "name": "상대방존재",
                        "description": "상대방 또는 그 친족의 생명·신체·자유·명예·재산에 대한 해악",
                        "indicators": ["생명위호", "신체해악", "자유침해", "명예훼손", "재산피해"],
                        "required": True,
                    },
                    {
                        "name": "의사표시강요",
                        "description": "상대방으로 하여금 의사표시를 하게 할 목적",
                        "indicators": ["강요", "압박", "의사결정방해"],
                        "required": True,
                    },
                ],
            },
            "공갈": {
                "legal_code": "형법 제350조",
                "requirements": [
                    {
                        "name": "협박 또는 모략",
                        "description": "사람을 협박하거나 모략",
                        "indicators": ["협박", "모략", "위호"],
                        "required": True,
                    },
                    {
                        "name": "재물교사",
                        "description": "재물의 교사를 받거나 재산상 이익 취득",
                        "indicators": ["금품요구", "재산교사", "이익취득"],
                        "required": True,
                    },
                ],
            },
            "강요": {
                "legal_code": "형법 제324조",
                "requirements": [
                    {
                        "name": "폭행협박",
                        "description": "폭행 또는 협박",
                        "indicators": ["폭행", "협박", "위력행사"],
                        "required": True,
                    },
                    {
                        "name": "의무없는행위요구",
                        "description": "의무 없는 행위를 요구",
                        "indicators": ["강요", "강제", "명령"],
                        "required": True,
                    },
                    {
                        "name": "권리침해",
                        "description": "권리 행사 방해",
                        "indicators": ["권리침해", "의사결정방해", "행동방해"],
                        "required": True,
                    },
                ],
            },
            "모욕": {
                "legal_code": "형법 제311조",
                "requirements": [
                    {
                        "name": "공연한 사실",
                        "description": "공연히 사실을 적시하여",
                        "indicators": ["공연", "공개", "사실적시"],
                        "required": True,
                    },
                    {
                        "name": "모욕행위",
                        "description": "사람을 모욕",
                        "indicators": ["경멸", "비하", "천대", "조롱"],
                        "required": True,
                    },
                ],
            },
            "횡령": {
                "legal_code": "형법 제355조 제1항",
                "requirements": [
                    {
                        "name": "업무상점유",
                        "description": "업무상 타인의 재물을 점유",
                        "indicators": ["점유", "보관", "관리"],
                        "required": True,
                    },
                    {
                        "name": "횡령행위",
                        "description": "영득의사로 사용·처분",
                        "indicators": ["사용", "처분", "유용", "착복"],
                        "required": True,
                    },
                ],
            },
            "배임": {
                "legal_code": "형법 제355조 제2항",
                "requirements": [
                    {
                        "name": "업무상처분권한",
                        "description": "업무상 타인의 사무를 처리하는 자",
                        "indicators": ["사무처리", "대리", "위임"],
                        "required": True,
                    },
                    {
                        "name": "임무위배",
                        "description": "임무에 위배되는 행위",
                        "indicators": ["위배", "위법", "의무위반"],
                        "required": True,
                    },
                    {
                        "name": "재산상이익",
                        "description": "재산상 이익 취득 또는 제3자로 하여금 취득",
                        "indicators": ["이익", "재산상득", "제3자이익"],
                        "required": True,
                    },
                ],
            },
            "가스라이팅": {
                "legal_code": "형법 제283조(협박), 제311조(모욕)",
                "requirements": [
                    {
                        "name": "현실왜곡",
                        "description": "상대방의 현실 인식 왜곡",
                        "indicators": ["기억부정", "현실부정", "의심부여"],
                        "required": True,
                    },
                    {
                        "name": "감정무효화",
                        "description": "상대방 감정과 경험 무효화",
                        "indicators": ["과민", "예민", "정서부정"],
                        "required": True,
                    },
                ],
            },
            "가정폭력": {
                "legal_code": "가정폭력범죄의 처벌 등에 관한 특례법",
                "requirements": [
                    {
                        "name": "가정구성원",
                        "description": "가정구성원에 대한 행위",
                        "indicators": ["배우자", "가족", "동거"],
                        "required": True,
                    },
                    {
                        "name": "폭력행위",
                        "description": "신체·정신·재산적 피해",
                        "indicators": ["신체폭력", "언어폭력", "정서폭력", "경제폭력"],
                        "required": True,
                    },
                ],
            },
            "스토킹": {
                "legal_code": "스토킹 범죄의 처벌 등에 관한 법률",
                "requirements": [
                    {
                        "name": "지속적접근",
                        "description": "상대방의 의사에 반하여 지속적으로 접근",
                        "indicators": ["접근", "연락", "만남", "미행"],
                        "required": True,
                    },
                    {
                        "name": "공포심유발",
                        "description": "공포심이나 불안감 유발",
                        "indicators": ["공포", "불안", "두려움"],
                        "required": True,
                    },
                ],
            },
            "조세포탈": {
                "legal_code": "조세범처벌법",
                "requirements": [
                    {
                        "name": "사기기타부정행위",
                        "description": "사기 기타 부정한 행위",
                        "indicators": ["사기", "허위신고", "장부조작"],
                        "required": True,
                    },
                    {
                        "name": "조세포탈",
                        "description": "조세를 포탈하거나 환급받음",
                        "indicators": ["포탈", "환급", "징수회피"],
                        "required": True,
                    },
                ],
            },
        }

    def map_evidence(
        self, crime_type: str, detected_indicators: List[str], evidence_items: List[str]
    ) -> LegalEvidenceMapping:
        """
        Map detected indicators to legal requirements

        Args:
            crime_type: Korean crime type name
            detected_indicators: List of detected indicators from forensic analysis
            evidence_items: List of evidence items

        Returns:
            LegalEvidenceMapping object
        """
        if crime_type not in self._legal_requirements:
            # Return empty mapping for unknown crime types
            return LegalEvidenceMapping(
                crime_type=crime_type,
                legal_code="미정",
                requirements=[],
                fulfillment_rate=0.0,
                legal_viability="낮음",
            )

        crime_data = self._legal_requirements[crime_type]
        legal_code = crime_data["legal_code"]
        requirement_defs = crime_data["requirements"]

        requirements = []
        satisfied_count = 0

        for req_def in requirement_defs:
            # Check if any indicators are present
            matching_indicators = [
                ind for ind in req_def["indicators"] if ind in detected_indicators
            ]

            # Find supporting evidence
            supporting_evidence = [
                ev for ev in evidence_items if any(ind in ev for ind in matching_indicators)
            ]

            # Requirement is satisfied if it has matching indicators
            satisfied = len(matching_indicators) > 0
            if satisfied and req_def["required"]:
                satisfied_count += 1

            requirement = LegalRequirement(
                name=req_def["name"],
                description=req_def["description"],
                indicators=req_def["indicators"],
                satisfied=satisfied,
                evidence=supporting_evidence,
            )
            requirements.append(requirement)

        # Calculate fulfillment rate
        total_required = len([r for r in requirement_defs if r["required"]])

        if total_required > 0:
            fulfillment_rate = satisfied_count / total_required
        else:
            fulfillment_rate = 0.0

        # Determine legal viability
        if fulfillment_rate >= 0.8:
            legal_viability = "높음"
        elif fulfillment_rate >= 0.5:
            legal_viability = "보통"
        else:
            legal_viability = "낮음"

        return LegalEvidenceMapping(
            crime_type=crime_type,
            legal_code=legal_code,
            requirements=requirements,
            fulfillment_rate=fulfillment_rate,
            legal_viability=legal_viability,
        )
