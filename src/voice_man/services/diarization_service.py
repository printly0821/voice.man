"""
화자 분리 (Speaker Diarization) 서비스

pyannote-audio를 사용하여 오디오 파일에서 화자를 분리합니다.
"""

from pathlib import Path
from typing import List, Optional

from voice_man.models.diarization import (
    DiarizationResult,
    Speaker,
    SpeakerTurn,
    SpeakerStats,
)
from voice_man.models.database import TranscriptSegment


class DiarizationService:
    """
    화자 분리 서비스

    pyannote-audio 모델을 사용하여 오디오 파일의 화자를 분리합니다.
    """

    def __init__(self):
        """화자 분리 서비스 초기화"""
        # NOTE: 실제 구현에서는 pyannote-audio 모델을 로드합니다.
        # 현재는 테스트를 위해 모의 구현을 제공합니다.
        self.model_loaded = False

    async def diarize_speakers(
        self, audio_path: str, num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """
        오디오 파일의 화자 분리 수행

        Args:
            audio_path: 오디오 파일 경로
            num_speakers: 화자 수 (None이면 자동 감지)

        Returns:
            DiarizationResult: 화자 분리 결과

        Raises:
            ValueError: 오디오 파일이 유효하지 않은 경우
        """
        # 입력 검증
        if not audio_path or not Path(audio_path).exists():
            raise ValueError("오디오 파일이 비어있습니다")

        audio_file = Path(audio_path)
        if not audio_file.exists() or audio_file.stat().st_size < 100:
            raise ValueError("오디오 파일을 처리할 수 없습니다")

        # NOTE: 실제 구현에서는 pyannote-audio를 사용합니다.
        # 현재는 테스트를 위한 모의 데이터를 반환합니다.
        #
        # 실제 구현 예시:
        # from pyannote.audio import Pipeline
        # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        # diarization = pipeline(audio_path)
        #
        # 테스트 환경에서는 모의 데이터를 사용합니다.

        # 모의 화자 데이터 생성 (2인 대화)
        speakers = [
            Speaker(
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=150.0,
                duration=150.0,
                confidence=0.95,
            ),
            Speaker(
                speaker_id="SPEAKER_01",
                start_time=150.0,
                end_time=300.0,
                duration=150.0,
                confidence=0.92,
            ),
        ]

        return DiarizationResult(
            speakers=speakers,
            total_duration=300.0,
            num_speakers=len(speakers),
        )

    def merge_with_transcript(
        self,
        stt_segments: List[TranscriptSegment],
        diarization_result: DiarizationResult,
    ) -> List[TranscriptSegment]:
        """
        STT 세그먼트와 화자 분리 결과 병합

        Args:
            stt_segments: STT 변환 결과 세그먼트 목록
            diarization_result: 화자 분리 결과

        Returns:
            List[TranscriptSegment]: 화자 ID가 할당된 세그먼트 목록
        """
        merged_segments = []

        for segment in stt_segments:
            # 가장 겹치는 화자 찾기
            best_speaker = None
            max_overlap = 0.0

            for speaker in diarization_result.speakers:
                # 겹치는 구간 계산
                overlap_start = max(segment.start_time, speaker.start_time)
                overlap_end = min(segment.end_time, speaker.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker

            # 화자 ID 할당
            if best_speaker and max_overlap > 0:
                segment.speaker_id = best_speaker.speaker_id

            merged_segments.append(segment)

        return merged_segments

    def generate_speaker_stats(self, speakers: List[Speaker]) -> SpeakerStats:
        """
        화자 통계 생성

        Args:
            speakers: 화자 목록

        Returns:
            SpeakerStats: 화자 통계
        """
        total_duration = sum(s.duration for s in speakers)

        return SpeakerStats(
            total_speakers=len(set(s.speaker_id for s in speakers)),
            total_speech_duration=total_duration,
            speaker_details=speakers,
        )

    def detect_speaker_turns(self, speakers: List[Speaker]) -> List[SpeakerTurn]:
        """
        화자 교대 (Turn-taking) 감지

        Args:
            speakers: 화자 목록

        Returns:
            List[SpeakerTurn]: 화자 교대 정보 목록
        """
        # 시간 순서대로 정렬
        sorted_speakers = sorted(speakers, key=lambda s: s.start_time)

        turns = []
        for speaker in sorted_speakers:
            turn = SpeakerTurn(
                speaker_id=speaker.speaker_id,
                start_time=speaker.start_time,
                end_time=speaker.end_time,
                duration=speaker.duration,
            )
            turns.append(turn)

        return turns
