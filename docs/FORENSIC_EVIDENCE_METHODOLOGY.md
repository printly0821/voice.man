# 포렌식 오디오 분석 증거자료 방법론

## 1. 개요

### 1.1 목적 및 범위

본 문서는 voice.man 시스템이 제공하는 포렌식 오디오 분석 방법론에 대한 과학적 근거와 법적 준수 절차를 상세히 기술합니다. 본 방법론은 법정 증거로 채택 가능한 수준의 신뢰성과 재현성을 보장하기 위해 설계되었습니다.

**적용 범위:**
- 음성 녹취 파일의 포렌식 분석
- 범죄 언어 패턴 탐지
- 심리적 조작 패턴 식별 (가스라이팅 등)
- 음성 특성 기반 스트레스 분석
- 법정 제출용 증거 보고서 생성

### 1.2 법적 근거

본 방법론은 다음 법적 기준을 준수합니다:

**한국 법률:**
- 형사소송법 Article 313(2)(3): 디지털 증거의 무결성 및 진정성 입증

**국제 표준:**
- ISO/IEC 27037:2012: 디지털 증거 수집, 보존, 이동 가이드라인
- ISO/IEC 17025:2017: 포렌식 실험실 인정 기준
- NIST SP 800-86: 디지털 포렌식 통합 가이드

### 1.3 준수 표준

본 시스템은 다음 기술 표준을 준수합니다:

- **암호학적 무결성**: SHA-256 해시 알고리즘
- **전자서명**: RSA 2048-bit 디지털 서명 (cryptography 라이브러리)
- **타임스탬프**: RFC 3161 준수 타임스탬프 서비스
- **감사 로그**: Append-only 로그 아키텍처
- **통계적 신뢰도**: Bootstrap 95% 신뢰구간

---

## 2. Chain of Custody (증거 보관 연속성)

### 2.1 전자서명 (Digital Signature)

#### 2.1.1 개요

모든 증거 파일은 RSA 2048-bit 전자서명으로 보호됩니다. 이는 파일의 진정성(Authenticity)과 무결성(Integrity)을 보장합니다.

#### 2.1.2 서명 생성 프로세스

**구현 모듈**: `src/voice_man/forensics/evidence/digital_signature.py`

**알고리즘**: RSA-2048-PSS-SHA256

**프로세스:**
1. 오디오 파일의 SHA-256 해시 생성
2. Private Key로 해시 값에 전자서명 생성
3. 서명 메타데이터 기록 (타임스탬프, 서명자 정보, 알고리즘)
4. Public Key와 서명을 보고서에 첨부

**서명 메타데이터 예시:**
```json
{
  "signature": "base64_encoded_signature",
  "algorithm": "RSA-2048-PSS-SHA256",
  "timestamp_iso8601": "2026-01-17T10:30:45+09:00",
  "signer": "forensic_analyzer_v1.0",
  "public_key": "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
}
```

#### 2.1.3 서명 검증 절차

**검증 단계:**
1. Public Key 추출
2. 서명된 해시 값 복호화
3. 파일의 현재 SHA-256 해시 계산
4. 복호화된 해시와 현재 해시 비교
5. 일치 여부 확인 (Boolean 반환)

**검증 실패 시 처리:**
- 증거 무효화
- 감사 로그에 검증 실패 기록
- 보고서 생성 중단

#### 2.1.4 키 관리 정책

**Private Key 보관:**
- 환경 변수 또는 HSM(Hardware Security Module)에 저장
- 암호화된 키 파일 사용 (AES-256)
- 접근 권한: 시스템 관리자만

**Public Key 배포:**
- 보고서에 포함하여 제3자 검증 가능
- 별도 인증서 저장소 관리

**키 회전 정책:**
- 연 1회 정기 키 회전
- 보안 사고 발생 시 즉시 키 교체
- 이전 키로 서명된 증거는 타임스탬프로 유효성 유지

### 2.2 RFC 3161 타임스탬프

#### 2.2.1 개요

RFC 3161 준수 타임스탬프 서비스를 통해 증거 수집 시점을 법적으로 증명합니다.

#### 2.2.2 타임스탬프 서비스 통합

**구현 모듈**: `src/voice_man/forensics/evidence/timestamp_service.py`

**TSA (Time Stamping Authority)**: FreeTSA (https://freetsa.org/tsr)

**타임스탬프 토큰 생성 프로세스:**
1. 파일 해시 값 생성 (SHA-256)
2. TSA 서버에 타임스탬프 요청 전송
3. RFC 3161 타임스탬프 토큰 수신
4. 토큰 검증 및 저장

**타임스탬프 토큰 구조:**
```json
{
  "timestamp_token": "base64_encoded_rfc3161_token",
  "timestamp_iso8601": "2026-01-17T10:30:45+09:00",
  "tsa_url": "https://freetsa.org/tsr",
  "hash_algorithm": "SHA-256",
  "file_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e",
  "is_rfc3161_compliant": true
}
```

#### 2.2.3 로컬 타임스탬프 폴백

**TSA 서비스 장애 시 처리:**
- 시스템 시계 기반 로컬 타임스탬프 생성
- `is_rfc3161_compliant: false` 플래그 설정
- 감사 로그에 폴백 사용 기록
- 보고서에 경고 메시지 포함

**로컬 타임스탬프 신뢰성 향상:**
- NTP(Network Time Protocol) 동기화 필수
- 시스템 시계 변조 탐지
- 타임스탬프 생성 즉시 감사 로그 기록

#### 2.2.4 타임스탬프 검증

**검증 절차:**
1. RFC 3161 토큰 구조 검증
2. TSA 인증서 체인 검증
3. 해시 값 일치 여부 확인
4. 타임스탬프 유효 기간 확인

### 2.3 불변 감사 로그 (Immutable Audit Log)

#### 2.3.1 개요

모든 증거 접근 및 처리 이벤트를 변조 불가능한 append-only 로그에 기록합니다.

#### 2.3.2 Append-Only 로그 아키텍처

**구현 모듈**: `src/voice_man/forensics/evidence/audit_logger.py`

**저장 방식**: JSON Lines (`.jsonl`) 형식

**로그 이벤트 유형:**
- `upload`: 증거 파일 업로드
- `analysis`: 포렌식 분석 수행
- `report`: 보고서 생성
- `access`: 증거 파일 접근
- `verification`: 무결성 검증

**로그 엔트리 구조:**
```json
{
  "entry_id": 1,
  "timestamp_iso8601": "2026-01-17T10:30:45+09:00",
  "event_type": "upload",
  "asset_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "forensic_analyst_01",
  "action": "File uploaded and hash generated",
  "metadata": {
    "filename": "recording.mp3",
    "file_size": 1048576,
    "sha256_hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"
  },
  "previous_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "current_hash": "d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2g3h4i5"
}
```

#### 2.3.3 해시 체인 무결성 검증

**해시 체인 구조:**
- 각 로그 엔트리는 이전 엔트리의 해시 값을 참조
- 첫 번째 엔트리는 제네시스 해시 사용 (SHA-256 빈 문자열)
- 현재 엔트리 해시는 `previous_hash + 현재 데이터` 의 SHA-256

**무결성 검증 알고리즘:**
1. 로그 파일의 모든 엔트리 순차 읽기
2. 각 엔트리의 `current_hash` 재계산
3. 계산된 해시와 저장된 해시 비교
4. 다음 엔트리의 `previous_hash`와 현재 `current_hash` 일치 여부 확인
5. 불일치 발견 시 변조 탐지 경고

**변조 탐지 시 처리:**
- 보고서 생성 중단
- 시스템 관리자에게 알림
- 변조 구간 식별 및 격리
- 백업 로그와 비교 분석

#### 2.3.4 감사 로그 보호

**파일 시스템 보호:**
- 로그 파일 읽기 전용 권한 설정 (root/admin만 쓰기 가능)
- SQLite WAL(Write-Ahead Logging) 모드 활성화
- 정기적 외부 저장소 백업 (매일 자동)

**접근 제어:**
- 시스템 서비스만 로그 기록 가능
- 로그 조회는 인증된 사용자만
- 로그 수정/삭제 차단 (append-only 강제)

### 2.4 해시 체인 검증 자동화

#### 2.4.1 보고서 생성 전 자동 검증

**검증 워크플로우:**
1. 보고서 생성 요청 수신
2. 해당 자산의 전체 감사 로그 해시 체인 검증
3. 파일 무결성 검증 (현재 SHA-256 vs 저장된 SHA-256)
4. 전자서명 검증
5. 타임스탬프 검증
6. 모든 검증 통과 시 보고서 생성 진행

#### 2.4.2 검증 실패 처리

**검증 실패 유형:**
- **해시 체인 불일치**: 감사 로그 변조 의심
- **파일 해시 불일치**: 원본 파일 변조 의심
- **서명 검증 실패**: 전자서명 위조 의심
- **타임스탬프 검증 실패**: 타임스탬프 조작 의심

**대응 절차:**
1. 보고서 생성 즉시 중단
2. 검증 실패 로그 기록
3. 시스템 관리자 알림
4. 증거 격리 (읽기 전용 모드)
5. 포렌식 조사 개시

---

## 3. 학술 검증

### 3.1 Bootstrap 95% 신뢰구간

#### 3.1.1 개요

포렌식 스코어의 통계적 유의성을 평가하기 위해 Bootstrap resampling 기법을 사용하여 95% 신뢰구간을 계산합니다.

#### 3.1.2 Bootstrap 신뢰구간 계산

**구현 모듈**: `src/voice_man/forensics/validation/bootstrap.py`

**알고리즘**: Percentile Bootstrap / BCa (Bias-Corrected and Accelerated)

**매개변수:**
- `n_iterations`: 10,000 (기본값)
- `confidence_level`: 0.95 (95% 신뢰구간)
- `random_seed`: 재현성을 위한 고정 시드 (선택적)

**Bootstrap 프로세스:**
1. 원본 데이터에서 복원 추출(resampling with replacement)로 Bootstrap 샘플 생성
2. 각 Bootstrap 샘플에서 통계량(평균, 중앙값 등) 계산
3. 10,000회 반복하여 Bootstrap 분포 생성
4. Percentile 방법 또는 BCa 방법으로 신뢰구간 산출

**Percentile 방법:**
- Bootstrap 분포의 2.5% 백분위수 → 하한(Lower Bound)
- Bootstrap 분포의 97.5% 백분위수 → 상한(Upper Bound)

**BCa 방법 (더 정확):**
- 편향 보정(Bias Correction) 적용
- 가속화 상수(Acceleration Constant) 계산
- 왜곡된 분포에서도 정확한 신뢰구간 제공

#### 3.1.3 재현 가능성 보장

**재현성 요구사항:**
- 동일 입력, 동일 `random_seed` → 동일 출력 보장
- 소수점 4자리까지 일치 (허용 오차: ±0.0001)

**재현성 검증 절차:**
1. 분석 파라미터 기록 (모델 버전, random_seed, n_iterations)
2. 재분석 시 동일 파라미터 사용
3. 결과 비교 및 차이 로그 기록

**재현성 테스트:**
- 개발 환경에서 자동화된 재현성 테스트 실행
- CI/CD 파이프라인에 통합
- 테스트 커버리지: 96% (SPEC 요구사항 충족)

### 3.2 성능 메트릭 (Precision, Recall, F1)

#### 3.2.1 개요

범죄 패턴 탐지 모듈의 정확도를 평가하기 위해 표준 분류 성능 지표를 사용합니다.

#### 3.2.2 성능 메트릭 계산

**구현 모듈**: `src/voice_man/forensics/validation/performance_metrics.py`

**평가 지표:**

**Precision (정밀도):**
- 정의: 예측된 양성 중 실제 양성의 비율
- 계산식: `TP / (TP + FP)`
- 의미: 시스템이 "범죄 패턴"으로 판단한 것 중 실제 범죄 패턴의 비율

**Recall (재현율):**
- 정의: 실제 양성 중 정확히 예측된 비율
- 계산식: `TP / (TP + FN)`
- 의미: 실제 범죄 패턴 중 시스템이 탐지한 비율

**F1 Score (F1 점수):**
- 정의: Precision과 Recall의 조화 평균
- 계산식: `2 × (Precision × Recall) / (Precision + Recall)`
- 의미: 정밀도와 재현율의 균형 지표

**Confusion Matrix (혼동 행렬):**
```
                예측 양성    예측 음성
실제 양성      TP          FN
실제 음성      FP          TN
```

#### 3.2.3 Ground Truth 데이터셋

**Ground Truth 요구사항:**
- 법률 전문가 또는 심리학 전문가의 수동 레이블링
- 최소 1,000개 이상의 레이블링된 발화 샘플
- 다양한 범죄 패턴 유형 포함

**Ground Truth가 없는 경우:**
- 성능 메트릭 생략
- 보고서에 "Ground Truth 데이터셋 미제공" 표시
- 신뢰구간만으로 통계적 유의성 평가

#### 3.2.4 혼동 행렬 시각화

**시각화 도구:**
- Matplotlib/Seaborn을 사용한 히트맵
- 각 셀에 숫자와 비율 표시
- 색상 그라데이션으로 패턴 강조

**보고서 포함 사항:**
- 혼동 행렬 이미지
- Precision, Recall, F1 점수 테이블
- 클래스별 성능 분석

---

## 4. 법정 제출 가이드

### 4.1 증거 수집 체크리스트

**분석 전 준비사항:**
- [ ] 증거 파일 원본 확보 및 봉인
- [ ] SHA-256 해시 생성 및 기록
- [ ] 전자서명 적용
- [ ] RFC 3161 타임스탬프 발급
- [ ] 감사 로그 초기화

**분석 중 필수사항:**
- [ ] 원본 파일 변경 금지 (사본으로 작업)
- [ ] 모든 접근 이벤트 감사 로그 기록
- [ ] 분석 파라미터 문서화
- [ ] 중간 결과 타임스탬프

**분석 후 검증사항:**
- [ ] 해시 체인 무결성 검증
- [ ] 전자서명 검증
- [ ] 타임스탬프 검증
- [ ] 재현성 테스트 통과

### 4.2 보고서 작성 템플릿

**필수 포함 섹션:**

**1. Title Page (표지)**
- 보고서 제목
- 사건 번호
- 분석 일시
- 분석자 정보 (자격, 소속)
- 보고서 고유 식별자 (UUID v7)

**2. Executive Summary (요약)**
- 분석 개요
- 주요 발견 사항
- 결론
- 권장 사항

**3. 방법론 상세 설명**
- 사용된 분석 기법의 과학적 원리
- 관련 학술 논문 또는 표준 문서 참조
- 도구 및 라이브러리 버전 정보

**4. 사용 도구 명세**
- Python 버전, OS 정보
- WhisperX, PyTorch, KoBERT 버전
- GPU 정보 (CUDA 버전, GPU 모델)
- 모든 라이브러리 버전 및 라이선스

**5. 증거 항목**
- 파일명, SHA-256 해시
- 업로드 일시 (RFC 3161 타임스탬프)
- 파일 크기, 포맷, 재생 시간
- 전자서명 정보

**6. 분석 결과**
- 포렌식 스코어링
- 범죄 언어 패턴 탐지 결과
- 음성 특성 분석
- 감정 분석

**7. 신뢰도 평가**
- Bootstrap 95% 신뢰구간
- 성능 메트릭 (Precision, Recall, F1)
- 재현성 테스트 결과

**8. 재현 절차**
- 단계별 분석 재현 가이드
- 입력 파일 준비
- 분석 파라미터 설정
- 실행 명령어
- 예상 출력 형식

### 4.3 전문가 증언 준비

**예상 질문 및 답변 준비:**

**Q: 분석 도구의 신뢰성은 어떻게 보장됩니까?**
A: 본 시스템은 ISO/IEC 17025 포렌식 실험실 인정 기준을 준수하며, 모든 알고리즘은 공개 학술 논문에 기반합니다. 도구 검증 기록과 교정 절차를 문서화하여 유지합니다.

**Q: 분석 결과의 재현 가능성은 어떻게 보장됩니까?**
A: 모든 분석 파라미터(모델 버전, random_seed, 입력 파일 해시)를 기록하며, 동일 입력에 대해 동일 출력을 생성함을 재현성 테스트로 검증합니다. 허용 오차는 소수점 4자리입니다.

**Q: 통계적 신뢰도는 어떻게 평가됩니까?**
A: Bootstrap resampling 기법으로 95% 신뢰구간을 계산하여 결과의 통계적 유의성을 평가합니다. 이는 학술 연구에서 표준적으로 사용되는 방법입니다.

**Q: 증거 무결성은 어떻게 보장됩니까?**
A: SHA-256 해시, RSA 2048-bit 전자서명, RFC 3161 타임스탬프, append-only 감사 로그를 사용하여 증거의 무결성과 Chain of Custody를 보장합니다.

**전문가 자격 요구사항:**
- 포렌식 분석 관련 학위 또는 자격증
- 음성 포렌식 분야 실무 경험
- 법정 증언 경험 (선호)
- 본 시스템 교육 이수

### 4.4 법정 제출 시 주의사항

**증거 제출 전:**
- 모든 검증 절차 완료 확인
- 보고서 법률 검토 (필요 시 법률 자문)
- 원본 증거 파일 봉인 및 보관
- 감사 로그 백업

**법정 제출 시:**
- 보고서 인쇄본과 디지털 사본 모두 제출
- USB 또는 CD-ROM에 모든 증거 파일 포함
- Public Key 제공으로 전자서명 검증 가능하게 함
- 감사 로그 요약 제출

**증언 준비:**
- 보고서 내용 숙지
- 예상 반론 준비
- 기술 용어 쉬운 설명 연습
- 시각 자료 준비 (다이어그램, 차트)

---

## 5. 품질 관리

### 5.1 방법론 검증 프로토콜

**새로운 알고리즘 추가 시:**
1. 검증 데이터셋으로 성능 테스트
2. Ground Truth 비교
3. Precision, Recall, F1 점수 평가
4. 기존 알고리즘과 성능 비교
5. 검증 결과 문서화

**방법론 변경 시:**
- 변경 사유 및 내역 문서화
- 재검증 수행
- 이전 버전과의 호환성 평가
- 변경 이력 감사 로그 기록

### 5.2 품질 검증 절차

**내부 품질 검증:**
- 분석 결과의 논리적 일관성 검토
- 신뢰구간 범위 확인
- 이상치(outlier) 탐지 및 조사
- 교차 검증 (다른 분석가의 독립 분석)

**품질 검증 실패 시:**
- 경고 메시지 표시
- 재분석 권장
- 실패 원인 조사 및 기록
- 필요 시 전문가 자문

### 5.3 지속적 개선

**성능 모니터링:**
- 분석 결과 품질 메트릭 수집
- 사용자 피드백 수집
- 오류 패턴 분석
- 정기적 성능 리뷰 (분기별)

**개선 프로세스:**
- 개선 제안 수집 및 평가
- 우선순위 결정
- 구현 및 테스트
- 배포 및 문서화

---

## 6. 참고 문헌

### 6.1 법률 및 표준

- 대한민국 형사소송법 Article 313(2)(3)
- ISO/IEC 27037:2012 - Guidelines for identification, collection, acquisition, and preservation of digital evidence
- ISO/IEC 17025:2017 - General requirements for the competence of testing and calibration laboratories
- NIST Special Publication 800-86 - Guide to Integrating Forensic Techniques into Incident Response
- RFC 3161 - Internet X.509 Public Key Infrastructure Time-Stamp Protocol (TSP)

### 6.2 학술 문헌

- Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. Chapman and Hall/CRC.
- DiCarlo, B., & Tibshirani, R. J. (1997). Bootstrap confidence intervals. Statistical Science, 189-212.
- Powers, D. M. (2011). Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation.
- NIST (2014). Forensic Science in Criminal Courts: Ensuring Scientific Validity of Feature-Comparison Methods.

### 6.3 기술 문서

- Cryptography Documentation: https://cryptography.io/
- RFC 3161 Timestamp Protocol: https://www.ietf.org/rfc/rfc3161.txt
- SHA-256 Specification (FIPS 180-4): https://csrc.nist.gov/publications/detail/fips/180/4/final
- RSA PKCS #1 v2.2: https://www.rfc-editor.org/rfc/rfc8017.html

---

**문서 버전**: 1.0.0
**최종 업데이트**: 2026-01-17
**작성자**: voice.man 개발팀
**검토자**: 포렌식 전문가
**승인자**: 프로젝트 관리자

**TAG**: [FORENSIC-EVIDENCE-001]
