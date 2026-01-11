---
id: SPEC-FORENSIC-WEB-001
version: "1.0.0"
status: "planned"
created: "2026-01-10"
updated: "2026-01-10"
author: "지니"
priority: "HIGH"
title: "웹 기반 포렌식 증거 프레젠테이션 시스템"
related_specs:
  - SPEC-FORENSIC-001
  - SPEC-VOICE-001
  - SPEC-WHISPERX-001
tags:
  - 웹애플리케이션
  - 포렌식증거
  - 법적프레젠테이션
  - 시각화대시보드
  - Next.js
  - FastAPI
lifecycle: "spec-anchored"
---

# SPEC-FORENSIC-WEB-001: 웹 기반 포렌식 증거 프레젠테이션 시스템

## HISTORY

| 버전 | 날짜 | 작성자 | 변경 내용 |
|------|------|--------|-----------|
| 1.0.0 | 2026-01-10 | 지니 | 초안 작성 - 웹 기반 포렌식 증거 프레젠테이션 시스템 요구사항 정의 |

---

## 1. 개요

### 1.1 목적

SPEC-FORENSIC-001에서 분석 완료된 183개 오디오 파일의 포렌식 결과를 웹 기반 인터페이스로 제공한다. 법률 전문가, 의뢰인, 포렌식 전문가가 증거를 효과적으로 탐색하고 고소장의 각 혐의와 매핑할 수 있는 시스템을 구축한다.

### 1.2 핵심 목표

1. **인터랙티브 고소장 뷰어**: 고소장 텍스트와 포렌식 증거 연결
2. **포렌식 분석 탐색기**: 183개 파일의 메타데이터 검색 및 필터링
3. **증거 비교 대시보드**: 텍스트 vs 포렌식 증거 대조
4. **보안 및 접근 제어**: 역할 기반 인증 및 데이터 암호화

### 1.3 범위

- 기존 SPEC-FORENSIC-001 분석 결과물 활용 (104개 HTML 보고서, 15개 JSON 파일)
- Next.js 16 기반 프론트엔드 구축
- FastAPI 기반 백엔드 API 확장
- 두 고소장 (신기연: 8개 혐의, 신동식: 3개 혐의) 증거 매핑

### 1.4 고소장 구조

#### 신기연 고소장 (8개 혐의)
1. 사기 (형법 제347조)
2. 공갈 (형법 제350조)
3. 강요 (형법 제324조)
4. 협박 (형법 제283조)
5. 모욕 (형법 제311조)
6. 횡령 (형법 제355조)
7. 조세포탈 (조세범처벌법)
8. 무등록영업 (관련 행정법규)

#### 신동식 고소장 (3개 혐의)
1. 배임 (형법 제355조 제2항)
2. 사기 (형법 제347조)
3. 공갈 (형법 제350조)

---

## 2. 환경 (Environment)

### 2.1 기술 스택

#### 2.1.1 프론트엔드
- **Next.js 16**: App Router, Server Components, Server Actions
- **React 19**: Concurrent features, use() hook
- **TypeScript 5.9+**: 타입 안전성
- **Tailwind CSS 4.0**: 스타일링
- **shadcn/ui**: 컴포넌트 라이브러리
- **Chart.js / Recharts**: 시각화
- **React Query (TanStack Query)**: 서버 상태 관리

#### 2.1.2 백엔드
- **FastAPI 0.115+**: REST API (기존 voice.man 확장)
- **Pydantic v2.9+**: 데이터 검증
- **SQLAlchemy 2.0+**: ORM (비동기)
- **PostgreSQL 17**: 데이터베이스
- **Redis**: 세션 및 캐싱

#### 2.1.3 인증 및 보안
- **JWT (PyJWT)**: 토큰 기반 인증
- **bcrypt**: 비밀번호 해싱
- **cryptography**: AES-256 데이터 암호화
- **python-jose**: JOSE 표준 구현

#### 2.1.4 미디어 처리
- **Wavesurfer.js**: 오디오 파형 시각화
- **Howler.js**: 오디오 재생

### 2.2 시스템 요구사항

- Node.js 22+ (프론트엔드)
- Python 3.13+ (백엔드)
- PostgreSQL 17+
- Redis 7+
- HTTPS 필수 (법적 증거 전송)

---

## 3. 가정 (Assumptions)

### 3.1 기술적 가정

1. SPEC-FORENSIC-001의 분석 결과물(104개 HTML, 15개 JSON)이 사용 가능하다
2. 기존 FastAPI 백엔드에 새 엔드포인트 추가가 가능하다
3. 183개 오디오 파일이 스트리밍 가능한 형태로 저장되어 있다
4. PostgreSQL 데이터베이스에 포렌식 결과 스키마 추가가 가능하다

### 3.2 비즈니스 가정

1. 접근 권한은 변호사, 의뢰인, 포렌식 전문가 3가지 역할로 구분된다
2. 증거 자료는 법적 제출 전까지 외부 공개되지 않아야 한다
3. 6개월간의 에스컬레이션 패턴 시각화가 핵심 요구사항이다
4. 고소장 텍스트와 증거 매핑은 수동으로 1차 입력 후 시스템에서 관리한다

### 3.3 보안 가정

1. 모든 통신은 HTTPS로 암호화된다
2. 민감 데이터(개인정보, 녹취 내용)는 저장 시 AES-256 암호화된다
3. 세션은 JWT 기반으로 관리되며 만료 시간이 설정된다
4. 감사 로그(Audit Log)가 모든 접근 기록을 저장한다

---

## 4. 요구사항 (Requirements)

### 4.1 Ubiquitous Requirements (U) - 전역 요구사항

#### U1: HTTPS 통신 필수
**요구사항**: 시스템은 **항상** HTTPS를 통해 모든 클라이언트-서버 통신을 암호화해야 한다.

**세부사항**:
- TLS 1.3 이상
- 유효한 SSL 인증서
- HTTP 요청은 HTTPS로 리다이렉트

**WHY**: 법적 증거 데이터의 전송 보안은 필수 요건이다.
**IMPACT**: 비암호화 통신 시 증거 무결성 및 기밀성 훼손.

---

#### U2: 감사 로그 기록
**요구사항**: 시스템은 **항상** 모든 사용자 접근 및 데이터 조회 기록을 감사 로그로 저장해야 한다.

**세부사항**:
- 사용자 ID, 타임스탬프, 접근 리소스, IP 주소
- 로그 보존 기간: 최소 5년
- 로그 변조 방지 (append-only)

**WHY**: 법적 증거 관리 과정의 추적성 및 책임성 확보.
**IMPACT**: 감사 로그 부재 시 증거 관리 과정 입증 불가.

---

#### U3: 응답 시간 기준
**요구사항**: 시스템은 **항상** 일반 페이지 로드 2초 이내, API 응답 500ms 이내를 유지해야 한다.

**세부사항**:
- 페이지 로드: P95 < 2초
- API 응답: P95 < 500ms
- 오디오 스트리밍: 초기 버퍼링 < 1초

**WHY**: 법률 전문가의 효율적인 증거 검토 지원.
**IMPACT**: 느린 응답은 사용자 경험 저하 및 업무 효율성 감소.

---

### 4.2 Event-Driven Requirements (E) - 이벤트 기반 요구사항

#### E1: 고소장 혐의 클릭 시 증거 로드
**요구사항**: **WHEN** 사용자가 고소장 내 특정 혐의를 클릭하면 **THEN** 시스템은 해당 혐의와 매핑된 모든 증거 목록을 표시해야 한다.

**세부사항**:
- 혐의별 증거 매핑 데이터 로드
- 증거 유형별 그룹핑 (오디오, 텍스트, 포렌식 분석)
- 증거 신뢰도 점수 표시

**WHY**: 혐의별 증거 탐색의 직관성 확보.
**IMPACT**: 연결 불가 시 증거 활용도 저하.

---

#### E2: 증거 클릭 시 오디오 재생
**요구사항**: **WHEN** 사용자가 증거 목록에서 오디오 증거를 클릭하면 **THEN** 시스템은 해당 시점으로 오디오를 로드하고 재생을 시작해야 한다.

**세부사항**:
- 타임스탬프 기반 시점 이동
- 파형 시각화와 동기화
- 해당 구간 전사 텍스트 하이라이트

**WHY**: 오디오 증거의 즉각적 확인 지원.
**IMPACT**: 재생 기능 미비 시 증거 검증 불가.

---

#### E3: 필터 변경 시 실시간 검색
**요구사항**: **WHEN** 사용자가 검색 필터(포렌식 점수, 날짜, 화자, 혐의 유형)를 변경하면 **THEN** 시스템은 500ms 이내에 필터링된 결과를 표시해야 한다.

**세부사항**:
- 디바운싱 적용 (300ms)
- 점진적 로딩 (Pagination)
- 필터 조합 AND/OR 지원

**WHY**: 대용량 증거 데이터의 효율적 탐색.
**IMPACT**: 느린 필터링은 사용자 경험 저하.

---

#### E4: 보고서 생성 요청 시 비동기 처리
**요구사항**: **WHEN** 사용자가 PDF/HTML 보고서 생성을 요청하면 **THEN** 시스템은 비동기로 생성 작업을 처리하고 완료 시 알림을 제공해야 한다.

**세부사항**:
- 작업 대기열(Queue) 관리
- 진행률 표시
- 완료 시 다운로드 링크 제공

**WHY**: 대용량 보고서 생성은 시간이 소요되므로 비동기 처리 필요.
**IMPACT**: 동기 처리 시 요청 타임아웃 및 UX 저하.

---

### 4.3 State-Driven Requirements (S) - 상태 기반 요구사항

#### S1: 인증 상태 기반 접근 제어
**요구사항**: **IF** 사용자가 인증되지 않은 상태이면 **THEN** 시스템은 로그인 페이지로 리다이렉트하고 보호된 리소스 접근을 차단해야 한다.

**세부사항**:
- JWT 토큰 유효성 검증
- 세션 만료 처리 (기본 4시간)
- 자동 로그아웃 경고

**WHY**: 법적 증거에 대한 무단 접근 방지.
**IMPACT**: 인증 미비 시 증거 유출 위험.

---

#### S2: 역할 기반 기능 제한
**요구사항**: **IF** 사용자 역할이 "의뢰인(client)"이면 **THEN** 시스템은 보고서 생성 및 증거 매핑 수정 기능을 숨겨야 한다.

**역할별 권한**:
| 기능 | 변호사(lawyer) | 의뢰인(client) | 전문가(expert) |
|------|---------------|----------------|----------------|
| 고소장 조회 | O | O | O |
| 증거 탐색 | O | O | O |
| 오디오 재생 | O | O | O |
| 증거 매핑 수정 | O | X | O |
| 보고서 생성 | O | X | O |
| 사용자 관리 | O | X | X |

**WHY**: 역할별 기능 분리로 오용 방지.
**IMPACT**: 권한 혼동 시 부적절한 데이터 수정 위험.

---

#### S3: 오프라인 상태 대응
**요구사항**: **IF** 네트워크 연결이 끊어진 상태이면 **THEN** 시스템은 캐시된 데이터를 표시하고 연결 복구 시 자동 동기화해야 한다.

**세부사항**:
- Service Worker 기반 캐싱
- 읽기 전용 오프라인 모드
- 연결 복구 시 자동 재시도

**WHY**: 불안정한 네트워크 환경에서도 기본 기능 유지.
**IMPACT**: 오프라인 미지원 시 현장 업무 중단 가능.

---

### 4.4 Feature Requirements (F) - 기능 요구사항

#### F1: 인터랙티브 고소장 뷰어

**요구사항**: 시스템은 고소장 텍스트를 인터랙티브하게 표시하고 증거와 연결해야 한다.

**세부사항**:
1. **고소장 텍스트 렌더링**
   - 마크다운 또는 HTML 형식 지원
   - 혐의별 섹션 구분
   - 텍스트 하이라이팅 (증거 연결 구간)

2. **증거 연결 표시**
   - 인라인 주석 (tooltip)
   - 사이드바 증거 목록
   - 클릭 시 증거 상세 팝업

3. **혐의별 증거 수 표시**
   - 혐의 제목 옆 증거 개수 배지
   - 증거 유형별 아이콘 표시

**WHY**: 변호사의 고소장 검토 효율성 극대화.
**IMPACT**: 수동 증거 검색 대비 시간 단축.

---

#### F2: 타임라인 시각화

**요구사항**: 시스템은 6개월간의 에스컬레이션 패턴을 타임라인으로 시각화해야 한다.

**세부사항**:
1. **시간축 타임라인**
   - X축: 날짜 (2025년 7월 ~ 2026년 1월)
   - Y축: 심리 압박 강도 (0-100)
   - 마커: 개별 녹취 파일

2. **패턴 유형별 색상 코드**
   - 가스라이팅: 보라색
   - 위협: 빨간색
   - 강압: 주황색
   - 기만: 노란색

3. **인터랙션**
   - 호버 시 상세 정보 표시
   - 클릭 시 해당 증거로 이동
   - 줌 및 패닝 지원

**WHY**: 장기간 심리 조작 패턴의 시각적 입증.
**IMPACT**: 산발적 증거의 연속성 증명.

---

#### F3: 포렌식 분석 탐색기

**요구사항**: 시스템은 183개 파일의 포렌식 분석 결과를 검색 및 필터링해야 한다.

**세부사항**:
1. **검색 필터**
   - 포렌식 점수 범위 (가스라이팅, 위협, 강압, 기만)
   - 날짜 범위
   - 화자 선택
   - 혐의 유형

2. **결과 목록**
   - 페이지네이션 (20개씩)
   - 정렬 (점수, 날짜, 파일명)
   - 그리드/리스트 뷰 전환

3. **상세 보기**
   - 포렌식 점수 레이더 차트
   - 음성 특성 그래프 (F0, Jitter, 음량)
   - 전사 텍스트

**WHY**: 대용량 증거 데이터의 효율적 탐색.
**IMPACT**: 수동 검색 대비 시간 90% 이상 단축.

---

#### F4: 시각화 그래프

**요구사항**: 시스템은 포렌식 분석 데이터를 시각화 그래프로 표시해야 한다.

**세부사항**:
1. **F0 (기본 주파수) 그래프**
   - 시간축 피치 변화
   - 화자별 분리 표시
   - 정상 범위 음영

2. **Jitter/Shimmer 차트**
   - 스트레스 지표 시각화
   - 임계값 표시선
   - 이상치 강조

3. **위협 강도 히트맵**
   - 시간-강도 매트릭스
   - 색상 그라데이션
   - 클릭 시 해당 구간 이동

**WHY**: 복잡한 포렌식 데이터의 직관적 이해.
**IMPACT**: 전문 지식 없이도 패턴 인식 가능.

---

#### F5: 증거 비교 대시보드

**요구사항**: 시스템은 텍스트 증거와 포렌식 분석 결과를 대조해야 한다.

**세부사항**:
1. **분할 뷰**
   - 좌측: 전사 텍스트
   - 우측: 포렌식 분석 결과
   - 동기화 스크롤

2. **불일치 강조**
   - 텍스트 감정 vs 음성 감정 불일치 표시
   - 신뢰도 점수 비교
   - 설명 주석

3. **신뢰도 게이지**
   - 종합 신뢰도 점수
   - 구성 요소별 점수 분해
   - 신뢰도 등급 (높음/중간/낮음)

**WHY**: 텍스트-음성 교차 검증으로 증거 신뢰도 강화.
**IMPACT**: 불일치 발견 시 증거 해석 오류 방지.

---

#### F6: 보고서 자동 생성

**요구사항**: 시스템은 선택된 증거를 기반으로 PDF/HTML 보고서를 자동 생성해야 한다.

**세부사항**:
1. **보고서 유형**
   - 혐의별 증거 요약 보고서
   - 타임라인 분석 보고서
   - 화자별 프로파일 보고서
   - 전체 종합 보고서

2. **커스터마이징**
   - 포함 증거 선택
   - 표지 정보 입력
   - 출력 형식 선택 (PDF/HTML)

3. **생성 프로세스**
   - 비동기 생성
   - 진행률 표시
   - 다운로드/이메일 전송

**WHY**: 법정 제출용 문서 작성 자동화.
**IMPACT**: 수동 보고서 작성 대비 시간 80% 단축.

---

#### F7: JWT 인증 시스템

**요구사항**: 시스템은 JWT 기반 인증을 구현해야 한다.

**세부사항**:
1. **로그인**
   - 이메일/비밀번호 인증
   - Access Token (4시간) + Refresh Token (7일)
   - 다중 기기 로그인 지원

2. **토큰 관리**
   - Access Token: Authorization 헤더
   - Refresh Token: HTTP-only 쿠키
   - 자동 갱신 (만료 10분 전)

3. **보안 기능**
   - 비밀번호 해싱 (bcrypt)
   - 로그인 시도 제한 (5회/15분)
   - 비정상 접근 알림

**WHY**: 상태 없는(stateless) 확장 가능한 인증.
**IMPACT**: 세션 서버 부하 감소 및 수평 확장 용이.

---

#### F8: 역할 기반 접근 제어 (RBAC)

**요구사항**: 시스템은 역할 기반 접근 제어를 구현해야 한다.

**세부사항**:
1. **역할 정의**
   - lawyer: 전체 접근
   - client: 조회 전용
   - expert: 분석 및 보고서 접근

2. **권한 검사**
   - 프론트엔드: 컴포넌트 조건부 렌더링
   - 백엔드: API 미들웨어 검증
   - 데이터베이스: Row-level 보안

3. **관리 기능**
   - 사용자 초대 (이메일)
   - 역할 변경
   - 접근 기록 조회

**WHY**: 민감 데이터의 최소 권한 원칙 적용.
**IMPACT**: 권한 초과 접근으로 인한 데이터 오용 방지.

---

#### F9: AES-256 데이터 암호화

**요구사항**: 시스템은 민감 데이터를 AES-256으로 암호화해야 한다.

**세부사항**:
1. **암호화 대상**
   - 전사 텍스트
   - 오디오 파일 경로
   - 개인 식별 정보

2. **키 관리**
   - 환경 변수 기반 마스터 키
   - 키 로테이션 지원
   - 키 백업 절차

3. **암/복호화**
   - 저장 시 암호화 (at-rest)
   - 전송 시 HTTPS (in-transit)
   - 복호화는 권한 있는 요청에만

**WHY**: 법적 증거의 기밀성 보장.
**IMPACT**: 암호화 미적용 시 데이터 유출 위험.

---

### 4.5 Optional Requirements (O) - 선택적 요구사항

#### O1: 다국어 지원
**요구사항**: **가능하면** 시스템은 한국어 외 영어 인터페이스를 지원할 수 있다.

**WHY**: 국제 법률 협력 시 활용 가능.
**IMPACT**: 미구현 시 한국어 전용 운영.

---

#### O2: 실시간 협업
**요구사항**: **가능하면** 여러 사용자가 동시에 증거를 검토하고 주석을 공유할 수 있다.

**WHY**: 변호사팀 협업 효율성 향상.
**IMPACT**: 미구현 시 개별 작업 후 통합.

---

#### O3: 모바일 최적화
**요구사항**: **가능하면** 모바일 기기에서도 주요 기능을 사용할 수 있다.

**WHY**: 현장 증거 검토 지원.
**IMPACT**: 미구현 시 데스크톱 전용.

---

### 4.6 Unwanted Requirements (N) - 금지 요구사항

#### N1: 원본 데이터 노출 금지
**요구사항**: 시스템은 **절대** 암호화되지 않은 원본 오디오 또는 전사 텍스트를 외부 네트워크로 전송하지 않아야 한다.

**세부사항**:
- 모든 전송 데이터 암호화
- 다운로드 시 워터마크 적용
- 직접 URL 접근 차단

**WHY**: 증거 유출 방지.
**IMPACT**: 원본 노출 시 법적 증거력 및 개인정보 침해.

---

#### N2: 세션 정보 클라이언트 저장 금지
**요구사항**: 시스템은 **절대** Access Token을 localStorage에 저장하지 않아야 한다.

**세부사항**:
- Access Token: 메모리(React state/context)
- Refresh Token: HTTP-only 쿠키
- XSS 공격 방지

**WHY**: 토큰 탈취 위험 최소화.
**IMPACT**: localStorage 저장 시 XSS 취약점 노출.

---

#### N3: 증거 삭제 기능 금지
**요구사항**: 시스템은 **절대** 업로드된 증거 파일이나 분석 결과를 삭제하는 기능을 제공하지 않아야 한다.

**세부사항**:
- Soft delete만 허용 (숨김 처리)
- 삭제 시도 감사 로그 기록
- 관리자 권한으로도 물리적 삭제 불가

**WHY**: 증거 훼손 방지 및 Chain of Custody 유지.
**IMPACT**: 삭제 기능 존재 시 증거 조작 의혹 제기 가능.

---

## 5. 명세 (Specifications)

### 5.1 데이터 모델

```typescript
// 고소장
interface Complaint {
  id: string;
  title: string;           // "신기연 고소장", "신동식 고소장"
  defendant: string;       // 피고소인
  charges: Charge[];       // 혐의 목록
  createdAt: Date;
  updatedAt: Date;
}

// 혐의
interface Charge {
  id: string;
  complaintId: string;
  chargeType: ChargeType;  // 사기, 공갈, 강요 등
  legalBasis: string;      // 형법 제347조
  description: string;     // 혐의 내용
  evidences: Evidence[];   // 매핑된 증거
  order: number;           // 표시 순서
}

// 증거
interface Evidence {
  id: string;
  chargeIds: string[];     // 연결된 혐의 (다대다)
  audioFileId: string;     // 원본 오디오
  forensicReportId: string; // 포렌식 분석 결과
  startTime: number;       // 시작 타임스탬프
  endTime: number;         // 종료 타임스탬프
  transcriptExcerpt: string; // 전사 발췌
  confidenceScore: number; // 증거 신뢰도
  evidenceType: EvidenceType; // 유형
  createdAt: Date;
  createdBy: string;       // 생성자
}

// 포렌식 분석 결과
interface ForensicReport {
  id: string;
  audioFileId: string;
  gaslightingScore: number;  // 0-100
  threatScore: number;       // 0-100
  coercionScore: number;     // 0-100
  deceptionScore: number;    // 0-100
  emotionProfile: EmotionProfile;
  audioFeatures: AudioFeatures;
  timestamp: Date;
}

// 사용자
interface User {
  id: string;
  email: string;
  passwordHash: string;
  role: UserRole;
  name: string;
  createdAt: Date;
  lastLogin: Date;
  isActive: boolean;
}

type UserRole = 'lawyer' | 'client' | 'expert';
type ChargeType = 'fraud' | 'extortion' | 'coercion' | 'threat' | 'insult' | 'embezzlement' | 'tax_evasion' | 'unlicensed_business' | 'breach_of_trust';
type EvidenceType = 'audio' | 'text' | 'forensic' | 'document';
```

### 5.2 API 엔드포인트

#### 인증 API
| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/v1/auth/login` | POST | 로그인 |
| `/api/v1/auth/logout` | POST | 로그아웃 |
| `/api/v1/auth/refresh` | POST | 토큰 갱신 |
| `/api/v1/auth/me` | GET | 현재 사용자 정보 |

#### 고소장 API
| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/v1/complaints` | GET | 고소장 목록 조회 |
| `/api/v1/complaints/{id}` | GET | 고소장 상세 조회 |
| `/api/v1/complaints/{id}/charges` | GET | 혐의 목록 조회 |
| `/api/v1/complaints/{id}/timeline` | GET | 타임라인 데이터 |

#### 증거 API
| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/v1/evidences` | GET | 증거 목록 조회 (필터링) |
| `/api/v1/evidences/{id}` | GET | 증거 상세 조회 |
| `/api/v1/evidences/{id}/audio` | GET | 오디오 스트리밍 |
| `/api/v1/evidences/mapping` | POST | 증거-혐의 매핑 (lawyer, expert) |
| `/api/v1/evidences/mapping/{id}` | PUT | 매핑 수정 |

#### 포렌식 API
| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/v1/forensic/reports` | GET | 포렌식 리포트 목록 |
| `/api/v1/forensic/reports/{id}` | GET | 리포트 상세 |
| `/api/v1/forensic/timeline` | GET | 에스컬레이션 타임라인 |
| `/api/v1/forensic/comparison/{id}` | GET | 텍스트-음성 비교 |

#### 보고서 API
| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/v1/reports/generate` | POST | 보고서 생성 요청 |
| `/api/v1/reports/{id}/status` | GET | 생성 상태 조회 |
| `/api/v1/reports/{id}/download` | GET | 보고서 다운로드 |

### 5.3 페이지 구조

```
/                           # 대시보드 (인증 후)
/login                      # 로그인
/complaints                 # 고소장 목록
/complaints/[id]            # 고소장 상세 (인터랙티브 뷰어)
/complaints/[id]/charges/[chargeId]  # 혐의별 증거
/evidence                   # 증거 탐색기
/evidence/[id]              # 증거 상세
/timeline                   # 타임라인 시각화
/reports                    # 보고서 관리
/reports/generate           # 보고서 생성
/admin/users                # 사용자 관리 (lawyer only)
```

---

## 6. 추적성 (Traceability)

### 6.1 관련 문서

- `SPEC-FORENSIC-001`: 범죄 프로파일링 기반 음성 포렌식 분석 시스템 (입력)
- `SPEC-VOICE-001`: 음성 녹취 기반 증거 분석 시스템 (기반)
- `SPEC-WHISPERX-001`: WhisperX 통합 파이프라인 (전사)
- `plan.md`: 구현 계획 및 마일스톤
- `acceptance.md`: 인수 기준 및 테스트 시나리오

### 6.2 태그

```
[SPEC-FORENSIC-WEB-001]
[웹애플리케이션] [포렌식증거] [법적프레젠테이션]
[Next.js] [FastAPI] [JWT인증] [RBAC]
[시각화대시보드] [보안암호화]
```

### 6.3 구현 예정 파일

```
# 프론트엔드 (Next.js 16)
frontend/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                    # 대시보드
│   ├── login/page.tsx              # 로그인
│   ├── complaints/
│   │   ├── page.tsx                # 고소장 목록
│   │   └── [id]/
│   │       ├── page.tsx            # 고소장 상세
│   │       └── charges/[chargeId]/page.tsx
│   ├── evidence/
│   │   ├── page.tsx                # 증거 탐색기
│   │   └── [id]/page.tsx           # 증거 상세
│   ├── timeline/page.tsx           # 타임라인
│   ├── reports/
│   │   ├── page.tsx                # 보고서 목록
│   │   └── generate/page.tsx       # 생성
│   └── admin/users/page.tsx        # 사용자 관리
├── components/
│   ├── complaint/
│   │   ├── ComplaintViewer.tsx     # 고소장 뷰어
│   │   ├── ChargeCard.tsx          # 혐의 카드
│   │   └── EvidenceLink.tsx        # 증거 링크
│   ├── evidence/
│   │   ├── EvidenceList.tsx        # 증거 목록
│   │   ├── EvidenceFilter.tsx      # 필터
│   │   └── AudioPlayer.tsx         # 오디오 플레이어
│   ├── visualization/
│   │   ├── Timeline.tsx            # 타임라인
│   │   ├── ForensicRadar.tsx       # 레이더 차트
│   │   └── F0Chart.tsx             # F0 그래프
│   └── auth/
│       ├── LoginForm.tsx           # 로그인 폼
│       └── ProtectedRoute.tsx      # 보호 라우트
├── lib/
│   ├── api.ts                      # API 클라이언트
│   ├── auth.ts                     # 인증 유틸
│   └── encryption.ts               # 클라이언트 암호화
└── hooks/
    ├── useAuth.ts                  # 인증 훅
    └── useForensicData.ts          # 데이터 훅

# 백엔드 (FastAPI)
src/voice_man/
├── api/
│   ├── web/
│   │   ├── __init__.py
│   │   ├── auth.py                 # 인증 API
│   │   ├── complaints.py           # 고소장 API
│   │   ├── evidences.py            # 증거 API
│   │   ├── forensic.py             # 포렌식 API
│   │   └── reports.py              # 보고서 API
│   └── middleware/
│       ├── auth.py                 # JWT 검증
│       └── audit.py                # 감사 로그
├── models/
│   └── web/
│       ├── complaint.py            # 고소장 모델
│       ├── evidence.py             # 증거 모델
│       ├── user.py                 # 사용자 모델
│       └── audit_log.py            # 감사 로그
├── services/
│   └── web/
│       ├── auth_service.py         # 인증 서비스
│       ├── complaint_service.py    # 고소장 서비스
│       ├── evidence_service.py     # 증거 서비스
│       └── report_service.py       # 보고서 서비스
└── security/
    ├── jwt.py                      # JWT 처리
    ├── encryption.py               # AES-256
    └── rbac.py                     # 역할 기반 접근 제어
```

---

## 7. 참조 (References)

### 7.1 기술 문서

- [Next.js 16 Documentation](https://nextjs.org/docs)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT Best Practices](https://auth0.com/docs/secure/tokens/json-web-tokens)
- [OWASP Authentication Cheatsheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

### 7.2 법적 참조

- 형법 관련 조항 (제283조, 제311조, 제324조, 제347조, 제350조, 제355조)
- 개인정보보호법
- 정보통신망법

---

**문서 끝**
