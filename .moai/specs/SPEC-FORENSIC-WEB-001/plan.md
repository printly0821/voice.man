---
id: SPEC-FORENSIC-WEB-001
version: "1.0.0"
status: "planned"
created: "2026-01-10"
updated: "2026-01-10"
author: "지니"
priority: "HIGH"
---

# 구현 계획: SPEC-FORENSIC-WEB-001

## 개요

웹 기반 포렌식 증거 프레젠테이션 시스템의 구현 계획서입니다.

---

## 마일스톤

### 1차 목표 (Primary Goal): 핵심 인프라 구축

#### Phase 1.1: 백엔드 인증 시스템
- JWT 인증 모듈 구현 (`src/voice_man/security/jwt.py`)
- 사용자 모델 및 RBAC 구현 (`src/voice_man/models/web/user.py`)
- 로그인/로그아웃 API (`src/voice_man/api/web/auth.py`)
- 감사 로그 미들웨어 (`src/voice_man/api/middleware/audit.py`)

#### Phase 1.2: 프론트엔드 인증
- Next.js 16 프로젝트 초기 설정 (`frontend/`)
- 로그인 페이지 및 폼 (`frontend/app/login/`)
- 인증 컨텍스트 및 훅 (`frontend/hooks/useAuth.ts`)
- 보호 라우트 컴포넌트 (`frontend/components/auth/`)

#### Phase 1.3: 데이터베이스 스키마
- PostgreSQL 스키마 설계 (고소장, 혐의, 증거, 사용자)
- SQLAlchemy 모델 정의
- 마이그레이션 스크립트

**완료 기준**:
- 로그인/로그아웃 동작
- JWT 토큰 발급 및 검증
- 역할별 API 접근 제어 동작

---

### 2차 목표 (Secondary Goal): 고소장 및 증거 뷰어

#### Phase 2.1: 고소장 API
- 고소장 모델 (`src/voice_man/models/web/complaint.py`)
- 고소장 CRUD API (`src/voice_man/api/web/complaints.py`)
- 혐의-증거 매핑 API (`src/voice_man/api/web/evidences.py`)

#### Phase 2.2: 인터랙티브 고소장 뷰어
- 고소장 목록 페이지 (`frontend/app/complaints/page.tsx`)
- 고소장 상세 뷰어 (`frontend/app/complaints/[id]/page.tsx`)
- 혐의 카드 컴포넌트 (`frontend/components/complaint/ChargeCard.tsx`)
- 증거 링크 컴포넌트 (`frontend/components/complaint/EvidenceLink.tsx`)

#### Phase 2.3: 증거 탐색기
- 증거 목록 및 필터링 (`frontend/app/evidence/page.tsx`)
- 검색 필터 컴포넌트 (`frontend/components/evidence/EvidenceFilter.tsx`)
- 증거 상세 페이지 (`frontend/app/evidence/[id]/page.tsx`)

**완료 기준**:
- 고소장 텍스트 표시 및 혐의별 섹션 구분
- 혐의 클릭 시 매핑된 증거 로드
- 증거 필터링 및 검색 동작

---

### 3차 목표 (Tertiary Goal): 시각화 및 오디오 재생

#### Phase 3.1: 타임라인 시각화
- 타임라인 API (`src/voice_man/api/web/forensic.py`)
- 타임라인 페이지 (`frontend/app/timeline/page.tsx`)
- 타임라인 컴포넌트 (Chart.js/Recharts) (`frontend/components/visualization/Timeline.tsx`)

#### Phase 3.2: 포렌식 분석 그래프
- F0 그래프 컴포넌트 (`frontend/components/visualization/F0Chart.tsx`)
- 포렌식 레이더 차트 (`frontend/components/visualization/ForensicRadar.tsx`)
- Jitter/Shimmer 시각화

#### Phase 3.3: 오디오 플레이어
- Wavesurfer.js 통합 (`frontend/components/evidence/AudioPlayer.tsx`)
- 타임스탬프 기반 시점 이동
- 파형 시각화 및 전사 텍스트 동기화

**완료 기준**:
- 6개월 에스컬레이션 타임라인 시각화
- 포렌식 점수 그래프 표시
- 오디오 재생 및 파형 표시

---

### 4차 목표 (Final Goal): 보고서 및 보안 강화

#### Phase 4.1: 보고서 생성
- 보고서 생성 API (`src/voice_man/api/web/reports.py`)
- 비동기 작업 큐 (Redis + Celery 또는 ARQ)
- 보고서 생성 페이지 (`frontend/app/reports/generate/page.tsx`)
- 보고서 다운로드 및 상태 조회

#### Phase 4.2: AES-256 암호화
- 암호화 모듈 (`src/voice_man/security/encryption.py`)
- 민감 데이터 암호화 적용
- 키 관리 및 로테이션

#### Phase 4.3: 증거 비교 대시보드
- 텍스트-음성 비교 API
- 분할 뷰 컴포넌트
- 불일치 강조 기능

**완료 기준**:
- PDF/HTML 보고서 생성 및 다운로드
- 민감 데이터 AES-256 암호화
- 텍스트-포렌식 비교 대시보드

---

### 선택 목표 (Optional Goal): 고급 기능

#### Phase 5.1: 다국어 지원
- i18n 설정 (next-intl 또는 next-i18next)
- 한국어/영어 번역 파일
- 언어 전환 UI

#### Phase 5.2: 모바일 최적화
- 반응형 레이아웃
- 터치 친화적 인터랙션
- 모바일 네비게이션

#### Phase 5.3: 실시간 협업
- WebSocket 연결
- 실시간 주석 공유
- 동시 편집 표시

**완료 기준**:
- 영어 인터페이스 제공
- 모바일 기기 지원
- 실시간 협업 기능

---

## 기술적 접근 방식

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      사용자 (브라우저)                        │
└─────────────────────────────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Next.js 16 Frontend                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │고소장뷰어│  │증거탐색기│  │타임라인 │  │보고서   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│  ┌────────────────────────────────────────────────┐        │
│  │              인증 컨텍스트 / 상태 관리           │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │ REST API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Auth    │  │Complaint│  │Evidence │  │ Report  │        │
│  │ API     │  │  API    │  │  API    │  │  API    │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│  ┌────────────────────────────────────────────────┐        │
│  │       JWT 검증 / RBAC / 감사 로그 미들웨어       │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌───────────┐   ┌───────────┐   ┌───────────┐
       │PostgreSQL │   │   Redis   │   │   Files   │
       │ (데이터)  │   │(세션/캐시)│   │ (오디오)  │
       └───────────┘   └───────────┘   └───────────┘
```

### 프론트엔드 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| Next.js | 16.x | App Router, Server Components |
| React | 19.x | UI 라이브러리 |
| TypeScript | 5.9+ | 타입 안전성 |
| Tailwind CSS | 4.x | 스타일링 |
| shadcn/ui | latest | UI 컴포넌트 |
| TanStack Query | 5.x | 서버 상태 관리 |
| Chart.js / Recharts | latest | 시각화 |
| Wavesurfer.js | 7.x | 오디오 파형 |
| Zustand | 5.x | 클라이언트 상태 |

### 백엔드 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| FastAPI | 0.115+ | REST API |
| Pydantic | 2.9+ | 데이터 검증 |
| SQLAlchemy | 2.0+ | ORM (비동기) |
| PostgreSQL | 17+ | 데이터베이스 |
| Redis | 7+ | 세션/캐싱 |
| PyJWT | 2.x | JWT 토큰 |
| bcrypt | latest | 비밀번호 해싱 |
| cryptography | latest | AES-256 암호화 |
| Celery / ARQ | latest | 비동기 작업 |

### 인증 흐름

```
1. 로그인 요청
   POST /api/v1/auth/login
   { email, password }

2. 서버 검증
   - 비밀번호 bcrypt 검증
   - Access Token (4h) + Refresh Token (7d) 발급

3. 응답
   - Access Token: Response body
   - Refresh Token: HTTP-only cookie

4. API 요청
   Authorization: Bearer {access_token}

5. 토큰 갱신
   POST /api/v1/auth/refresh
   (Refresh Token from cookie)
```

### 데이터베이스 스키마

```sql
-- 사용자
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL, -- 'lawyer', 'client', 'expert'
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- 고소장
CREATE TABLE complaints (
    id UUID PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    defendant VARCHAR(100) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 혐의
CREATE TABLE charges (
    id UUID PRIMARY KEY,
    complaint_id UUID REFERENCES complaints(id),
    charge_type VARCHAR(50) NOT NULL,
    legal_basis VARCHAR(100),
    description TEXT,
    display_order INT DEFAULT 0
);

-- 증거
CREATE TABLE evidences (
    id UUID PRIMARY KEY,
    audio_file_id UUID,
    forensic_report_id UUID,
    start_time FLOAT,
    end_time FLOAT,
    transcript_excerpt_encrypted BYTEA, -- AES-256
    confidence_score FLOAT,
    evidence_type VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- 증거-혐의 매핑 (다대다)
CREATE TABLE charge_evidence_mapping (
    charge_id UUID REFERENCES charges(id),
    evidence_id UUID REFERENCES evidences(id),
    PRIMARY KEY (charge_id, evidence_id),
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- 감사 로그
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    ip_address VARCHAR(45),
    user_agent TEXT,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 위험 요소 및 대응 방안

### 기술적 위험

| 위험 | 영향도 | 대응 방안 |
|------|--------|-----------|
| Next.js 16 + React 19 호환성 | 중간 | 안정 버전 사용, 점진적 마이그레이션 |
| 대용량 오디오 스트리밍 지연 | 높음 | CDN 활용, 청크 스트리밍 |
| JWT 토큰 탈취 | 높음 | 짧은 만료 시간, Refresh Token 회전 |
| 데이터 암호화 성능 | 중간 | 필수 필드만 암호화, 캐싱 활용 |

### 보안 위험

| 위험 | 영향도 | 대응 방안 |
|------|--------|-----------|
| XSS 공격 | 높음 | CSP 헤더, 입력 검증, React 자동 이스케이프 |
| CSRF 공격 | 중간 | SameSite 쿠키, CSRF 토큰 |
| SQL Injection | 높음 | SQLAlchemy ORM, 파라미터화 쿼리 |
| 권한 상승 | 높음 | 백엔드 RBAC 필수 검증 |

---

## 의존성

### 선행 조건

- SPEC-FORENSIC-001 분석 결과물 완료
- 183개 오디오 파일 접근 가능
- 104개 HTML 보고서 및 15개 JSON 파일

### 외부 의존성

- PostgreSQL 17 데이터베이스 서버
- Redis 7 캐싱 서버
- SSL 인증서 (HTTPS)

---

## 추적성 태그

```
[SPEC-FORENSIC-WEB-001] [plan.md]
[Next.js] [FastAPI] [JWT] [RBAC] [AES-256]
[고소장뷰어] [증거탐색기] [타임라인] [보고서생성]
```

---

**문서 끝**
