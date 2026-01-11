---
id: SPEC-FORENSIC-WEB-001
version: "1.0.0"
status: "planned"
created: "2026-01-10"
updated: "2026-01-10"
author: "지니"
priority: "HIGH"
---

# 인수 기준: SPEC-FORENSIC-WEB-001

## 개요

웹 기반 포렌식 증거 프레젠테이션 시스템의 인수 기준 및 테스트 시나리오입니다.

---

## 품질 게이트 기준

### TRUST 5 기준

| 기준 | 목표 | 측정 방법 |
|------|------|-----------|
| Test-first | 85% 이상 | pytest coverage, jest coverage |
| Readable | ruff/eslint 통과 | 자동화 검사 |
| Unified | 일관된 코드 스타일 | black, prettier 적용 |
| Secured | OWASP 준수 | 보안 테스트, 취약점 스캔 |
| Trackable | 구조화된 커밋 | Conventional Commits |

### 성능 기준

| 항목 | 목표 | 측정 방법 |
|------|------|-----------|
| 페이지 로드 | P95 < 2초 | Lighthouse, Web Vitals |
| API 응답 | P95 < 500ms | API 모니터링 |
| 오디오 버퍼링 | 초기 < 1초 | 실제 테스트 |
| 동시 접속자 | 50명 | 부하 테스트 |

---

## 인증 및 보안 테스트

### AC-AUTH-001: 로그인 성공

**Given** 유효한 이메일과 비밀번호를 가진 사용자가 있을 때
**When** 로그인 API에 올바른 자격 증명으로 요청하면
**Then** Access Token과 Refresh Token이 발급되어야 한다
**And** Access Token은 응답 본문에 포함되어야 한다
**And** Refresh Token은 HTTP-only 쿠키로 설정되어야 한다

```python
def test_login_success():
    response = client.post("/api/v1/auth/login", json={
        "email": "lawyer@example.com",
        "password": "ValidPassword123!"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "refresh_token" not in response.json()  # 쿠키로만
    assert response.cookies.get("refresh_token") is not None
```

---

### AC-AUTH-002: 로그인 실패 - 잘못된 비밀번호

**Given** 등록된 이메일을 가진 사용자가 있을 때
**When** 잘못된 비밀번호로 로그인 요청하면
**Then** 401 Unauthorized 응답이 반환되어야 한다
**And** 실패 시도가 감사 로그에 기록되어야 한다

```python
def test_login_wrong_password():
    response = client.post("/api/v1/auth/login", json={
        "email": "lawyer@example.com",
        "password": "WrongPassword"
    })
    assert response.status_code == 401
    assert "Invalid credentials" in response.json()["detail"]
```

---

### AC-AUTH-003: 로그인 시도 제한

**Given** 5회 연속 로그인 실패한 사용자가 있을 때
**When** 6번째 로그인을 시도하면
**Then** 429 Too Many Requests 응답이 반환되어야 한다
**And** 15분 후 다시 시도 가능해야 한다

```python
def test_login_rate_limit():
    for _ in range(5):
        client.post("/api/v1/auth/login", json={
            "email": "user@example.com",
            "password": "wrong"
        })
    response = client.post("/api/v1/auth/login", json={
        "email": "user@example.com",
        "password": "wrong"
    })
    assert response.status_code == 429
```

---

### AC-AUTH-004: JWT 토큰 검증

**Given** 유효한 Access Token이 있을 때
**When** 보호된 API에 Authorization 헤더와 함께 요청하면
**Then** 정상 응답이 반환되어야 한다

```python
def test_protected_route_with_valid_token():
    token = get_valid_access_token()
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "email" in response.json()
```

---

### AC-AUTH-005: 만료된 토큰 거부

**Given** 만료된 Access Token이 있을 때
**When** 보호된 API에 요청하면
**Then** 401 Unauthorized 응답이 반환되어야 한다
**And** "Token expired" 메시지가 포함되어야 한다

```python
def test_expired_token_rejected():
    expired_token = create_expired_token()
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()
```

---

### AC-AUTH-006: 역할 기반 접근 제어 - 변호사

**Given** "lawyer" 역할을 가진 사용자가 로그인했을 때
**When** 사용자 관리 API에 접근하면
**Then** 정상 응답이 반환되어야 한다

```python
def test_lawyer_can_access_user_management():
    token = login_as("lawyer@example.com")
    response = client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

---

### AC-AUTH-007: 역할 기반 접근 제어 - 의뢰인 제한

**Given** "client" 역할을 가진 사용자가 로그인했을 때
**When** 증거 매핑 수정 API에 접근하면
**Then** 403 Forbidden 응답이 반환되어야 한다

```python
def test_client_cannot_modify_evidence():
    token = login_as("client@example.com")
    response = client.put(
        "/api/v1/evidences/mapping/123",
        headers={"Authorization": f"Bearer {token}"},
        json={"charge_id": "456"}
    )
    assert response.status_code == 403
```

---

## 고소장 뷰어 테스트

### AC-COMPLAINT-001: 고소장 목록 조회

**Given** 두 개의 고소장(신기연, 신동식)이 등록되어 있을 때
**When** 고소장 목록 API를 호출하면
**Then** 두 개의 고소장이 반환되어야 한다
**And** 각 고소장에 혐의 개수가 표시되어야 한다

```python
def test_list_complaints():
    response = client.get("/api/v1/complaints")
    assert response.status_code == 200
    complaints = response.json()
    assert len(complaints) == 2
    assert any(c["title"] == "신기연 고소장" for c in complaints)
    assert any(c["title"] == "신동식 고소장" for c in complaints)
```

---

### AC-COMPLAINT-002: 고소장 상세 조회

**Given** 신기연 고소장이 있을 때
**When** 고소장 상세 API를 호출하면
**Then** 8개 혐의가 포함되어야 한다
**And** 각 혐의에 법적 근거가 포함되어야 한다

```python
def test_get_complaint_detail():
    response = client.get(f"/api/v1/complaints/{shinkiyeon_id}")
    assert response.status_code == 200
    complaint = response.json()
    assert complaint["title"] == "신기연 고소장"
    assert len(complaint["charges"]) == 8
    assert any(c["charge_type"] == "fraud" for c in complaint["charges"])
```

---

### AC-COMPLAINT-003: 혐의별 증거 로드

**Given** "사기" 혐의에 5개의 증거가 매핑되어 있을 때
**When** 해당 혐의의 증거 목록을 요청하면
**Then** 5개의 증거가 반환되어야 한다
**And** 각 증거에 신뢰도 점수가 포함되어야 한다

```python
def test_load_evidences_for_charge():
    response = client.get(f"/api/v1/complaints/{id}/charges/{fraud_charge_id}")
    assert response.status_code == 200
    evidences = response.json()["evidences"]
    assert len(evidences) == 5
    for evidence in evidences:
        assert "confidence_score" in evidence
```

---

## 증거 탐색기 테스트

### AC-EVIDENCE-001: 포렌식 점수 필터링

**Given** 183개 증거 파일이 있을 때
**When** 가스라이팅 점수 70 이상으로 필터링하면
**Then** 조건에 맞는 증거만 반환되어야 한다
**And** 모든 결과의 gaslighting_score >= 70이어야 한다

```python
def test_filter_by_gaslighting_score():
    response = client.get(
        "/api/v1/evidences",
        params={"gaslighting_score_min": 70}
    )
    assert response.status_code == 200
    for evidence in response.json()["items"]:
        assert evidence["forensic_report"]["gaslighting_score"] >= 70
```

---

### AC-EVIDENCE-002: 날짜 범위 필터링

**Given** 2025년 7월~2026년 1월 녹취 파일이 있을 때
**When** 2025년 10월~12월로 필터링하면
**Then** 해당 기간의 증거만 반환되어야 한다

```python
def test_filter_by_date_range():
    response = client.get(
        "/api/v1/evidences",
        params={
            "date_from": "2025-10-01",
            "date_to": "2025-12-31"
        }
    )
    assert response.status_code == 200
    for evidence in response.json()["items"]:
        date = evidence["recorded_date"]
        assert "2025-10" <= date <= "2025-12-31"
```

---

### AC-EVIDENCE-003: 복합 필터링

**Given** 다양한 조건의 증거가 있을 때
**When** 가스라이팅 점수 60 이상 AND 화자 "SPEAKER_00"으로 필터링하면
**Then** 두 조건을 모두 만족하는 증거만 반환되어야 한다

```python
def test_combined_filter():
    response = client.get(
        "/api/v1/evidences",
        params={
            "gaslighting_score_min": 60,
            "speaker": "SPEAKER_00"
        }
    )
    assert response.status_code == 200
    for evidence in response.json()["items"]:
        assert evidence["forensic_report"]["gaslighting_score"] >= 60
        assert evidence["speaker"] == "SPEAKER_00"
```

---

### AC-EVIDENCE-004: 필터링 응답 시간

**Given** 183개 증거 파일이 있을 때
**When** 필터 조건을 변경하면
**Then** 500ms 이내에 결과가 반환되어야 한다

```python
def test_filter_response_time():
    import time
    start = time.time()
    response = client.get(
        "/api/v1/evidences",
        params={"gaslighting_score_min": 50}
    )
    elapsed = time.time() - start
    assert response.status_code == 200
    assert elapsed < 0.5  # 500ms
```

---

## 오디오 재생 테스트

### AC-AUDIO-001: 오디오 스트리밍

**Given** 증거 오디오 파일이 있을 때
**When** 오디오 스트리밍 API를 호출하면
**Then** audio/mpeg 또는 audio/wav 형식으로 스트리밍되어야 한다
**And** Range 요청을 지원해야 한다

```python
def test_audio_streaming():
    response = client.get(
        f"/api/v1/evidences/{evidence_id}/audio",
        headers={"Range": "bytes=0-1023"}
    )
    assert response.status_code in [200, 206]
    assert response.headers["Content-Type"].startswith("audio/")
```

---

### AC-AUDIO-002: 타임스탬프 기반 시점 이동

**Given** 오디오 플레이어가 로드되어 있을 때
**When** 특정 증거 구간(10초~15초)을 클릭하면
**Then** 오디오가 10초 지점으로 이동해야 한다
**And** 재생이 시작되어야 한다

```typescript
test('audio seeks to evidence timestamp', async () => {
  render(<AudioPlayer audioId={testId} />);

  // 증거 클릭 시뮬레이션
  fireEvent.click(screen.getByTestId('evidence-marker-10s'));

  const audioElement = screen.getByTestId('audio-element');
  expect(audioElement.currentTime).toBe(10);
  expect(audioElement.paused).toBe(false);
});
```

---

## 타임라인 시각화 테스트

### AC-TIMELINE-001: 에스컬레이션 패턴 표시

**Given** 6개월간의 녹취 데이터가 있을 때
**When** 타임라인 페이지에 접근하면
**Then** 시간순으로 정렬된 데이터 포인트가 표시되어야 한다
**And** 각 포인트에 심리 압박 강도가 표시되어야 한다

```python
def test_timeline_data():
    response = client.get("/api/v1/forensic/timeline")
    assert response.status_code == 200
    timeline = response.json()
    dates = [item["date"] for item in timeline["data_points"]]
    assert dates == sorted(dates)  # 시간순 정렬
    for point in timeline["data_points"]:
        assert "pressure_index" in point
```

---

### AC-TIMELINE-002: 패턴 유형별 색상 코드

**Given** 타임라인이 렌더링되었을 때
**When** 가스라이팅 패턴 마커를 확인하면
**Then** 보라색(#8B5CF6)으로 표시되어야 한다

```typescript
test('gaslighting markers are purple', () => {
  render(<Timeline data={mockData} />);

  const gaslightingMarkers = screen.getAllByTestId('marker-gaslighting');
  gaslightingMarkers.forEach(marker => {
    expect(marker).toHaveStyle({ backgroundColor: '#8B5CF6' });
  });
});
```

---

## 보고서 생성 테스트

### AC-REPORT-001: 비동기 보고서 생성

**Given** 보고서 생성 권한이 있는 사용자가 있을 때
**When** 보고서 생성을 요청하면
**Then** 작업 ID가 반환되어야 한다
**And** 상태 조회 API로 진행률을 확인할 수 있어야 한다

```python
def test_async_report_generation():
    response = client.post(
        "/api/v1/reports/generate",
        json={
            "report_type": "charge_summary",
            "charge_ids": [charge_id_1, charge_id_2],
            "format": "pdf"
        }
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    status_response = client.get(f"/api/v1/reports/{task_id}/status")
    assert status_response.status_code == 200
    assert "progress" in status_response.json()
```

---

### AC-REPORT-002: 보고서 다운로드

**Given** 보고서 생성이 완료되었을 때
**When** 다운로드 API를 호출하면
**Then** PDF 파일이 다운로드되어야 한다

```python
def test_report_download():
    # 보고서 생성 완료 후
    response = client.get(f"/api/v1/reports/{completed_task_id}/download")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/pdf"
    assert "Content-Disposition" in response.headers
```

---

## 암호화 테스트

### AC-ENCRYPT-001: 민감 데이터 암호화 저장

**Given** 전사 텍스트가 포함된 증거가 있을 때
**When** 데이터베이스에 저장되면
**Then** transcript_excerpt 필드는 암호화되어야 한다
**And** 직접 DB 조회 시 평문이 노출되지 않아야 한다

```python
def test_transcript_encrypted_in_db():
    # DB 직접 조회
    result = db.execute(
        "SELECT transcript_excerpt_encrypted FROM evidences WHERE id = %s",
        [evidence_id]
    )
    encrypted_data = result.fetchone()[0]

    # 평문 텍스트가 포함되어 있지 않아야 함
    assert b"가스라이팅" not in encrypted_data
```

---

### AC-ENCRYPT-002: 권한 있는 복호화

**Given** 암호화된 증거 데이터가 있을 때
**When** 권한 있는 사용자가 API로 조회하면
**Then** 복호화된 전사 텍스트가 반환되어야 한다

```python
def test_authorized_decryption():
    token = login_as("lawyer@example.com")
    response = client.get(
        f"/api/v1/evidences/{evidence_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "가스라이팅" in response.json()["transcript_excerpt"]
```

---

## 감사 로그 테스트

### AC-AUDIT-001: 접근 기록 저장

**Given** 사용자가 증거를 조회할 때
**When** 조회 API가 호출되면
**Then** 감사 로그에 기록이 저장되어야 한다

```python
def test_audit_log_created():
    initial_count = get_audit_log_count()

    token = login_as("lawyer@example.com")
    client.get(
        f"/api/v1/evidences/{evidence_id}",
        headers={"Authorization": f"Bearer {token}"}
    )

    new_count = get_audit_log_count()
    assert new_count == initial_count + 1

    latest_log = get_latest_audit_log()
    assert latest_log["action"] == "view"
    assert latest_log["resource_type"] == "evidence"
    assert latest_log["resource_id"] == evidence_id
```

---

## UI/UX 테스트

### AC-UI-001: 반응형 레이아웃

**Given** 데스크톱 브라우저(1920px)에서 접속할 때
**When** 고소장 뷰어를 열면
**Then** 좌측에 고소장 텍스트, 우측에 증거 패널이 표시되어야 한다

```typescript
test('desktop layout shows side-by-side panels', () => {
  render(<ComplaintViewer complaintId={testId} />, {
    viewport: { width: 1920, height: 1080 }
  });

  expect(screen.getByTestId('complaint-panel')).toBeVisible();
  expect(screen.getByTestId('evidence-panel')).toBeVisible();
});
```

---

### AC-UI-002: 로딩 상태 표시

**Given** 데이터 로딩 중일 때
**When** 페이지가 렌더링되면
**Then** 스켈레톤 UI 또는 로딩 스피너가 표시되어야 한다

```typescript
test('shows loading state', () => {
  // 느린 네트워크 시뮬레이션
  server.use(
    rest.get('/api/v1/complaints', (req, res, ctx) => {
      return res(ctx.delay(1000), ctx.json(mockData));
    })
  );

  render(<ComplaintList />);
  expect(screen.getByTestId('loading-skeleton')).toBeVisible();
});
```

---

## 정의 완료 (Definition of Done)

### 기능 완료 기준

- [ ] 모든 API 엔드포인트 구현 완료
- [ ] 프론트엔드 모든 페이지 구현 완료
- [ ] 단위 테스트 커버리지 85% 이상
- [ ] 통합 테스트 통과
- [ ] E2E 테스트 주요 시나리오 통과

### 보안 완료 기준

- [ ] JWT 인증 동작 확인
- [ ] RBAC 모든 역할 테스트 통과
- [ ] AES-256 암호화 적용 확인
- [ ] HTTPS 적용 확인
- [ ] OWASP Top 10 취약점 점검 통과

### 성능 완료 기준

- [ ] 페이지 로드 P95 < 2초
- [ ] API 응답 P95 < 500ms
- [ ] Lighthouse 성능 점수 80점 이상

### 문서화 완료 기준

- [ ] API 문서화 (OpenAPI/Swagger)
- [ ] 사용자 가이드 작성
- [ ] 관리자 가이드 작성

---

## 추적성 태그

```
[SPEC-FORENSIC-WEB-001] [acceptance.md]
[테스트시나리오] [인수기준] [품질게이트]
[인증테스트] [보안테스트] [성능테스트]
```

---

**문서 끝**
