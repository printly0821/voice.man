#!/bin/bash

# 포렌식 증거 분석 실행 스크립트
# 형사소송용 음성 증거 분석 자동화

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 헤더 출력
clear
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🔍 포렌식 증거 분석 시스템 - 형사소송 증거자료 생성${NC}         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 환경 확인
log_info "시스템 환경 확인 중..."

# Python 버전 확인
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python 버전: $PYTHON_VERSION"

# GPU 확인
log_info "GPU 가용성 확인..."
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}' )" 2>/dev/null || log_warning "GPU 정보 조회 실패"

# 필수 디렉토리 생성
log_info "필수 디렉토리 생성..."
mkdir -p logs
mkdir -p forensic_evidence_results
mkdir -p forensic_evidence_results/batch_reports
mkdir -p forensic_evidence_results/evidence_files

log_success "환경 준비 완료"
echo ""

# 분석 시작
log_info "═════════════════════════════════════════════════════════════════"
log_info "포렌식 증거 분석 시작"
log_info "분석 대상: 184개 통화 녹음 파일"
log_info "배치 크기: 10개 파일 단위"
log_info "모니터링: 성능, 온도, 리소스"
log_info "═════════════════════════════════════════════════════════════════"
echo ""

# 분석 실행 (백그라운드)
log_info "분석 엔진 시작..."
python3 scripts/forensic_evidence_analyzer.py > logs/analysis_output.log 2>&1 &
ANALYSIS_PID=$!

log_success "분석 프로세스 시작 (PID: $ANALYSIS_PID)"
log_info "로그 파일: logs/forensic_evidence_analysis.log"
echo ""

# 모니터링 대시보드 시작
log_info "모니터링 대시보드 시작..."
echo ""

# 백그라운드 프로세스 모니터링
MONITOR_COUNT=0
while kill -0 $ANALYSIS_PID 2>/dev/null; do
    MONITOR_COUNT=$((MONITOR_COUNT + 1))

    # 매 12번(약 1분)마다 진행 상황 출력
    if [ $((MONITOR_COUNT % 12)) -eq 0 ]; then
        ELAPSED=$((MONITOR_COUNT / 2))
        MINUTES=$((ELAPSED / 60))
        SECONDS=$((ELAPSED % 60))

        # 시스템 메트릭
        CPU=$(ps aux | grep -v grep | grep forensic_evidence_analyzer | awk '{sum+=$3} END {print sum}')
        MEM=$(ps aux | grep -v grep | grep forensic_evidence_analyzer | awk '{sum+=$4} END {print sum}')

        log_info "진행 상황: ${MINUTES}분 ${SECONDS}초 경과 | CPU: ${CPU}% | MEM: ${MEM}%"

        # 온도 확인
        if command -v nvidia-smi &> /dev/null; then
            GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [ ! -z "$GPU_TEMP" ]; then
                log_info "GPU 온도: ${GPU_TEMP}°C"
            fi
        fi

        # 배치 보고서 확인
        BATCH_COUNT=$(ls forensic_evidence_results/batch_*.json 2>/dev/null | wc -l)
        if [ $BATCH_COUNT -gt 0 ]; then
            log_info "완료된 배치 보고서: $BATCH_COUNT개"
        fi
    fi

    sleep 5
done

# 분석 완료
log_success "분석 완료!"
echo ""

# 최종 결과 확인
log_info "최종 결과 확인..."

if [ -f "forensic_evidence_results/forensic_evidence_complete_analysis.json" ]; then
    log_success "분석 결과 파일 생성 완료"

    # JSON에서 통계 추출
    python3 << 'EOF'
import json
from pathlib import Path

result_file = Path("forensic_evidence_results/forensic_evidence_complete_analysis.json")
if result_file.exists():
    with open(result_file) as f:
        data = json.load(f)

    stats = data.get("statistics", {})
    print(f"\n📊 분석 결과 요약:")
    print(f"  총 파일: {stats.get('total_files_processed', 0)}개")
    print(f"  성공: {stats.get('successful_files', 0)}개 ✅")
    print(f"  실패: {stats.get('failed_files', 0)}개 ❌")
    print(f"  성공률: {stats.get('success_rate', 0):.1f}%")
    print(f"  총 처리 시간: {stats.get('total_processing_time', 0):.1f}초")
    print(f"  평균 처리 시간: {stats.get('average_per_file', 0):.2f}초/파일")
EOF
else
    log_warning "분석 결과 파일을 찾을 수 없습니다."
fi

echo ""
log_success "═════════════════════════════════════════════════════════════════"
log_success "포렌식 증거 분석 완료!"
log_success "═════════════════════════════════════════════════════════════════"
echo ""
log_info "결과 위치:"
log_info "  보고서: forensic_evidence_results/batch_*_report.json"
log_info "  전체 결과: forensic_evidence_results/forensic_evidence_complete_analysis.json"
log_info "  로그: logs/forensic_evidence_analysis.log"
echo ""
