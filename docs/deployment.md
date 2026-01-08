# 배포 가이드

Voice Man 시스템을 프로덕션 환경에 배포하는 방법을 안내합니다.

## 사전 요구사항

### 시스템 요구사항

**최소 사양**:
- CPU: 4코어 이상
- RAM: 16GB 이상
- 저장 공간: 100GB 이상 SSD
- GPU: NVIDIA RTX 3080 이상 (권장)

**운영체제**:
- Ubuntu 22.04 LTS (권장)
- Debian 12+
- macOS 13+ (개발 환경)

### 소프트웨어 요구사항

```bash
# Python 3.13+
python3 --version

# FFmpeg 6.0+
ffmpeg -version

# Git
git --version

# NVIDIA CUDA (GPU 사용 시)
nvidia-smi
```

## 로컬 배포 (개발 환경)

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/voice.man.git
cd voice.man
```

### 2. 가상 환경 설정

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -e ".[dev]"
```

### 4. 환경 변수 설정

```bash
# .env 파일 생성
cat > .env << EOF
# Database
DATABASE_URL=sqlite:///./data/voice_man.db

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Storage
UPLOAD_DIR=./data/uploads

# GPU
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
EOF
```

### 5. 데이터베이스 초기화

```bash
alembic upgrade head
```

### 6. 서버 시작

```bash
uvicorn voice_man.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker 배포

### 1. Dockerfile 생성

```dockerfile
FROM python:3.13-slim

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 복사 및 설치
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# 애플리케이션 복사
COPY ./src /app/src
COPY ./alembic /app/alembic
COPY ./alembic.ini /app/

# 데이터베이스 초기화
RUN alembic upgrade head

# 포트 노출
EXPOSE 8000

# 서버 시작
CMD ["uvicorn", "voice_man.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose 생성

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/voice_man
      - CUDA_VISIBLE_DEVICES=0
    depends_on:
      - db
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:16
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=voice_man
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### 3. 컨테이너 실행

```bash
docker-compose up -d
```

## 클라우드 배포

### AWS 배포

#### 1. EC2 인스턴스 설정

```bash
# 인스턴스 유형: g4dn.xlarge (NVIDIA T4 GPU)
# AMI: Ubuntu 22.04 LTS
# 저장소: 100GB GP3 SSD

# 인스턴스 접속
ssh -i key.pem ubuntu@ec2-xxx.compute.amazonaws.com
```

#### 2. Docker 설치

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
```

#### 3. NVIDIA Docker 설정

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 4. 애플리케이션 배포

```bash
git clone https://github.com/yourusername/voice.man.git
cd voice.man
docker-compose up -d
```

#### 5. Nginx 리버스 프록시

```nginx
# /etc/nginx/sites-available/voice_man
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 파일 업로드 크기 제한
        client_max_body_size 500M;
    }
}
```

### Google Cloud Platform 배포

#### 1. Compute Engine VM 생성

```bash
# 머신 유형: n1-standard-4 + NVIDIA Tesla T4
# 부팅 디스크: Ubuntu 22.04 LTS, 100GB

gcloud compute instances create voice-man-api \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd
```

#### 2. 배포 스크립트 실행

```bash
gcloud compute ssh voice-man-api --zone=us-central1-a --command="
  git clone https://github.com/yourusername/voice.man.git
  cd voice.man
  docker-compose up -d
"
```

### Azure 배포

#### 1. Azure Kubernetes Service (AKS) 생성

```bash
# 리소스 그룹 생성
az group create --name voice-man-rg --location eastus

# AKS 클러스터 생성 (GPU 노드)
az aks create \
  --resource-group voice-man-rg \
  --name voice-man-aks \
  --node-vm-size Standard_NC4as_T4_v3 \
  --node-count 2 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5

# 자격 증명 가져오기
az aks get-credentials --resource-group voice-man-rg --name voice-man-aks
```

#### 2. Kubernetes 배포

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-man
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-man
  template:
    metadata:
      labels:
        app: voice-man
    spec:
      containers:
      - name: voice-man
        image: gcr.io/your-project/voice-man:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: url
---
apiVersion: v1
kind: Service
metadata:
  name: voice-man-service
spec:
  selector:
    app: voice-man
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

```bash
kubectl apply -f k8s/deployment.yaml
```

## 데이터베이스 마이그레이션

### SQLite → PostgreSQL

```bash
# 1. PostgreSQL 덤프 생성
pg_dump -U user -h localhost voice_man > schema.sql

# 2. SQLite 데이터 내보내기
sqlite3 data/voice_man.db .dump > data.sql

# 3. PostgreSQL로 가져오기
psql -U user -h localhost voice_man < data.sql

# 4. Alembic 마이그레이션
alembic upgrade head
```

## 모니터링 및 로깅

### 1. Prometheus 메트릭

```python
# main.py에 추가
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

### 2. 로그 집계

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### 3. 헬스체크

```bash
# curl로 헬스체크
curl http://localhost:8000/health

# 또는 워치독 설정
watch -n 5 'curl http://localhost:8000/health'
```

## 보안 설정

### 1. HTTPS 설정 (Let's Encrypt)

```bash
# Certbot 설치
sudo apt-get install certbot python3-certbot-nginx

# 인증서 발급
sudo certbot --nginx -d api.example.com

# 자동 갱신
sudo certbot renew --dry-run
```

### 2. 방화벽 설정

```bash
# UFW 활성화
sudo ufw enable

# 허용된 포트만 열기
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
```

### 3. JWT 인증 (Phase 2)

```python
# main.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    token = credentials.credentials
    # 토큰 검증 로직
    return token
```

## 백업 및 복구

### 1. 데이터베이스 백업

```bash
# 매일 백업 스크립트
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U user voice_man > backups/backup_$DATE.sql

# 7일 이상 된 백업 삭제
find backups/ -name "backup_*.sql" -mtime +7 -delete
```

### 2. 파일 백업

```bash
# 오디오 파일 백업
rsync -avz data/uploads/ backups/uploads_$(date +%Y%m%d)/
```

### 3. 복구 절차

```bash
# 데이터베이스 복구
psql -U user voice_man < backups/backup_20260108.sql

# 파일 복구
rsync -avz backups/uploads_20260108/ data/uploads/
```

## 성능 최적화

### 1. 캐싱

```python
# Redis 캐싱
from redis import Redis

redis = Redis(host='localhost', port=6379, decode_responses=True)

# 캐시 저장
redis.set(f'transcript:{file_id}', transcript_data, ex=3600)

# 캐시 조회
cached = redis.get(f'transcript:{file_id}')
```

### 2. 로드 밸런싱

```bash
# Nginx 로드 밸런싱
upstream voice_man_backend {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    location / {
        proxy_pass http://voice_man_backend;
    }
}
```

### 3. CDN 설정

```bash
# CloudFlare CDN
# 1. 정적 파일 CDN 업로드
# 2. 도메인 설정
# 3. 캐시 규칙 구성
```

## 트러블슈팅

### 일반적인 문제

1. **GPU 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   # 또는 CPU 모드 사용
   CUDA_VISIBLE_DEVICES="" uvicorn voice_man.main:app
   ```

2. **파일 업로드 실패**
   ```bash
   # 파일 크기 제한 확인
   # Nginx: client_max_body_size
   # FastAPI: UploadFile 제한
   ```

3. **데이터베이스 연결 실패**
   ```bash
   # PostgreSQL 상태 확인
   sudo systemctl status postgresql

   # 연결 테스트
   psql -U user -h localhost voice_man
   ```

### 로그 확인

```bash
# 애플리케이션 로그
tail -f logs/app.log

# Docker 로그
docker-compose logs -f app

# Kubernetes 로그
kubectl logs -f deployment/voice-man
```

---

**관련 문서**:
- [아키텍처](architecture.md)
- [API 레퍼런스](api-reference.md)
- [개발 가이드](development.md)
