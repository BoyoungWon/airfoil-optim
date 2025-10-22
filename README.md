# X-foil Airfoil Optimizer - Complete Project Structure

## 📁 프로젝트 디렉터리 구조

```
xfoil-optimizer/
├── README.md
├── docker-compose.yml
├── .env
├── .gitignore
│
├── backend/                    # FastAPI 백엔드
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── start.sh
│   ├── main.py                # FastAPI 메인 애플리케이션
│   ├── xfoil_wrapper.py       # X-foil Python 래퍼
│   ├── nurbs_airfoil.py       # NURBS 에어포일 클래스
│   ├── optimizer.py           # 최적화 알고리즘
│   ├── config.py              # 설정 파일
│   └── utils/
│       ├── __init__.py
│       ├── validators.py      # 검증 함수들
│       └── helpers.py         # 유틸리티 함수들
│
├── frontend/                   # React 프론트엔드
│   ├── Dockerfile
│   ├── package.json
│   ├── package-lock.json
│   ├── nginx.conf             # 프로덕션용 nginx 설정
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── manifest.json
│   └── src/
│       ├── index.js           # React 엔트리 포인트
│       ├── App.js             # 메인 애플리케이션 컴포넌트
│       ├── components/        # React 컴포넌트들
│       │   ├── XfoilConfigPanel.js
│       │   ├── NurbsControlPanel.js
│       │   ├── AirfoilVisualization.js
│       │   ├── PerformanceCharts.js
│       │   ├── OptimizationPanel.js
│       │   └── OptimizationResults.js
│       ├── hooks/             # 커스텀 훅들
│       │   ├── useWebSocket.js
│       │   └── useLocalStorage.js
│       ├── utils/             # 유틸리티 함수들
│       │   ├── api.js
│       │   └── constants.js
│       └── styles/            # 스타일 파일들
│           └── App.css
│
├── nginx/                     # 프로덕션용 리버스 프록시
│   ├── nginx.conf
│   └── ssl/                   # SSL 인증서 (옵션)
│
├── data/                      # 입력 데이터
│   ├── airfoils/              # 기본 에어포일 파일들
│   └── presets/               # 프리셋 설정들
│
├── results/                   # 결과 파일들
│   ├── optimization/          # 최적화 결과
│   └── analysis/              # 해석 결과
│
└── docs/                      # 문서
    ├── API.md                 # API 문서
    ├── DEVELOPMENT.md         # 개발 가이드
    └── USER_GUIDE.md          # 사용자 가이드
```

## 🚀 설치 및 실행 방법

### 1. Docker를 이용한 전체 시스템 실행 (권장)

```bash
# 프로젝트 클론 및 이동
git clone <repository-url>
cd xfoil-optimizer

# 환경 변수 설정 (.env 파일 생성)
cat > .env << EOF
# Backend Configuration
PYTHONPATH=/app
XFOIL_PATH=/app/xfoil6.99/bin/xfoil
CORS_ORIGINS=http://localhost:3000

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000

# Redis Configuration
REDIS_URL=redis://redis:6379
EOF

# 전체 스택 실행 (개발 모드)
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build

# 프로덕션 모드 실행
docker-compose --profile production up --build
```

### 2. 개별 서비스 실행

#### 백엔드 (FastAPI) 실행
```bash
cd backend

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# X-foil 설치 (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install gfortran build-essential
wget https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz
tar -xzf xfoil6.99.tgz
cd xfoil6.99/src
make install
cd ../..

# 백엔드 서버 시작
python main.py
```

#### 프론트엔드 (React) 실행
```bash
cd frontend

# Node.js 패키지 설치
npm install

# 개발 서버 시작
npm start

# 프로덕션 빌드
npm run build
```

## 🔧 주요 설정 파일들

### Backend 설정 (backend/config.py)
```python
import os
from pathlib import Path

# X-foil 설정
XFOIL_PATH = os.getenv('XFOIL_PATH', 'xfoil')
XFOIL_TIMEOUT = 120  # seconds

# NURBS 설정
DEFAULT_CONTROL_POINTS_UPPER = 8
DEFAULT_CONTROL_POINTS_LOWER = 8
NURBS_DEGREE = 3

# 최적화 설정
DEFAULT_POPULATION_SIZE = 20
DEFAULT_MAX_GENERATIONS = 50
MAX_OPTIMIZATION_TIME = 3600  # seconds

# 파일 경로
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
AIRFOILS_DIR = Path("airfoils")
```

### Frontend 환경변수 (.env in frontend/)
```bash
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_VERSION=2.0.0
GENERATE_SOURCEMAP=false
```

## 🛠️ 개발 환경 설정

### VS Code 설정 (.vscode/settings.json)
```json
{
  "python.defaultInterpreterPath": "./backend/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/node_modules": true
  }
}
```

### Git 설정 (.gitignore)
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.env.local
.env.development.local
.env.test.local
.env.production.local

# Build outputs
/frontend/build
/backend/dist

# Data files
/data/temp/
/results/temp/

# X-foil files
xfoil6.99.tgz
xfoil6.99/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore
```

## 📊 주요 기능 설명

### 1. X-foil 세부 파라미터 제어
- **Reynolds Number**: 50,000 ~ 10,000,000
- **Mach Number**: 0.0 ~ 0.8
- **Ncrit**: 1.0 (자연천이) ~ 12.0 (거친조건)
- **최대 반복수**: 50 ~ 500
- **점성/비점성 해석** 선택 가능

### 2. NURBS 기반 에어포일 제어
- 상면/하면 각각 독립 제어점
- 실시간 형상 검증
- 프리셋 에어포일 (NACA 계열 등)
- 제작성 제약조건 적용

### 3. 다목적 최적화
- **NSGA-II**: 빠른 비지배 정렬
- **MOEA/D**: 분해 기반 접근법
- **SPEA2**: 강도 파레토 진화 알고리즘

### 4. 실시간 시각화
- **Plotly.js** 기반 인터랙티브 차트
- **Polar Curve**: 극곡선 (Cl vs Cd)
- **Performance Charts**: 받음각별 계수 변화
- **Pareto Front**: 다목적 최적해 시각화

## 🔍 API 엔드포인트

### WebSocket 엔드포인트
- `WS /ws`: 실시간 통신 (해석, 최적화, 형상 업데이트)

### REST API 엔드포인트
- `GET /`: API 정보
- `GET /health`: 시스템 상태 확인
- `GET /api/airfoil/default`: 기본 에어포일 정보
- `POST /api/airfoil/analyze`: 에어포일 해석 실행

### WebSocket 메시지 타입
```javascript
// 클라이언트 → 서버
{
  "action": "initialize|analyze|optimize|update_control_points",
  "config": { /* X-foil/최적화 설정 */ },
  "parameters": [ /* NURBS 제어점 파라미터 */ ]
}

// 서버 → 클라이언트
{
  "type": "initialization|analysis_result|optimization_progress|shape_update|error",
  "data": { /* 결과 데이터 */ },
  "message": "상태 메시지",
  "progress": 75
}
```

## 🎯 최적화 알고리즘 설정

### NSGA-II (기본값)
```python
{
  "algorithm": "nsga2",
  "population_size": 20,
  "max_iterations": 50,
  "crossover_prob": 0.9,
  "mutation_prob": 0.1,
  "eta_crossover": 15,
  "eta_mutation": 20
}
```

### 목적함수
1. **maximize Cl**: 최대 양력계수 달성
2. **minimize Cd**: 최소 항력계수 달성
3. **minimize dCl/dCm**: 안정성 개선 (종방향 안정성)

### 제약조건
1. **두께비**: 5% ~ 25% chord
2. **제작성**: 급격한 곡률 변화 방지
3. **구조적 타당성**: 앞전/뒷전 연속성 보장

## 📈 성능 지표

### 해석 결과 지표
- **Max Cl**: 최대 양력계수
- **Min Cd**: 최소 항력계수  
- **Max L/D**: 최대 양항비
- **Cd @ Cl=1.0**: 설계점에서의 항력계수
- **Stall Angle**: 실속 받음각
- **Convergence Rate**: X-foil 수렴률

### 최적화 성능 지표
- **Pareto Solutions**: 파레토 최적해 개수
- **Convergence History**: 세대별 수렴 이력
- **Hypervolume**: 파레토 프론트 품질 지표
- **Overall Score**: 통합 성능 점수

## 🛡️ 보안 및 운영

### Docker 보안 설정
```dockerfile
# 비root 사용자 생성
RUN addgroup --system app && adduser --system --group app
USER app

# 최소한의 권한 부여
RUN chmod 755 /app/start.sh
RUN chown -R app:app /app/data /app/results
```

### Nginx 프로덕션 설정
```nginx
upstream backend {
    server xfoil-backend:8000;
}

upstream frontend {
    server xfoil-frontend:3000;
}

server {
    listen 80;
    server_name localhost;

    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket
    location /ws {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

## 🧪 테스트 및 검증

### 백엔드 테스트
```bash
cd backend

# 단위 테스트
python -m pytest tests/

# X-foil 연동 테스트
python -c "
from xfoil_wrapper import XfoilWrapper
xfoil = XfoilWrapper()
print('X-foil validation:', xfoil.validate_xfoil_installation())
"

# API 테스트
curl http://localhost:8000/health
```

### 프론트엔드 테스트
```bash
cd frontend

# 테스트 실행
npm test

# 빌드 테스트
npm run build

# 린트 검사
npm run lint
```

### 통합 테스트
```bash
# 전체 시스템 헬스체크
curl http://localhost:8000/health
curl http://localhost:3000

# WebSocket 연결 테스트
wscat -c ws://localhost:8000/ws
```

## 📝 개발 가이드

### 새로운 컴포넌트 추가
1. `frontend/src/components/` 에 React 컴포넌트 생성
2. 필요시 해당 WebSocket 메시지 타입 정의
3. `App.js`에서 컴포넌트 import 및 사용
4. CSS 스타일링 적용

### 새로운 최적화 알고리즘 추가
1. `backend/optimizer.py`에 알고리즘 클래스 구현
2. `OptimizationConfig`에 새 알고리즘 옵션 추가
3. Frontend의 `OptimizationPanel`에서 선택 옵션 추가
4. 테스트 케이스 작성

### X-foil 파라미터 확장
1. `XfoilConfig` 모델에 새 파라미터 추가
2. `xfoil_wrapper.py`에서 해당 파라미터 처리 로직 구현
3. Frontend의 `XfoilConfigPanel`에 UI 추가
4. 검증 로직 추가

## 🐛 문제 해결

### 일반적인 문제들

#### X-foil 수렴 실패
```bash
# 해결 방법:
1. Reynolds 수를 1e6 이상으로 증가
2. 받음각 범위를 줄임 (0-2도)
3. 최대 반복수 증가 (200 → 500)
4. Ncrit 값 조정 (9 → 4 또는 12)
```

#### Docker 빌드 실패
```bash
# X-foil 컴파일 오류 해결:
docker-compose build --no-cache
docker-compose up xfoil-backend

# 컨테이너 내부 디버깅:
docker exec -it xfoil-backend bash
```

#### React 핫 리로드 문제
```bash
# WSL2/Docker 환경에서:
export CHOKIDAR_USEPOLLING=true
export WATCHPACK_POLLING=true
npm start
```

#### WebSocket 연결 실패
```bash
# CORS 설정 확인:
# backend/main.py에서 allow_origins 확인
# 방화벽/프록시 설정 점검
```

## 🚀 배포 가이드

### 개발 환경
```bash
# 로컬 개발
docker-compose up --build

# 포트 확인
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs
```

### 프로덕션 배포
```bash
# 프로덕션 빌드
docker-compose --profile production up --build -d

# 스케일링
docker-compose up --scale xfoil-backend=3 -d

# 모니터링
docker-compose logs -f
```

### 클라우드 배포 (예: AWS)
```bash
# ECR 이미지 푸시
aws ecr get-login-password | docker login --username AWS --password-stdin
docker build -t xfoil-optimizer .
docker tag xfoil-optimizer:latest 123456789012.dkr.ecr.region.amazonaws.com/xfoil-optimizer:latest
docker push 123456789012.dkr.ecr.region.amazonaws.com/xfoil-optimizer:latest

# ECS 또는 EKS 배포
# Kubernetes Deployment, Service 매니페스트 작성
```

## 📊 모니터링 및 로깅

### 로그 설정
```python
# backend/config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': '/app/logs/xfoil_optimizer.log',
            'formatter': 'detailed',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

### 성능 메트릭 수집
```python
# 최적화 성능 추적
metrics = {
    'optimization_time': time.time() - start_time,
    'converged_solutions': len(pareto_front),
    'xfoil_success_rate': convergence_rate,
    'memory_usage': psutil.Process().memory_info().rss,
    'cpu_usage': psutil.cpu_percent()
}
```


**주요 특징:**
- ✅ React 기반 현대적 UI
- ✅ X-foil 세부 파라미터 제어 (Reynolds, Mach, Ncrit 등)
- ✅ 실시간 WebSocket 통신
- ✅ NURBS 기반 에어포일 파라미터화
- ✅ 다목적 최적화 (NSGA-II)
- ✅ Docker 기반 완전한 개발/배포 환경
- ✅ 인터랙티브 시각화 (Plotly.js)
