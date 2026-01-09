# Airfoil Optimization

Airfoil 최적화를 위한 Docker 기반 개발 환경입니다. XFOIL을 사용한 airfoil 해석 및 최적화 기능을 제공합니다.

## 환경 구성

### 필수 요구사항

- Docker
- Docker Compose

### 빠른 시작

1. **Docker 이미지 빌드 및 컨테이너 시작**

```bash
docker-compose up -d xfoil-dev
```

2. **개발 컨테이너 접속**

```bash
docker-compose exec xfoil-dev bash
```

3. **XFOIL 실행 확인**

```bash
xfoil
```

### 서비스 구성

#### xfoil-dev (메인 개발 환경)

XFOIL이 설치된 주 개발 환경입니다.

```bash
# 컨테이너 시작
docker-compose up -d xfoil-dev

# 컨테이너 접속
docker-compose exec xfoil-dev bash

# 컨테이너 종료
docker-compose down
```

#### jupyter (Jupyter Notebook 서버)

최적화 알고리즘 개발을 위한 Jupyter Notebook 환경입니다.

```bash
# Jupyter 서버 시작
docker-compose up -d jupyter

# 브라우저에서 접속: http://localhost:8888
# 토큰은 로그에서 확인
docker-compose logs jupyter
```

## 프로젝트 구조

```
.
├── xfoil/                  # XFOIL 소스 코드
├── environment.yml         # Conda 환경 설정
├── Dockerfile             # Docker 이미지 정의
├── docker-compose.yml     # Docker Compose 설정
└── README.md             # 본 문서
```

## 개발 환경 정보

- **Base OS**: Ubuntu 22.04
- **Fortran Compiler**: gfortran
- **C/C++ Compiler**: gcc/g++
- **Build System**: CMake
- **Python**: 3.12 (Conda environment)
- **Scientific Libraries**: NumPy, SciPy, MPI4py, Numba

## 사용 예제

### NACA Airfoil 생성

```bash
# 컨테이너 접속
docker-compose exec xfoil-dev bash

# 단일 NACA airfoil 생성
python scripts/generate_naca_airfoil.py 0012

# 여러 airfoil 일괄 생성
python scripts/generate_naca_airfoil.py --batch

# 생성된 파일 확인
ls public/airfoil/
```

### XFOIL 기본 사용

```bash
# 컨테이너 접속
docker-compose exec xfoil-dev bash

# XFOIL 실행
xfoil

# XFOIL 명령어 예제 (XFOIL 프롬프트에서)
# NACA 0012 airfoil 불러오기
NACA 0012

# 패널 생성
PANE

# 점성 해석 모드
OPER

# Reynolds 수 설정
RE 1000000

# 받음각 별 해석
ASEQ 0 10 1
```

### Python에서 XFOIL 사용

```python
import subprocess
import os

# XFOIL 명령어 스크립트 생성
commands = """
NACA 0012
PANE
OPER
VISC 1000000
PACC
polar.txt

ASEQ 0 10 1

QUIT
"""

# XFOIL 실행
with open('xfoil_input.txt', 'w') as f:
    f.write(commands)

subprocess.run(['xfoil'], stdin=open('xfoil_input.txt'))
```

## 문제 해결

### xfoil 명령어를 찾을 수 없는 경우

```bash
# xfoil 재빌드
cd /workspace/xfoil
rm -rf build
mkdir build && cd build
cmake ..
make
make install
```

### MPI 관련 오류

```bash
# MPI 환경 확인
mpirun --version

# 테스트 실행
mpirun -np 4 python your_script.py
```

## 라이선스

XFOIL: GNU General Public License v2.0

## 참고 자료

- [XFOIL 공식 웹사이트](http://web.mit.edu/drela/Public/web/xfoil/)
- [XFOIL Documentation](xfoil/xfoil_doc.txt)
