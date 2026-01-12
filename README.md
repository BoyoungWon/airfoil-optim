# Airfoil Optimization Framework

Surrogate model 기반 airfoil 최적화 프레임워크입니다. XFOIL 및 OpenFOAM을 사용한 공력 해석과 다양한 형상 매개변수화 방법(NACA, CST, FFD)을 지원하며, Kriging, Neural Network 등의 surrogate model을 활용한 효율적인 최적화를 제공합니다.

## 주요 기능

- **다중 형상 매개변수화**: NACA (3 params), CST (8-30 params), FFD (15-100+ params)
- **CFD 솔버**: XFOIL (2D panel method), OpenFOAM (3D RANS/LES)
- **Surrogate 모델**: Kriging/GPR, Neural Network, Polynomial RSM
- **최적화 알고리즘**: SLSQP, NSGA-II, Bayesian Optimization
- **다중 설계점 최적화**: 가중 평균 기반 multi-point optimization
- **시나리오 기반 실행**: YAML 설정 파일로 간편한 최적화 실행

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

**xfoil-dev**: XFOIL이 설치된 메인 개발 환경

```bash
# 컨테이너 시작
docker-compose up -d xfoil-dev

# 컨테이너 접속
docker-compose exec xfoil-dev bash

# 컨테이너 종료
docker-compose down
```

## 프로젝트 구조

```
.
├── xfoil/                   # XFOIL 소스 코드
├── scripts/                 # Python 스크립트
│   ├── generate_naca_airfoil.py   # NACA airfoil 생성
│   ├── ffd_airfoil.py             # FFD airfoil 생성
│   ├── optimize_airfoil.py        # 메인 최적화 스크립트
│   ├── validate_scenario.py       # 시나리오 검증
│   ├── run_cruise_wing.py         # Cruise Wing CLI
│   ├── test_cruise_wing.py        # Cruise Wing 테스트
│   ├── cruise_wing/               # Cruise Wing 전용 모듈
│   │   ├── __init__.py            # 패키지 초기화
│   │   ├── database.py            # NACA 데이터베이스
│   │   ├── analyzer.py            # XFOIL 해석 인터페이스
│   │   ├── kriging.py             # Kriging surrogate 모델
│   │   ├── optimizer.py           # SLSQP 최적화
│   │   ├── visualizer.py          # 결과 시각화
│   │   ├── workflow.py            # 4-phase 워크플로우
│   │   └── README.md              # 모듈 설명서
│   └── optimize/                  # 최적화 모듈 (범용)
│       ├── parametrization.py     # 형상 매개변수화 (NACA/CST/FFD)
│       ├── surrogate.py           # Surrogate 모델
│       └── xfoil_interface.py     # XFOIL 인터페이스
├── scenarios/               # 최적화 시나리오 (YAML)
│   ├── cruise_wing.yaml           # 순항 익형 최적화 ✓ 구현완료
│   ├── high_lift.yaml             # 고양력 익형 최적화
│   ├── low_speed.yaml             # 저속 UAV 익형
│   ├── propeller.yaml             # 프로펠러 익형
│   ├── wind_turbine.yaml          # 풍력 터빈 익형
│   └── control_surface.yaml       # 조종면 익형
├── output/                  # 프로젝트 산출물 (gitignore)
│   ├── airfoil/             # 생성된 airfoil 형상
│   ├── analysis/            # XFOIL 해석 결과
│   ├── surrogate/           # Surrogate model 학습 결과
│   └── optimization/        # 최적화 결과
├── public/airfoil/          # 공유 airfoil 저장소
├── environment.yml          # Conda 환경 설정
├── Dockerfile               # Docker 이미지 정의
├── docker-compose.yml       # Docker Compose 설정
└── README.md                # 본 문서
```

## 개발 환경 정보

- **Base OS**: Ubuntu 22.04
- **Fortran Compiler**: gfortran
- **C/C++ Compiler**: gcc/g++
- **Build System**: CMake
- **Python**: 3.12 (Conda environment)
- **Scientific Libraries**: NumPy, SciPy, MPI4py, Numba
- **CFD Solvers**:
  - XFOIL (built from source) - 2D panel method
  - OpenFOAM (optional, for 3D scenarios) - RANS/LES

## 빠른 시작

### 1. 시나리오 검증

```bash
docker-compose exec xfoil-dev bash

# 모든 시나리오 검증
python scripts/validate_scenario.py --all

# 특정 시나리오 검증
python scripts/validate_scenario.py --scenario scenarios/cruise_wing.yaml
```

### 2. 최적화 실행

#### Cruise Wing (전용 모듈 구현완료)

```bash
# Cruise Wing 최적화 실행 (NACA + XFOIL + Kriging + SLSQP)
python scripts/run_cruise_wing.py --scenario scenarios/cruise_wing.yaml

# 테스트 실행
python scripts/test_cruise_wing.py

# 직접 최적화 (surrogate 없이 XFOIL 직접 호출)
python scripts/run_cruise_wing.py --direct --reynolds 1000000 --aoa 5.0 --mach 0.2
```

#### 범용 최적화 (다른 시나리오)

```bash
# 순항 익형 최적화 (NACA + Kriging) - 범용 모듈
python scripts/optimize_airfoil.py --scenario scenarios/cruise_wing.yaml

# 고양력 익형 최적화 (CST + Neural Network)
python scripts/optimize_airfoil.py --scenario scenarios/high_lift.yaml

# 프로펠러 익형 최적화 (FFD + Neural Network)
python scripts/optimize_airfoil.py --scenario scenarios/propeller.yaml --verbose
```

### 3. 결과 확인

```bash
# 최적화 결과 확인
ls output/optimization/cruise_wing/

# 최적 airfoil 형상
cat output/optimization/cruise_wing/optimal_airfoil.dat

# 최적화 히스토리
cat output/optimization/cruise_wing/optimization_history.json
```

## Cruise Wing 최적화 (구현완료)

### 개요

**Application**: 일반 항공기, 글라이더, 소형 무인기  
**Operating Condition**:

- Reynolds: 50,000 - 50,000,000 (XFOIL valid range)
- Mach: 0 - 0.5 (with compressibility correction)
- AoA: -5° - 15° (pre-stall region)

**Design Objective**: Maximize L/D at cruise  
**Complexity**: ★☆☆☆☆ (가장 단순, 3 parameters)  
**Timeline**: 1-2일

### 기술 스택

| Component           | Method               | Rationale                                      |
| ------------------- | -------------------- | ---------------------------------------------- |
| **Parametrization** | NACA 4-digit         | 3 variables → 빠른 최적화, 물리적 타당성       |
| **Solver**          | XFOIL                | Re 50k-50M, Ma≤0.5에서 신뢰성, <1초/evaluation |
| **Surrogate**       | Kriging (Matérn 5/2) | 적은 samples로 좋은 정확도, uncertainty 제공   |
| **Optimizer**       | SLSQP                | Smooth objective, 빠른 수렴, constraint 처리   |

### 워크플로우 (4 Phase)

```
Phase 1: Database Screening (30분)
├─ XFOIL로 NACA library 스캔
└─ Similar Re, Cl에서 top 5 선정

Phase 2: Surrogate Training (1-2시간)
├─ Latin Hypercube Sampling (80-100 samples)
├─ Kriging model 구축
└─ Cross-validation 검증

Phase 3: NACA Optimization (1-2시간)
├─ SLSQP with surrogate
├─ Multi-start optimization (5 starts)
└─ Optimal NACA parameters 도출

Phase 4: Validation (30분)
├─ XFOIL polar 분석
└─ 최종 성능 확인
```

### 사용 방법

```bash
# 기본 실행
python scripts/run_cruise_wing.py --scenario scenarios/cruise_wing.yaml

# 상세 로그 출력
python scripts/run_cruise_wing.py --scenario scenarios/cruise_wing.yaml --verbose

# 직접 최적화 (surrogate 없이)
python scripts/run_cruise_wing.py --direct --reynolds 1000000 --aoa 5.0 --mach 0.2

# 커스텀 파라미터
python scripts/run_cruise_wing.py --scenario scenarios/cruise_wing.yaml \
  --reynolds 2000000 --aoa 6.0 --mach 0.25
```

### 결과 확인

```bash
# 최적화 결과 디렉토리
ls output/optimization/cruise_wing_[timestamp]/

# 주요 파일
├── optimal_airfoil.dat        # 최적 익형 좌표
├── optimization_history.json   # 최적화 이력
├── surrogate_model.pkl         # 학습된 surrogate 모델
├── validation_results.json     # 검증 결과
└── plots/                      # 시각화 결과
    ├── convergence.png         # 수렴 곡선
    ├── airfoil_comparison.png  # 익형 비교
    ├── polar_comparison.png    # 극선 비교
    └── design_space.png        # 설계 공간
```

### 모듈 구조

- `database.py`: NACA 데이터베이스 관리 및 초기 스크리닝
- `analyzer.py`: XFOIL 인터페이스 (단일/극선/다중점 해석)
- `kriging.py`: Kriging surrogate 모델 (GPR with Matérn 5/2 kernel)
- `optimizer.py`: SLSQP 최적화 (surrogate/direct 모드)
- `visualizer.py`: 결과 시각화 (6가지 플롯 타입)
- `workflow.py`: 4-phase 워크플로우 오케스트레이터

자세한 내용은 [scripts/cruise_wing/README.md](scripts/cruise_wing/README.md)를 참조하세요.

## 사용 예제

### NACA Airfoil 생성

```bash
docker-compose exec xfoil-dev bash

# 단일 NACA airfoil 생성
python scripts/generate_naca_airfoil.py 2412

# 여러 airfoil 일괄 생성
python scripts/generate_naca_airfoil.py --batch
```

### FFD (Free Form Deformation) Airfoil 생성

```bash
# 단일 FFD airfoil 생성
python scripts/ffd_airfoil.py --naca 2412 --control-points 5 3 --amplitude 0.02

# Surrogate model용 다중 샘플 생성
python scripts/ffd_airfoil.py --naca 2412 --samples 100

# 생성된 샘플 확인
ls output/airfoil/ffd/
```

**FFD 주요 파라미터:**

- `--control-points NX NY`: 제어점 개수 (기본: 5 3)
- `--amplitude`: 변형 크기 (chord 비율, 기본: 0.02)
- `--samples N`: 생성할 랜덤 샘플 개수
- `--plot`: 변형 결과 시각화

### 커스텀 시나리오 생성

```yaml
# scenarios/my_custom.yaml
name: "Custom Airfoil Optimization"
description: "My custom optimization scenario"
category: "A. Fixed-Wing Aircraft"

parametrization:
  method: cst # naca, cst, or ffd
  n_upper: 6
  n_lower: 6

design_points:
  - reynolds: 500000
    aoa: 5.0
    mach: 0.0
    weight: 0.5
  - reynolds: 1000000
    aoa: 8.0
    mach: 0.0
    weight: 0.5

objectives:
  - metric: "CL/CD"
    type: maximize
    weight: 1.0

optimization:
  algorithm: scipy
  method: SLSQP
  max_iterations: 100
  convergence_tol: 1e-6

surrogate:
  method: kriging
  kernel: matern
  training_samples: 200
  validation_split: 0.2

output:
  directory: "output/optimization/my_custom"
```

실행:

```bash
python scripts/optimize_airfoil.py --scenario scenarios/my_custom.yaml
```

## 최적화 알고리즘

### 1. SLSQP (Sequential Least Squares Programming)

**현재 구현**: Cruise Wing, Control Surface

- **적용 대상**: 단순 형상 (3-6 parameters), 단일 목적 최적화
- **장점**:
  - 빠른 수렴 (평균 20-50 iterations)
  - 제약조건 처리 우수 (등식/부등식)
  - Gradient 기반으로 정확한 최적해
- **단점**:
  - 국소 최적해에 갇힐 수 있음
  - Smooth objective 필요
  - 초기값에 민감
- **권장 조합**: NACA + Kriging

### 2. NSGA-II (Non-dominated Sorting Genetic Algorithm II)

**라이브러리 추가됨**: pymoo

- **적용 대상**: 중간 복잡도 (8-30 parameters), 다중 목적, 비선형
- **장점**:
  - 전역 탐색 (global search)
  - 다중 목적 최적화 (Pareto front)
  - 국소 최적해 회피
  - Gradient-free (비미분 가능 함수 지원)
- **단점**:
  - 많은 evaluation 필요 (5,000-10,000)
  - 수렴 속도 느림
  - Surrogate 모델 필수
- **권장 조합**: CST/FFD + Neural Network

### 알고리즘 선택 가이드

| 조건             | 추천 알고리즘 | 이유              |
| ---------------- | ------------- | ----------------- |
| Parameters ≤ 6   | **SLSQP**     | 빠른 수렴, 효율적 |
| Parameters > 8   | **NSGA-II**   | 전역 탐색, robust |
| 단일 목적        | **SLSQP**     | 정확한 최적해     |
| 다중 목적        | **NSGA-II**   | Pareto front 제공 |
| Smooth objective | **SLSQP**     | Gradient 활용     |
| Non-convex space | **NSGA-II**   | 국소 최적해 회피  |
| 빠른 프로토타입  | **SLSQP**     | 1-2시간 완료      |
| 정밀 최적화      | **NSGA-II**   | 하루 소요 가능    |

## 해석 솔버 (CFD Solver)

### 1. XFOIL (2D Panel Method)

**현재 구현**: Cruise Wing, Control Surface, High Lift, Low Speed

- **적용 대상**: 2D 익형 해석, 빠른 프로토타입
- **유효 범위**:
  - Reynolds: 50,000 - 50,000,000
  - Mach: 0 - 0.5 (압축성 보정)
  - AoA: -10° - 15° (실속 전)
- **장점**:
  - 매우 빠름 (<1초/evaluation)
  - 설치 간단, 경량
  - 2D 정확도 우수
- **단점**:
  - 2D만 가능
  - 실속 후 부정확
  - 3D 효과 무시
- **권장 사용**: 고정익 2D 단면 최적화

### 2. OpenFOAM (3D RANS/LES)

**향후 구현**: Propeller, Wind Turbine

- **적용 대상**: 3D 유동, 회전익, 복잡한 형상
- **난류 모델**:
  - RANS: k-ω SST, Spalart-Allmaras
  - LES: Smagorinsky, WALE
- **장점**:
  - 3D 유동 정확
  - 회전 효과 반영
  - 복잡한 경계조건
- **단점**:
  - 느림 (10분-1시간/evaluation)
  - 높은 계산 비용
  - Surrogate 필수
- **권장 사용**: 프로펠러, 풍력 터빈, 3D 날개

### 솔버 선택 가이드

| 조건           | 추천 솔버    | 이유         |
| -------------- | ------------ | ------------ |
| 2D 익형 단면   | **XFOIL**    | 빠름, 정확   |
| 3D 날개/회전익 | **OpenFOAM** | 3D 효과 필수 |
| 빠른 반복      | **XFOIL**    | <1초/eval    |
| 정밀 해석      | **OpenFOAM** | RANS/LES     |
| Re < 50M       | **XFOIL**    | 신뢰 범위    |
| 회전 유동      | **OpenFOAM** | 회전 프레임  |
| 프로토타입     | **XFOIL**    | 1-2일 완료   |
| 최종 검증      | **OpenFOAM** | 실제 조건    |

## 형상 매개변수화 방법

### 1. NACA (3 parameters)

- **적용**: 간단한 익형 최적화, 초기 설계
- **파라미터**: m (캠버), p (캠버 위치), t (두께)
- **Surrogate**: Kriging/GPR 권장
- **샘플 수**: 30-60개

### 2. CST (8-30 parameters)

- **적용**: 일반적인 익형 최적화, 고양력 장치
- **파라미터**: Bernstein polynomial 계수
- **Surrogate**: Kriging 또는 Neural Network
- **샘플 수**: 80-600개

### 3. FFD (15-100+ parameters)

- **적용**: 복잡한 형상 최적화, 프로펠러, 터빈
- **파라미터**: 제어점 변위 (nx × ny × 2)
- **Surrogate**: Neural Network 권장
- **샘플 수**: 500-2000개

## 최적화 시나리오

| 시나리오               | 카테고리 | 매개변수화  | Solver   | Surrogate  | Optimizer | 목적          | 상태       |
| ---------------------- | -------- | ----------- | -------- | ---------- | --------- | ------------- | ---------- |
| `cruise_wing.yaml`     | 고정익   | NACA (3)    | XFOIL    | Kriging    | SLSQP     | max L/D       | ✓ 구현완료 |
| `control_surface.yaml` | 조종면   | NACA (3)    | XFOIL    | Kriging    | SLSQP     | effectiveness | 계획중     |
| `high_lift.yaml`       | 고정익   | CST (12-20) | XFOIL    | Neural Net | NSGA-II   | max CL_max    | 계획중     |
| `low_speed.yaml`       | 고정익   | CST (8-16)  | XFOIL    | Kriging    | NSGA-II   | max CL^1.5/CD | 계획중     |
| `propeller.yaml`       | 회전익   | FFD (30-60) | OpenFOAM | Neural Net | NSGA-II   | multi-point   | 계획중     |
| `wind_turbine.yaml`    | 회전익   | CST (20-30) | OpenFOAM | Neural Net | NSGA-II   | max AEP       | 계획중     |

## 필요 패키지

### 기본 패키지 (environment.yml에 포함)

```bash
# Cruise Wing 모듈 (구현완료)
- scikit-learn  # Kriging/GPR surrogate
- matplotlib    # 시각화
- pyyaml        # 설정 파일
- joblib        # 모델 저장

# 최적화 알고리즘
- pymoo         # NSGA-II, NSGA-III (다목적 최적화)

# OpenFOAM interface (향후 추가)
- PyFoam (pip)  # OpenFOAM Python 래퍼
- foampy (pip)  # OpenFOAM 후처리
```

### 향후 시나리오용 추가 패키지

```bash
# Neural Network surrogate
pip install torch
conda install -c conda-forge pytorch

# OpenFOAM (3D scenarios)
# OpenFOAM은 Dockerfile에서 설치 또는
sudo apt-get install openfoam

# OpenFOAM Python tools
pip install PyFoam foampy
```

## 문제 해결

### XFOIL 관련

```bash
# xfoil 재빌드
cd /workspace/xfoil
rm -rf build && mkdir build && cd build
cmake .. && make && make install
```

### Surrogate 모델 학습 실패

- **데이터 부족**: 파라미터당 10-20개 샘플 필요
- **스케일 문제**: 입력/출력 정규화 확인
- **수렴 실패**: XFOIL 설정 조정 (n_iter, reynolds)

### 최적화 수렴 안됨

- **초기값**: 실현 가능한 초기 설계 확인
- **제약조건**: 너무 엄격한 제약 완화
- **알고리즘**: Scipy → Genetic → Bayesian 순서로 시도

## 라이선스

XFOIL: GNU General Public License v2.0

## 참고 자료

- [XFOIL 공식 웹사이트](http://web.mit.edu/drela/Public/web/xfoil/)
- [XFOIL Documentation](xfoil/xfoil_doc.txt)
- [시나리오 설명서](scenarios/README.md)
