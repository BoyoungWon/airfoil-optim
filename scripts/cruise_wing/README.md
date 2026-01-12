# Cruise Wing Optimization Module

순항 익형 최적화를 위한 전용 모듈입니다.

## 개요

**Application**: 일반 항공기, 글라이더, 소형 무인기  
**Operating Condition** (XFOIL Valid Range):

- Reynolds: 50k - 50M
- Mach: 0 - 0.5 (with compressibility correction)
- AoA: -5° - 15° (pre-stall region)

**Design Objective**: Maximize L/D at cruise  
**Complexity**: ★☆☆☆☆ (가장 단순)  
**Timeline**: 1-2일

## 기술 스택

| Component           | Method               | Rationale                                         |
| ------------------- | -------------------- | ------------------------------------------------- |
| **Parametrization** | NACA 4-digit         | 3 variables → 빠른 최적화, 물리적 타당성 보장     |
| **Solver**          | XFOIL                | Re 50k-50M, Ma≤0.5에서 신뢰성, <1초/evaluation    |
| **Surrogate**       | Kriging (Matérn 5/2) | 적은 samples로 good accuracy, uncertainty 제공    |
| **Optimizer**       | SLSQP                | Smooth objective, 빠른 수렴, constraint 처리 우수 |

## 워크플로우

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

## 빠른 시작

### 기본 사용법

```python
from scripts.cruise_wing import optimize_cruise_wing

# 기본 설정으로 최적화 실행
result = optimize_cruise_wing(
    reynolds=3e6,
    aoa=3.0,
    mach=0.2
)

print(f"Optimal airfoil: {result.optimal_airfoil['name']}")
print(f"L/D: {result.optimal_airfoil['L/D']:.2f}")
```

### CLI 사용법

```bash
# 기본 최적화
python scripts/run_cruise_wing.py

# 커스텀 파라미터
python scripts/run_cruise_wing.py --reynolds 5e6 --aoa 4.0 --mach 0.25

# 시나리오 파일 사용
python scripts/run_cruise_wing.py --scenario scenarios/cruise_wing.yaml

# Direct optimization (surrogate 없이)
python scripts/run_cruise_wing.py --direct

# 더 많은 training samples
python scripts/run_cruise_wing.py --samples 150
```

### 상세 설정

```python
from scripts.cruise_wing.workflow import (
    CruiseWingOptimizer,
    CruiseWingConfig,
    DesignPoint
)

# 상세 설정
config = CruiseWingConfig(
    # Design points
    design_points=[
        DesignPoint(reynolds=3e6, aoa=3.0, mach=0.2, weight=1.0, name='cruise'),
        DesignPoint(reynolds=2e6, aoa=5.0, mach=0.15, weight=0.3, name='climb'),
    ],

    # Parameter bounds
    m_bounds=(0.0, 0.06),    # Max camber: 0-6%
    p_bounds=(0.2, 0.5),     # Position: 20-50% chord
    t_bounds=(0.09, 0.18),   # Thickness: 9-18%

    # Constraints
    cl_min=0.4,
    cm_min=-0.1,
    cm_max=0.0,
    ld_min=50.0,
    t_min=0.10,

    # Surrogate settings
    use_surrogate=True,
    n_training_samples=100,
    kriging_kernel='matern',

    # Optimization settings
    max_iterations=50,
    n_multistart=5,

    # Output
    output_dir="output/optimization/my_cruise_wing",
    save_history=True,
    create_plots=True
)

optimizer = CruiseWingOptimizer(config=config)
result = optimizer.run(verbose=True)
```

## 모듈 구조

```
scripts/cruise_wing/
├── __init__.py           # 패키지 초기화
├── workflow.py           # 메인 워크플로우 오케스트레이터
├── database.py           # NACA 데이터베이스 및 스캔
├── analyzer.py           # XFOIL 분석 인터페이스
├── kriging.py            # Kriging surrogate model
├── optimizer.py          # SLSQP 최적화
└── visualizer.py         # 결과 시각화
```

### 모듈별 설명

#### `database.py` - NACADatabase

NACA 익형 데이터베이스 관리 및 스캔

```python
from scripts.cruise_wing.database import NACADatabase

db = NACADatabase()

# NACA 코드 파싱
params = db.parse_naca_code("2412")  # {'m': 0.02, 'p': 0.4, 't': 0.12}

# 좌표 생성
coords = db.generate_naca_coords(0.02, 0.4, 0.12)

# 데이터베이스 스캔
results = db.scan_database(reynolds=3e6, aoa=3.0)
```

#### `analyzer.py` - AirfoilAnalyzer

XFOIL 분석 인터페이스

```python
from scripts.cruise_wing.analyzer import AirfoilAnalyzer

analyzer = AirfoilAnalyzer(ncrit=10, n_panels=160)

# 단일점 분석
result = analyzer.analyze_single(coords, reynolds=3e6, aoa=3.0)

# Polar 분석
polar = analyzer.analyze_polar(coords, reynolds=3e6, aoa_range=(-2, 12))

# 최대 L/D 찾기
max_ld = analyzer.find_max_ld(coords, reynolds=3e6)
```

#### `kriging.py` - CruiseWingKriging

Kriging surrogate model

```python
from scripts.cruise_wing.kriging import CruiseWingKriging, LHSSampler

# Sampling
sampler = LHSSampler(bounds=[(0, 0.06), (0.2, 0.5), (0.09, 0.18)])
X_train = sampler.sample(100)

# Training
kriging = CruiseWingKriging(kernel='matern')
stats = kriging.train(X_train, y_train)

# Prediction with uncertainty
pred, std = kriging.predict_single(params, return_std=True)

# Cross-validation
cv_scores = kriging.cross_validate(n_folds=5)
```

#### `optimizer.py` - SLSQPOptimizer

SLSQP 최적화

```python
from scripts.cruise_wing.optimizer import (
    SurrogateOptimizer,
    DirectXFOILOptimizer,
    create_cruise_constraints
)

# Constraints
constraints = create_cruise_constraints(cl_min=0.4, ld_min=50)

# Surrogate-based optimization
optimizer = SurrogateOptimizer(
    surrogate_model=kriging,
    bounds=bounds,
    constraints=constraints
)
result = optimizer.optimize_multistart(n_starts=5)

# Direct XFOIL optimization
direct_opt = DirectXFOILOptimizer(
    reynolds=3e6, aoa=3.0, mach=0.2,
    bounds=bounds,
    constraints=constraints
)
result = direct_opt.optimize()
```

## 출력 결과

최적화 완료 후 다음 파일들이 생성됩니다:

```
output/optimization/cruise_wing/
├── optimal_airfoil.dat           # 최적 익형 좌표
├── optimization_history.json     # 최적화 히스토리
├── optimization_summary.json     # 요약 정보
├── surrogate_model.pkl          # 학습된 surrogate model
├── database/
│   └── naca_database.json       # NACA 분석 캐시
└── figures/
    ├── convergence.png          # 수렴 그래프
    ├── airfoil_comparison.png   # 익형 비교
    ├── polar_comparison.png     # Polar 비교
    ├── design_space.png         # 설계 공간 탐색
    ├── surrogate_validation.png # Surrogate 검증
    └── optimization_summary.png # 종합 리포트
```

## 테스트

```bash
# 전체 테스트
python scripts/test_cruise_wing.py

# 빠른 테스트
python scripts/test_cruise_wing.py --quick

# 특정 모듈 테스트
python scripts/test_cruise_wing.py --module kriging
```

## 요구사항

### 필수

- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- PyYAML
- XFOIL (시스템 PATH에 설치)

### 선택

- Matplotlib (시각화)
- PyTorch (Neural Network surrogate - 미래 확장용)

## SaaS 플랫폼 통합

이 모듈은 SaaS 플랫폼의 **Tier 1: Quick Analysis** 기능으로 사용됩니다:

```
Tier 1: Quick Analysis (무료/저가)
├─ NACA parametrization
├─ XFOIL direct call (no surrogate)
├─ Single-point or simple polar
├─ Turnaround: Minutes
└─ Target: Hobbyist, students, initial screening
```

## 다음 단계

Cruise Wing 시나리오가 완료되면 다음 시나리오 구현:

1. **High-Lift** (Tier 2)

   - CST parametrization (12 variables)
   - Neural Network surrogate
   - GA optimizer

2. **Control Surface** (Tier 1)

   - Hinge position optimization
   - 간단하지만 실용적

3. **Propeller** (Tier 3)
   - CST + FFD
   - Multi-fidelity surrogate
   - VPM 통합 준비
