# Airfoil Analysis Solvers

통합 airfoil 공력 해석 솔버 패키지

## 솔버 비교

| Solver | 속도 | 정확도 | Re 범위 | Mach 범위 | 비고 |
|--------|------|--------|---------|-----------|------|
| **NeuralFoil** | ★★★★★ | ★★★☆☆ | 1e4-1e7 | 0-0.5 | 기본 솔버, 빠르고 안정 |
| **XFoil** | ★★★★☆ | ★★★★☆ | 1e4-1e6 | 0-0.5 | 패널법, 고정밀 |
| **SU2 SA** | ★★☆☆☆ | ★★★★☆ | 1e5-1e8 | 0-2.0 | RANS, 압축성 |
| **SU2 SST** | ★★☆☆☆ | ★★★★★ | 1e5-1e8 | 0-2.0 | 천음속, 분리류 |

## 디렉토리 구조

```
solvers/
├── __init__.py          # 통합 인터페이스, SolverType enum
├── unified.py           # 자동 solver 선택 및 분석
├── xfoil_solver.py      # XFoil 인터페이스
├── neuralfoil_solver.py # NeuralFoil 인터페이스
├── su2_solver.py        # SU2 인터페이스
├── su2/                 # SU2 설정 템플릿
│   ├── config_templates/
│   └── README.md
├── neuralfoil/          # NeuralFoil 문서
│   └── README.md
└── tests/               # 테스트
```

## 환경 설정

### Docker (권장)

```bash
# 빌드 (처음 ~30-60분 소요, SU2 컴파일 포함)
docker-compose build

# 실행
docker-compose up -d

# 컨테이너 접속
docker exec -it airfoil-optim bash
```

한 번 빌드 후, 모든 솔버(XFoil, NeuralFoil, SU2)가 준비된 상태로 사용 가능합니다.

## 빠른 시작

### 2. 사용

```python
# 자동 solver 선택 (권장)
from solvers.unified import analyze

result = analyze(
    airfoil_file='input/airfoil/naca0012.dat',
    reynolds=5e5,
    alpha=5.0
)
print(f"CL = {result['CL']:.6f}, CD = {result['CD']:.6f}")
```

### 3. 특정 solver 지정

```python
# NeuralFoil (빠른 분석)
result = analyze('naca0012.dat', reynolds=5e5, alpha=5.0, solver='neuralfoil')

# XFoil (고정밀)
result = analyze('naca0012.dat', reynolds=5e5, alpha=5.0, solver='xfoil')

# SU2 (압축성)
result = analyze('naca0012.dat', reynolds=3e6, alpha=2.0, mach=0.75, solver='su2_sst')
```

### 4. AoA Sweep

```python
from solvers.unified import analyze_sweep

results = analyze_sweep(
    'naca0012.dat',
    reynolds=5e5,
    alpha_range=(-5, 15, 1.0)  # min, max, step
)
```

## Solver 선택 로직

```
                    ┌─────────────────┐
                    │   Input: Re, M  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Mach ≥ 0.5?   │
                    └────────┬────────┘
                             │
                 Yes ────────┼──────── No
                 │                      │
        ┌────────▼────────┐    ┌───────▼───────┐
        │   SU2 Required  │    │   Re ≥ 1e6?   │
        └────────┬────────┘    └───────┬───────┘
                 │                      │
        ┌────────▼────────┐    Yes ─────┼───── No
        │ M≥0.7? SST:SA   │    │               │
        └─────────────────┘    │        ┌──────▼──────┐
                               │        │  NeuralFoil │
                        ┌──────▼──────┐ │  (Default)  │
                        │   SU2 SA    │ └─────────────┘
                        └─────────────┘
```

## Docker 환경

### 개발 환경 (모든 솔버)

```bash
docker-compose up -d airfoil-dev
docker-compose exec airfoil-dev bash
```

### SU2 전용 (압축성 해석)

```bash
docker-compose up -d su2
docker-compose exec su2 bash
# SU2_CFD config.cfg
```

### NeuralFoil 전용 (빠른 해석)

```bash
docker-compose up -d neuralfoil
docker-compose exec neuralfoil python -c "
from solvers.unified import analyze
print(analyze('input/airfoil/naca0012.dat', 5e5, 5.0))
"
```

## API 참조

### `analyze()`

```python
def analyze(
    airfoil_file: str,    # Airfoil .dat 파일 경로
    reynolds: float,       # Reynolds 수
    alpha: float,          # 받음각 (도)
    mach: float = 0.0,     # Mach 수 (default: 비압축성)
    solver: str = None,    # 'neuralfoil', 'xfoil', 'su2_sa', 'su2_sst'
    ncrit: float = 9.0,    # 천이 Ncrit 값
    use_fallback: bool = True,  # 실패 시 다른 solver 시도
) -> Dict
```

### `analyze_sweep()`

```python
def analyze_sweep(
    airfoil_file: str,
    reynolds: float,
    alpha_range: Tuple[float, float, float] | List[float],
    mach: float = 0.0,
    solver: str = None,
) -> List[Dict]
```

### `compare_solvers()`

```python
def compare_solvers(
    airfoil_file: str,
    reynolds: float,
    alpha: float,
    mach: float = 0.0,
) -> Dict[str, Dict]
```

## 결과 형식

모든 솔버는 동일한 결과 형식을 반환합니다:

```python
{
    'reynolds': float,           # Reynolds 수
    'aoa': float,                # 받음각 (도)
    'mach': float,               # Mach 수
    'CL': float,                 # 양력 계수
    'CD': float,                 # 항력 계수
    'CM': float,                 # 모멘트 계수
    'Top_Xtr': float,            # 윗면 천이 위치 (x/c)
    'Bot_Xtr': float,            # 아랫면 천이 위치 (x/c)
    'analysis_confidence': float, # 신뢰도 (NeuralFoil만)
    'converged': bool,           # 수렴 여부
    'solver': str,               # 사용된 솔버 이름
}
```

## 테스트

```bash
# 통합 테스트
python -m pytest solvers/tests/

# 솔버 가용성 확인
python -c "from solvers import get_available_solvers; print(get_available_solvers())"
```
