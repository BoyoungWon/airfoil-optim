# Input Airfoil Directory

이 디렉토리에 Xfoil 해석을 수행할 airfoil 형상 파일(\*.dat)을 저장하세요.

## 파일 형식

Airfoil 좌표 파일은 XFOIL 형식이어야 합니다:

- 첫 줄: Airfoil 이름
- 이후: x, y 좌표 (공백 또는 탭으로 구분)
- 좌표는 trailing edge에서 시작하여 상면을 따라 leading edge로, 다시 하면을 따라 trailing edge로 돌아가는 순서

## 사용 예시

### 1. AoA Sweep 해석

```bash
# 단일 파일 해석
python scripts/aoa_sweep.py input/airfoil/naca0012.dat 1000000 -5 15 0.5

# 모든 .dat 파일 자동 해석
python scripts/run_aoa_sweep_batch.py
```

### 2. Reynolds Sweep 해석

```bash
# 단일 파일 해석
python scripts/reynolds_sweep.py input/airfoil/naca0012.dat 5.0 100000 5000000 500000

# 모든 .dat 파일 자동 해석
python scripts/run_reynolds_sweep_batch.py
```

## 해석 결과

해석 결과는 다음 디렉토리에 저장됩니다:

- AoA Sweep: `output/analysis/aoa_sweep/`
- Reynolds Sweep: `output/analysis/reynolds_sweep/`
