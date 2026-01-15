# Airfoil Analysis Quick Start Guide

## 디렉토리 구조

```
input/airfoil/          # Airfoil 형상 파일 (.dat) 저장 위치
output/analysis/        # 해석 결과 저장 위치
  ├── aoa_sweep/        # AoA sweep 결과
  └── reynolds_sweep/   # Reynolds sweep 결과
scripts/                # 해석 스크립트들
```

## 사용 방법

### 1. Airfoil 파일 준비

`input/airfoil/` 디렉토리에 XFOIL 형식의 airfoil 좌표 파일(\*.dat)을 저장합니다.

예시 파일이 이미 포함되어 있습니다:

- `naca0012.dat` - NACA 0012 airfoil

### 2. 단일 파일 해석

#### AoA Sweep (고정 Reynolds number, AoA 변화)

```bash
# 기본 사용법
python scripts/aoa_sweep.py input/airfoil/naca0012.dat 1000000 -5 15 0.5

# 매개변수 설명:
# - airfoil 파일: input/airfoil/naca0012.dat
# - Reynolds number: 1000000 (1e6)
# - AoA 범위: -5° ~ 15°
# - AoA 증분: 0.5°
```

#### Reynolds Sweep (고정 AoA, Reynolds number 변화)

```bash
# 기본 사용법
python scripts/reynolds_sweep.py input/airfoil/naca0012.dat 5.0 100000 5000000 500000

# 매개변수 설명:
# - airfoil 파일: input/airfoil/naca0012.dat
# - AoA: 5.0°
# - Reynolds number 범위: 100,000 ~ 5,000,000
# - Reynolds number 증분: 500,000
```

### 3. 배치 해석 (input/airfoil의 모든 .dat 파일)

#### AoA Sweep 배치 해석

```bash
# 기본 설정으로 실행
python scripts/run_aoa_sweep_batch.py

# 사용자 설정으로 실행
python scripts/run_aoa_sweep_batch.py --re 1000000 --aoa-min -5 --aoa-max 15 --d-aoa 0.5

# 옵션:
#   --re RE            Reynolds number (기본값: 1000000)
#   --aoa-min MIN      최소 AoA (기본값: -5°)
#   --aoa-max MAX      최대 AoA (기본값: 15°)
#   --d-aoa STEP       AoA 증분 (기본값: 0.5°)
#   --ncrit NCRIT      Transition parameter (기본값: 9)
#   --iter ITER        최대 반복 횟수 (기본값: 100)
```

#### Reynolds Sweep 배치 해석

```bash
# 기본 설정으로 실행
python scripts/run_reynolds_sweep_batch.py

# 사용자 설정으로 실행
python scripts/run_reynolds_sweep_batch.py --aoa 5.0 --re-min 100000 --re-max 5000000 --d-re 500000

# 옵션:
#   --aoa AOA          Angle of attack (기본값: 5.0°)
#   --re-min MIN       최소 Reynolds number (기본값: 100000)
#   --re-max MAX       최대 Reynolds number (기본값: 5000000)
#   --d-re STEP        Reynolds number 증분 (기본값: 500000)
#   --ncrit NCRIT      Transition parameter (기본값: 9)
#   --iter ITER        최대 반복 횟수 (기본값: 100)
```

## 해석 결과 확인

### 결과 파일 위치

#### AoA Sweep

```
output/analysis/aoa_sweep/{airfoil_name}/
  ├── {airfoil}_Re{reynolds}_aoa{min}to{max}.txt   # XFOIL polar 데이터
  ├── {airfoil}_Re{reynolds}_aoa{min}to{max}.csv   # CSV 형식 데이터
  └── {airfoil}_Re{reynolds}_aoa{min}to{max}_dump.txt  # 상세 출력
```

#### Reynolds Sweep

```
output/analysis/reynolds_sweep/{airfoil_name}/
  ├── {airfoil}_aoa{angle}_Re{min}to{max}.csv   # CSV 형식 데이터
  └── {airfoil}_aoa{angle}_Re{min}to{max}.txt   # 텍스트 형식 데이터
```

### CSV 파일 구조

**AoA Sweep 결과:**

- `alpha`: 받음각 (degrees)
- `CL`: 양력 계수
- `CD`: 항력 계수
- `CDp`: 압력 항력 계수
- `CM`: 모멘트 계수
- `Top_Xtr`: 상면 천이점 위치 (x/c)
- `Bot_Xtr`: 하면 천이점 위치 (x/c)

**Reynolds Sweep 결과:**

- `Re`: Reynolds number
- `alpha`: 받음각 (degrees)
- `CL`: 양력 계수
- `CD`: 항력 계수
- `CDp`: 압력 항력 계수
- `CM`: 모멘트 계수
- `Top_Xtr`: 상면 천이점 위치 (x/c)
- `Bot_Xtr`: 하면 천이점 위치 (x/c)
- `converged`: 수렴 여부

## 예시 워크플로우

### 1. 새로운 airfoil 추가 및 해석

```bash
# 1. airfoil 파일을 input/airfoil/에 복사
# 예: myairfoil.dat

# 2. AoA sweep 실행
python scripts/aoa_sweep.py input/airfoil/myairfoil.dat 1000000 -5 15 0.5

# 3. 결과 확인
# output/analysis/aoa_sweep/myairfoil/ 디렉토리 참조
```

### 2. 여러 airfoil 비교 분석

```bash
# 1. 모든 airfoil 파일을 input/airfoil/에 추가
# 예: naca0012.dat, naca2412.dat, custom.dat

# 2. 배치 해석 실행
python scripts/run_aoa_sweep_batch.py --re 1000000 --aoa-min -5 --aoa-max 15 --d-aoa 0.5

# 3. 각 airfoil의 결과 비교
# output/analysis/aoa_sweep/{airfoil_name}/ 디렉토리들 참조
```

### 3. Reynolds number 영향 분석

```bash
# 특정 AoA에서 Reynolds number 변화에 따른 성능 분석
python scripts/reynolds_sweep.py input/airfoil/naca0012.dat 5.0 100000 5000000 500000
```

## 추가 정보

### XFOIL 설치 확인

```bash
xfoil
# XFOIL 프롬프트가 나타나면 정상 설치됨
# 종료: QUIT 입력
```

### 해석이 실패하는 경우

- Reynolds number가 너무 낮거나 높을 수 있습니다 (권장: 1e5 ~ 1e7)
- AoA가 실속 영역에 있을 수 있습니다
- `--ncrit` 값을 조정해보세요 (일반적으로 5~9)
- `--iter` 값을 늘려보세요 (100 → 200)

### 성능 최적화

- 배치 해석 시 너무 작은 증분은 시간이 오래 걸립니다
- AoA sweep: 0.25°~1.0° 권장
- Reynolds sweep: 선형 증분 대신 로그 스케일 자동 적용됨
