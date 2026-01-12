# XFOIL 스크립트

이 디렉토리는 XFOIL을 자동화하기 위한 유틸리티 스크립트를 포함합니다.

## 스크립트 목록

- **generate_naca_airfoil.py**: NACA airfoil 좌표 파일 생성
- **import_airfoil.py**: 외부 airfoil 좌표 파일 검증 및 import
- **aoa_sweep.py**: 받음각(AoA) sweep 해석
- **reynolds_sweep.py**: Reynolds 수 sweep 해석
- **ffd_airfoil.py**: Free Form Deformation으로 airfoil 형상 변형 및 샘플 생성 (surrogate model용)

---

## ffd_airfoil.py

Free Form Deformation(FFD)를 사용하여 airfoil 형상을 변형하고, surrogate model 생성을 위한 다양한 airfoil 샘플을 생성합니다.

FFD는 제어점(control points) 격자를 사용하여 형상을 매개변수화하며, Bernstein polynomial 기반으로 부드러운 변형을 제공합니다.

### 주요 기능

- ✨ NACA baseline으로부터 FFD 변형 생성
- 📁 기존 airfoil 파일에 FFD 적용
- 🎲 Surrogate model을 위한 랜덤 샘플 생성
- 🎛️ 제어점 개수 및 변형 크기 조절
- 📊 선택적 시각화 (matplotlib)

### 사용법

#### 1. 단일 FFD airfoil 생성

```bash
# NACA baseline에서 랜덤 변형 생성
python scripts/ffd_airfoil.py --naca 0012 --control-points 5 3 --amplitude 0.02 -o output/airfoil/ffd_0012.dat

# 기존 airfoil 파일 변형
python scripts/ffd_airfoil.py --input public/airfoil/naca2412.dat --control-points 4 3 --amplitude 0.01 -o output/airfoil/ffd_2412.dat

# 시각화 포함
python scripts/ffd_airfoil.py --naca 0012 --control-points 6 3 --amplitude 0.03 -o output/airfoil/ffd_test.dat --plot
```

#### 2. Surrogate model을 위한 다중 샘플 생성

```bash
# 100개의 랜덤 샘플 생성
python scripts/ffd_airfoil.py --naca 0012 --samples 100 --control-points 5 3 --amplitude 0.02 --output-dir output/airfoil/naca0012_ffd

# 다른 Reynolds 수 범위를 위한 다양한 샘플
python scripts/ffd_airfoil.py --naca 2412 --samples 200 --control-points 6 4 --amplitude 0.03 --output-dir output/airfoil/naca2412_ffd --seed 123
```

#### 3. 특정 변형 파라미터 적용

```bash
# 변형 벡터를 직접 지정 (y-displacement)
python scripts/ffd_airfoil.py --naca 0012 --control-points 3 3 --deformation 0 0.01 0 0.02 0.03 0.01 0 0.005 0 -o specific_deform.dat

# 파일에서 변형 파라미터 읽기
python scripts/ffd_airfoil.py --input custom.dat --control-points 5 3 --deformation-file params.txt -o deformed.dat
```

### 파라미터 설명

- `--naca CODE`: NACA 4 또는 5-digit 코드 (baseline airfoil)
- `--input FILE`: 기존 airfoil 좌표 파일 사용
- `--control-points NX NY`: 제어점 개수 (x방향, y방향)
  - 기본값: `5 3`
  - NX: chord 방향 제어점 (많을수록 세밀한 조절)
  - NY: 두께 방향 제어점 (3이면 윗면/중간/아랫면)
- `--amplitude VALUE`: 변형 크기 (chord의 비율)
  - 기본값: `0.02` (chord의 2%)
  - 권장 범위: `0.01 ~ 0.05`
- `--samples N`: 생성할 랜덤 샘플 개수
- `--seed VALUE`: 재현성을 위한 랜덤 시드
- `--output, -o`: 출력 파일 경로 (단일 샘플용)
- `--output-dir`: 출력 디렉토리 (다중 샘플용)
- `--plot`: 원본 및 변형된 airfoil 시각화

### 출력 파일

#### 단일 샘플

- 지정한 경로에 `.dat` 파일 생성

#### 다중 샘플 (`--samples` 옵션 사용)

- `{output_dir}/NACA_XXXX_baseline.dat` - 원본 baseline
- `{output_dir}/ffd_sample_0000.dat` ~ `ffd_sample_NNNN.dat` - FFD 변형 샘플들
- `{output_dir}/deformation_parameters.txt` - 모든 샘플의 변형 파라미터 (NumPy 포맷)

### Surrogate Model 워크플로우 예제

```bash
# 1. 다양한 FFD 샘플 생성
python scripts/ffd_airfoil.py --naca 0012 --samples 100 --control-points 5 3 --amplitude 0.03 --output-dir output/airfoil/ffd_dataset

# 2. 각 샘플에 대해 AoA sweep 수행
for f in output/airfoil/ffd_dataset/ffd_sample_*.dat; do
    python scripts/aoa_sweep.py "$f" 1000000 -5 15 0.5
done

# 3. 결과를 통합하여 surrogate model 학습
# (별도의 ML 스크립트 필요)
```

### FFD 이론 배경

FFD(Free Form Deformation)는 다음과 같이 동작합니다:

1. **제어점 격자**: airfoil 주변에 NX × NY 제어점 격자 생성
2. **Bernstein 다항식**: 각 점을 격자의 parametric 좌표 (u, v) ∈ [0,1]²로 매핑
3. **변형 적용**: 제어점을 이동시키면 영향 범위 내 모든 점이 부드럽게 변형

수식:

```
P_deformed = Σᵢ Σⱼ Bᵢ,ₙ(u) × Bⱼ,ₘ(v) × Pᵢⱼ
```

여기서 Bᵢ,ₙ(u)는 Bernstein basis function입니다.

### Python에서 사용

```python
from ffd_airfoil import FFDAirfoil, load_airfoil, save_airfoil, generate_random_deformation
import numpy as np

# Baseline airfoil 로드
coords, name = load_airfoil("public/airfoil/naca0012.dat")

# FFD 초기화
ffd = FFDAirfoil(n_control_x=5, n_control_y=3)
ffd.setup_lattice(coords, padding=0.15)

# 랜덤 변형 생성 및 적용
deformation = generate_random_deformation(5, 3, amplitude=0.02)
ffd.apply_deformation(deformation)

# Airfoil 변형
deformed = ffd.deform_airfoil(coords)

# 저장
save_airfoil("ffd_output.dat", deformed, "FFD NACA 0012")
```

### 제어점 개수 선택 가이드

- **3 × 3**: 빠른 테스트용, 대략적인 변형
- **5 × 3**: 일반적인 용도, 충분한 자유도 (권장)
- **7 × 3**: 세밀한 제어, 복잡한 형상 변형
- **10 × 5**: 매우 세밀한 제어, 계산 비용 증가

일반적으로 y-방향은 3개면 충분 (윗면, 캠버선, 아랫면 제어)

### 주의사항

⚠️ **변형 크기(`--amplitude`)가 너무 크면:**

- Self-intersection 발생 가능
- XFOIL 해석 실패 가능
- 권장: 0.01 ~ 0.03 (chord의 1~3%)

⚠️ **제어점이 너무 많으면:**

- 계산 시간 증가
- Overfitting 위험 (surrogate model 학습 시)
- 일반적으로 5×3 또는 6×3이 적절

---

## generate_naca_airfoil.py

XFOIL을 사용하여 NACA airfoil 좌표 파일을 생성합니다.

## import_airfoil.py

외부 airfoil 좌표 파일을 검증하고 import합니다.

### 사용법

```bash
# Docker 컨테이너 내부에서
python scripts/import_airfoil.py /path/to/custom_airfoil.dat
python scripts/import_airfoil.py my_airfoil.dat
```

### 지원 형식

XFOIL이 지원하는 모든 형식을 자동으로 인식합니다:

1. **Plain coordinate file** - 좌표만 포함

   ```
   1.0000  0.0000
   0.9500  0.0100
   ...
   ```

2. **Labeled coordinate file** - 이름 + 좌표

   ```
   Custom Airfoil Name
   1.0000  0.0000
   0.9500  0.0100
   ...
   ```

3. **ISES coordinate file** - ISES 그리드 파라미터 포함
4. **MSES coordinate file** - 멀티 엘리먼트 형식

### 검증 과정

1. ✓ 파일 존재 확인
2. ✓ .dat 확장자 확인
3. ✓ Python으로 기본 형식 검증 (좌표 쌍 확인)
4. ✓ XFOIL로 실제 로드 테스트
5. ✓ 성공 시 `public/airfoil/`에 저장

### Python에서 사용

```python
from scripts.import_airfoil import import_airfoil

# Airfoil 파일 import
result = import_airfoil("my_custom_airfoil.dat")

if result:
    print(f"Successfully imported to: {result}")
```

---

## aoa_sweep.py

XFOIL의 ASEQ (Alpha Sequence) 명령을 사용하여 AoA sweep 해석을 수행합니다.

### 사용법

```bash
python scripts/aoa_sweep.py <AIRFOIL_FILE> <Re> <AoA_min> <AoA_max> <dAoA> [Ncrit]
```

### 매개변수

- `AIRFOIL_FILE`: Airfoil 좌표 파일 (.dat)
- `Re`: Reynolds 수
- `AoA_min`: 최소 받음각 (degrees)
- `AoA_max`: 최대 받음각 (degrees)
- `dAoA`: 받음각 증분 (degrees)
- `Ncrit`: 천이 기준 (선택, 기본값: 9)

### 예제

```bash
# 기본 사용
python scripts/aoa_sweep.py naca0012.dat 1000000 -5 15 0.5

# Ncrit 지정
python scripts/aoa_sweep.py naca0012.dat 1000000 -5 15 0.5 9

# 더 정밀한 sweep
python scripts/aoa_sweep.py public/airfoil/naca2412.dat 3e6 -10 25 0.25
```

### 출력

- `results/aoa_sweep/[airfoil]_Re[Re]_aoa[min]to[max].txt` - XFOIL polar 형식
- `results/aoa_sweep/[airfoil]_Re[Re]_aoa[min]to[max].csv` - CSV 형식
- `results/aoa_sweep/[airfoil]_Re[Re]_aoa[min]to[max]_dump.txt` - 상세 데이터

CSV 파일 컬럼: `alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr`

### 테스트 결과

✅ **NACA 0012 @ Re=1e6, α=-2°~10°**

- 11개 포인트 성공
- CL: -0.0 ~ 1.079
- L/D max: 75.25 @ α=7°

✅ **NACA 2412 @ Re=1e6, α=0°~12°**

- 24개 포인트 성공
- CL: 0.237 ~ 1.409
- L/D max: 104.71 @ α=4.5°

---

## reynolds_sweep.py

고정 AoA에서 Reynolds 수를 변화시켜가며 해석을 수행합니다.

### 사용법

```bash
python scripts/reynolds_sweep.py <AIRFOIL_FILE> <AoA> <Re_min> <Re_max> <dRe> [Ncrit]
```

### 매개변수

- `AIRFOIL_FILE`: Airfoil 좌표 파일 (.dat)
- `AoA`: 받음각 (degrees)
- `Re_min`: 최소 Reynolds 수
- `Re_max`: 최대 Reynolds 수
- `dRe`: Reynolds 수 증분
- `Ncrit`: 천이 기준 (선택, 기본값: 9)

### 예제

```bash
# 기본 사용
python scripts/reynolds_sweep.py naca0012.dat 5.0 1000000 5000000 500000

# Ncrit 지정
python scripts/reynolds_sweep.py naca0012.dat 5.0 1000000 5000000 500000 9

# 낮은 Reynolds 수 범위
python scripts/reynolds_sweep.py custom_airfoil.dat 0.0 50000 1000000 50000 5
```

### 출력

- `results/reynolds_sweep/[airfoil]_aoa[aoa]_Re[min]to[max].csv` - CSV 형식
- `results/reynolds_sweep/[airfoil]_aoa[aoa]_Re[min]to[max].txt` - 텍스트 형식

CSV 파일 컬럼: `alpha, Re, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr, converged`

**참고:** 큰 Re 범위의 경우 자동으로 로그 스페이싱을 사용합니다.

### 테스트 결과

✅ **NACA 0012 @ α=5°, Re=1e6~3e6**

- 5개 포인트 모두 수렴
- CL: 0.539 ~ 0.556
- L/D max: 80.56 @ Re=3e6

✅ **NACA 2412 @ α=8°, Re=5e5~2e6**

- 7개 포인트 모두 수렴
- CL: 1.071 ~ 1.102
- L/D max: 102.23 @ Re=2e6

---

## generate_naca_airfoil.py

XFOIL을 사용하여 NACA airfoil 좌표 파일을 생성합니다.

### 사용법

#### 단일 airfoil 생성

```bash
# Docker 컨테이너 내부에서
# 기본 160 포인트
python scripts/generate_naca_airfoil.py 0012

# 커스텀 포인트 수 지정
python scripts/generate_naca_airfoil.py 0012 200
python scripts/generate_naca_airfoil.py 2412 100

# 커스텀 출력 디렉토리 지정
python scripts/generate_naca_airfoil.py 23012 160 custom/output
```

#### 여러 airfoil 일괄 생성

```bash
python scripts/generate_naca_airfoil.py --batch
```

일반적으로 사용되는 NACA airfoil들을 자동으로 생성합니다:

- 대칭 airfoil: 0006, 0009, 0012, 0015, 0018, 0021
- 4-digit cambered: 2412, 2415, 4412, 4415
- 5-digit: 23012, 23015

### Python에서 사용

```python
from scripts.generate_naca_airfoil import generate_naca_airfoil

# NACA 0012 생성 (기본 160 포인트)
airfoil_file = generate_naca_airfoil("0012", output_dir="public/airfoil")

# 패널 포인트 수 조정
airfoil_file = generate_naca_airfoil("2412", output_dir="public/airfoil", num_points=200)

# 적은 포인트 수로 빠른 테스트
airfoil_file = generate_naca_airfoil("6409", output_dir="public/airfoil", num_points=80)
```

### 출력 형식

생성된 파일은 labeled coordinate 형식입니다:

```
NACA 0012
 1.00000  0.00000
 0.99500  0.00060
 ...
```

### NACA 코드 설명

#### 4-digit series (예: NACA 2412)

- 첫 번째 숫자 (2): 최대 캠버 위치 / 10 chord (20% 위치)
- 두 번째 숫자 (4): 최대 캠버 / 100 chord (4% chord)
- 마지막 두 숫자 (12): 최대 두께 / 100 chord (12% chord)

#### 5-digit series (예: NACA 23012)

- 처음 숫자 (2): 설계 양력계수 × 3/20
- 두 번째, 세 번째 숫자 (30): 최대 캠버 위치 / 2 percent chord (15%)
- 마지막 두 숫자 (12): 최대 두께 / 100 chord (12% chord)

## 필요 환경

- XFOIL이 설치되어 있고 PATH에 있어야 함
- Python 3.x

## Docker 환경에서 실행

```bash
# 컨테이너 접속
docker-compose exec xfoil-dev bash

# 스크립트 실행
python scripts/generate_naca_airfoil.py 0012

# 또는 실행 권한 부여 후
chmod +x scripts/generate_naca_airfoil.py
./scripts/generate_naca_airfoil.py 0012
```

## 출력 디렉토리

기본 출력 디렉토리는 `public/airfoil/`이며, 필요에 따라 변경 가능합니다.
