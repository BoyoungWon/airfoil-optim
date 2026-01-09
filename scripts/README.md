# XFOIL 스크립트

이 디렉토리는 XFOIL을 자동화하기 위한 유틸리티 스크립트를 포함합니다.

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
