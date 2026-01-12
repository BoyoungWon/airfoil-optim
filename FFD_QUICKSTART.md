# FFD Airfoil Generator - Quick Start Guide

## 개요

Free Form Deformation(FFD)을 사용하여 airfoil 형상을 변형하고, surrogate model 생성을 위한 다양한 샘플을 생성하는 도구입니다.

## 설치 및 환경

이미 Docker 환경에 모든 의존성이 설치되어 있습니다:

- Python 3.12
- NumPy 2.3.5
- XFOIL

## 주요 기능

### 1. 단일 FFD Airfoil 생성

```bash
# NACA baseline에서 생성
python scripts/ffd_airfoil.py --naca 0012 --control-points 5 3 --amplitude 0.02 -o output/ffd_0012.dat

# 기존 airfoil 파일 변형
python scripts/ffd_airfoil.py --input custom.dat --control-points 5 3 --amplitude 0.02 -o output/custom_ffd.dat
```

### 2. Surrogate Model용 다중 샘플 생성

```bash
# 100개 샘플 생성
python scripts/ffd_airfoil.py --naca 0012 --samples 100 --control-points 5 3 \
    --amplitude 0.02 --output-dir output/ffd_dataset --seed 42
```

출력:

- `output/ffd_dataset/NACA_0012_baseline.dat` - 원본 baseline
- `output/ffd_dataset/ffd_sample_0000.dat` ~ `ffd_sample_0099.dat` - 변형 샘플
- `output/ffd_dataset/deformation_parameters.txt` - 변형 파라미터 (NumPy 형식)

### 3. 테스트 실행

```bash
# 빠른 테스트
python scripts/test_ffd.py
```

## 파라미터 가이드

### 제어점 개수 (`--control-points NX NY`)

- **3 × 3**: 빠른 테스트용
- **5 × 3**: 일반적인 용도 (권장) ⭐
- **7 × 3**: 세밀한 제어
- **10 × 5**: 매우 세밀한 제어 (계산 비용 증가)

### 변형 크기 (`--amplitude`)

- **0.01**: 작은 변형 (1% chord)
- **0.02**: 중간 변형 (2% chord) - 권장 ⭐
- **0.03**: 큰 변형 (3% chord)
- **> 0.05**: 너무 큼 (self-intersection 위험)

### 샘플 개수 (`--samples`)

- **10-50**: 빠른 테스트
- **100-200**: 기본 surrogate model ⭐
- **500-1000**: 고품질 surrogate model
- **> 1000**: 매우 정밀한 모델 (계산 시간 증가)

## 워크플로우 예제

### Scenario 1: Surrogate Model 생성

```bash
# Step 1: FFD 샘플 생성
python scripts/ffd_airfoil.py --naca 0012 --samples 100 \
    --control-points 5 3 --amplitude 0.02 \
    --output-dir output/naca0012_ffd

# Step 2: 각 샘플에 대해 XFOIL 해석 수행
cd output/naca0012_ffd
for file in ffd_sample_*.dat; do
    python ../../scripts/aoa_sweep.py "$file" 1000000 -5 15 0.5
done

# Step 3: 결과 통합 및 surrogate model 학습
# (별도 ML 스크립트 필요)
```

### Scenario 2: 다양한 설계 탐색

```bash
# 여러 NACA baseline에서 샘플 생성
for naca in 0012 2412 4412; do
    python scripts/ffd_airfoil.py --naca $naca --samples 50 \
        --control-points 5 3 --amplitude 0.02 \
        --output-dir output/naca${naca}_ffd
done
```

### Scenario 3: 파라미터 민감도 분석

```bash
# 다양한 amplitude로 샘플 생성
for amp in 0.01 0.02 0.03; do
    python scripts/ffd_airfoil.py --naca 0012 --samples 20 \
        --control-points 5 3 --amplitude $amp \
        --output-dir output/amplitude_${amp}
done
```

## Jupyter Notebook 튜토리얼

대화형 튜토리얼이 제공됩니다:

```bash
# Jupyter 서버 시작
docker-compose up -d jupyter

# 브라우저에서 http://localhost:8888 접속
# FFD_Tutorial.ipynb 열기
```

튜토리얼 내용:

1. FFD 기본 개념
2. Baseline airfoil 생성
3. 제어점 격자 설정
4. 변형 적용 및 시각화
5. 다중 샘플 생성
6. Surrogate model 워크플로우

## 출력 파일 형식

### Airfoil 좌표 파일 (.dat)

```
FFD Sample 0001
  1.00000000    0.00123456
  0.99500000    0.00234567
  ...
```

XFOIL과 호환되는 표준 형식

### 변형 파라미터 파일 (.txt)

NumPy 텍스트 형식으로 저장:

- 각 행 = 하나의 샘플
- 각 열 = 제어점의 y-displacement
- Shape: `(n_samples, n_control_x * n_control_y)`

```python
# Python에서 읽기
import numpy as np
parameters = np.loadtxt('deformation_parameters.txt')
```

## Python API 사용

```python
from ffd_airfoil import FFDAirfoil, load_airfoil, save_airfoil
import numpy as np

# 1. Baseline 로드
coords, name = load_airfoil("naca0012.dat")

# 2. FFD 초기화
ffd = FFDAirfoil(n_control_x=5, n_control_y=3)
ffd.setup_lattice(coords, padding=0.15)

# 3. 변형 적용
deformation = np.random.uniform(-0.02, 0.02, (5, 3))
ffd.apply_deformation(deformation)

# 4. Airfoil 변형
deformed = ffd.deform_airfoil(coords)

# 5. 저장
save_airfoil("output.dat", deformed, "FFD Airfoil")
```

## 시각화 (Optional)

matplotlib이 설치된 경우:

```bash
python scripts/ffd_airfoil.py --naca 0012 --control-points 5 3 \
    --amplitude 0.02 -o output.dat --plot
```

## 성능 팁

1. **병렬 처리**: 여러 샘플 생성 시 배치로 나눠서 실행
2. **적절한 제어점 개수**: 5×3이면 대부분의 경우 충분
3. **샘플 개수**: 100-200개로 시작하여 필요시 증가
4. **Amplitude 조절**: 0.02부터 시작하여 조정

## 문제 해결

### XFOIL 실행 실패

```bash
# XFOIL 경로 확인
which xfoil

# 컨테이너 재시작
docker-compose restart xfoil-dev
```

### NumPy 오류

```bash
# 환경 확인
python -c "import numpy; print(numpy.__version__)"

# 환경 재생성 (필요시)
conda env update -f environment.yml
```

### 메모리 부족 (대량 샘플 생성 시)

```bash
# 배치 단위로 생성
for i in {0..9}; do
    python scripts/ffd_airfoil.py --naca 0012 --samples 100 \
        --output-dir output/batch_${i} --seed $i
done
```

## 추가 자료

- [scripts/README.md](scripts/README.md) - 전체 스크립트 문서
- [FFD_Tutorial.ipynb](FFD_Tutorial.ipynb) - Jupyter 튜토리얼
- [xfoil_doc.txt](xfoil/xfoil_doc.txt) - XFOIL 문서

## 예제 결과

테스트 실행 결과:

```bash
$ python scripts/test_ffd.py

🎯 Test Summary
Tests Passed: 4/4
✅ All tests passed!

📂 Check results in: test_ffd_output/

Generated files:
  - test_single_ffd.dat (4.3K)
  - output/NACA_2412_baseline.dat (4.3K)
  - output/ffd_sample_0000.dat ~ 0004.dat (4.3K each)
  - output/deformation_parameters.txt (1.6K)
```

## 라이선스

이 프로젝트는 XFOIL의 라이선스(GNU GPL v2.0)를 따릅니다.

## 문의

프로젝트 이슈 또는 질문이 있으시면 GitHub Issues를 이용해주세요.
