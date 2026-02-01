# NeuralFoil 학습 가이드

FX63-137과 같은 특수 에어포일을 NeuralFoil로 예측하기 위한 학습 방법

## 📊 현재 문제점

### XFoil 수렴 실패
- **FX63-137** 에어포일이 Re=80,000~200,000 범위에서 XFoil 수렴 실패
- 원인: 복잡한 형상, 저 레이놀즈 수, boundary layer separation 예상

### NeuralFoil 예측 불가
- Re=80,000에서 **신뢰도 ≈ 0**, CD=inf 발생
- 훈련 데이터 분포를 벗어난 에어포일 형상

---

## 🎯 해결 방법: NeuralFoil 재학습

### 방법 1: 기존 모델 Fine-tuning (권장)

NeuralFoil의 사전 학습된 모델을 FX63-137 데이터로 fine-tuning

#### 1-1. 학습 데이터 생성

```bash
# XFoil로 다양한 조건에서 데이터 생성
cd /home/peterwon/airfoil-optim/neuralfoil

# 학습 데이터 디렉토리 생성
mkdir -p training_data/fx63_137

# Python 스크립트로 XFoil 데이터 생성
python << 'EOF'
import subprocess
import numpy as np
from pathlib import Path

# 학습할 조건 범위
reynolds_range = [5e4, 8e4, 1e5, 1.5e5, 2e5, 3e5, 5e5, 1e6]
alpha_range = np.arange(-15, 20, 0.5)
ncrit_values = [7.0, 7.5, 8.0, 9.0]

output_dir = Path("training_data/fx63_137")
output_dir.mkdir(parents=True, exist_ok=True)

for reynolds in reynolds_range:
    for ncrit in ncrit_values:
        output_file = output_dir / f"fx63_Re{reynolds:.0e}_N{ncrit}.txt"
        
        xfoil_cmd = f"""
PLOP
G

LOAD /home/peterwon/airfoil-optim/input/airfoil/FX63-137_normalized.dat
PANE

OPER
VISC {reynolds}
ITER 200
VPAR
N
{ncrit}

PACC
{output_file}

ASEQ {alpha_range.min()} {alpha_range.max()} 0.5

QUIT
"""
        
        print(f"Generating: Re={reynolds:.0e}, Ncrit={ncrit}")
        proc = subprocess.run(
            ['xfoil'],
            input=xfoil_cmd,
            text=True,
            capture_output=True,
            timeout=300
        )
        
        if output_file.exists() and output_file.stat().st_size > 1000:
            print(f"  ✓ Success: {output_file.stat().st_size} bytes")
        else:
            print(f"  ✗ Failed or insufficient data")

print("\n학습 데이터 생성 완료!")
EOF
```

#### 1-2. 데이터 형식 변환

```python
# NeuralFoil 학습 형식으로 변환
cd /home/peterwon/airfoil-optim/neuralfoil/training

# 데이터 포맷 변환 스크립트
python << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path

def parse_xfoil_polar(file_path):
    """XFoil polar 파일을 pandas DataFrame으로 변환"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 헤더 찾기
    data_start = None
    for i, line in enumerate(lines):
        if 'alpha' in line.lower() and 'CL' in line:
            data_start = i + 1
            break
    
    if data_start is None:
        return None
    
    # 데이터 파싱
    data = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 7:
            try:
                data.append([float(x) for x in parts[:7]])
            except ValueError:
                continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data, columns=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'])
    return df

# 모든 polar 파일 변환
training_dir = Path("../training_data/fx63_137")
all_data = []

for polar_file in training_dir.glob("*.txt"):
    df = parse_xfoil_polar(polar_file)
    if df is not None and len(df) > 0:
        # Re와 Ncrit 추출
        filename = polar_file.stem
        # 파일명 파싱: fx63_Re8e+04_N7.0.txt
        parts = filename.split('_')
        re_str = parts[1].replace('Re', '')
        ncrit_str = parts[2].replace('N', '')
        
        df['Re'] = float(re_str)
        df['Ncrit'] = float(ncrit_str)
        df['airfoil'] = 'FX63-137'
        
        all_data.append(df)
        print(f"✓ Converted: {polar_file.name} ({len(df)} points)")

# 통합 데이터셋 생성
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("fx63_137_training_data.csv", index=False)
    print(f"\n✓ 총 학습 데이터: {len(combined_df)} points")
    print(f"  Reynolds 범위: {combined_df['Re'].min():.0e} ~ {combined_df['Re'].max():.0e}")
    print(f"  Alpha 범위: {combined_df['alpha'].min():.1f}° ~ {combined_df['alpha'].max():.1f}°")
else:
    print("✗ 변환할 데이터가 없습니다")
EOF
```

#### 1-3. Fine-tuning 실행

```python
# NeuralFoil 모델 fine-tuning
cd /home/peterwon/airfoil-optim/neuralfoil

python << 'EOF'
import sys
sys.path.insert(0, '.')

from neuralfoil import NeuralFoil
import pandas as pd
import numpy as np

# 학습 데이터 로드
train_data = pd.read_csv("training/fx63_137_training_data.csv")

print("학습 데이터 통계:")
print(train_data.describe())

# 에어포일 좌표 로드
import aerosandbox as asb
airfoil = asb.Airfoil(
    coordinates="/home/peterwon/airfoil-optim/input/airfoil/FX63-137_normalized.dat"
)

# NeuralFoil 모델 로드 (사전 학습된 모델)
model = NeuralFoil(model_size="xlarge")

# Fine-tuning 설정
from neuralfoil.training import train_model

# 학습 파라미터
train_params = {
    'learning_rate': 1e-5,  # Fine-tuning은 작은 learning rate 사용
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'early_stopping_patience': 10
}

# Fine-tuning 실행
print("\nFine-tuning 시작...")
history = train_model(
    model=model,
    train_data=train_data,
    airfoil_coords=airfoil.coordinates,
    **train_params
)

# 모델 저장
model.save("models/fx63_137_finetuned.pkl")
print("\n✓ 모델 저장 완료: models/fx63_137_finetuned.pkl")

# 테스트
print("\n테스트 예측:")
test_result = model.get_aero(
    alpha=5.0,
    Re=80000,
    n_crit=7.2
)
print(f"  Alpha=5°, Re=8e4:")
print(f"  CL = {test_result['CL']:.4f}")
print(f"  CD = {test_result['CD']:.6f}")
print(f"  L/D = {test_result['CL']/test_result['CD']:.2f}")
print(f"  Confidence = {test_result['analysis_confidence']:.3f}")
EOF
```

---

### 방법 2: 데이터 증강 (Data Augmentation)

XFoil이 수렴하지 않는 경우 CFD 시뮬레이션 또는 실험 데이터 사용

#### 2-1. SU2로 CFD 데이터 생성

```bash
# SU2 (고정밀 CFD)로 학습 데이터 생성
# Docker 환경 사용 권장

# FX63-137 메시 생성
gmsh -2 fx63_137.geo -o fx63_137.su2

# SU2 시뮬레이션 실행 (다양한 조건)
for re in 50000 80000 100000 150000 200000; do
  for alpha in -15 -10 -5 0 5 10 15 20; do
    echo "Running: Re=$re, Alpha=$alpha"
    # SU2 설정 파일 생성 및 실행
    # ... (SU2 실행 스크립트)
  done
done
```

#### 2-2. Transfer Learning

다른 유사 에어포일 데이터로 보완

```python
# 유사한 에어포일 (FX 시리즈, Wortmann 계열) 데이터 추가
similar_airfoils = [
    "FX60-126",
    "FX63-120", 
    "FX66-S-196",
    "Wortmann FX74-CL5-140"
]

# 각 에어포일에 대해 XFoil 데이터 생성
# 통합 학습 데이터셋 구축
```

---

### 방법 3: Physics-Informed Learning

물리 제약 조건을 추가하여 학습 안정성 향상

```python
# 물리 법칙 기반 손실 함수 추가
def physics_informed_loss(predictions, targets):
    """
    물리 법칙을 위반하지 않도록 제약
    - CL과 alpha의 선형 관계 (작은 alpha에서)
    - CD > CDp (항상 true)
    - Kutta 조건: TE에서 압력 연속성
    """
    mse_loss = nn.MSELoss()(predictions, targets)
    
    # 물리 제약 손실
    cl = predictions[:, 0]
    cd = predictions[:, 1]
    cdp = predictions[:, 2]
    alpha = inputs[:, 0]
    
    # 제약 조건
    physics_loss = (
        torch.relu(cdp - cd) +  # CD >= CDp
        torch.relu(-cl * alpha) * 0.1  # 작은 alpha에서 CL과 alpha 동일 부호
    )
    
    return mse_loss + 0.1 * physics_loss.mean()
```

---

## 🔧 실전 권장 워크플로우

### 단계별 접근

1. **데이터 확보** (최우선)
   - XFoil로 수렴 가능한 조건부터 데이터 생성
   - Re > 100,000, -10° < alpha < 15° 범위
   - 최소 500~1000 데이터 포인트 확보

2. **기존 모델 활용**
   - NeuralFoil pre-trained 모델 사용
   - Confidence score가 0.5 이상인 조건에서만 사용
   - 낮은 신뢰도 구간은 보간/외삽 대신 XFoil/CFD 사용

3. **점진적 확장**
   ```python
   # 단계적 레이놀즈 수 확장
   step1_re = [1e5, 1.5e5, 2e5, 3e5, 5e5]  # 안정 영역
   step2_re = [8e4, 9e4] + step1_re          # 저Re 확장
   step3_re = [5e4, 6e4, 7e4] + step2_re     # 극저Re 확장
   ```

4. **검증**
   - 학습 후 known test case와 비교
   - Confidence score 모니터링
   - 물리적으로 타당한 결과 확인 (CL-alpha 곡선, stall 특성)

---

## 📈 성능 향상 팁

### 데이터 품질
- **다양성**: Reynolds, alpha, Ncrit 다양한 조합
- **밀도**: Critical 영역(stall 근처)은 더 촘촘하게
- **품질 검증**: 명백히 잘못된 데이터 제거

### 모델 구조
```python
# Ensemble 모델로 불확실성 추정
models = [
    NeuralFoil(model_size="large"),
    NeuralFoil(model_size="xlarge"),
    NeuralFoil(model_size="xxlarge")
]

# 예측 평균 + 표준편차로 신뢰도 계산
predictions = [m.predict(alpha, Re) for m in models]
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)
confidence = 1.0 / (1.0 + std_pred)
```

### 하이퍼파라미터
```python
optimal_params = {
    'learning_rate': 1e-5,      # Fine-tuning용
    'batch_size': 32,
    'dropout': 0.2,             # Overfitting 방지
    'weight_decay': 1e-4,
    'epochs': 100,
    'early_stopping': True
}
```

---

## ⚠️ 주의사항

1. **Re=80,000은 매우 어려운 조건**
   - XFoil조차 수렴 어려움
   - Laminar separation bubble 형성
   - Neural network도 예측 어려움

2. **FX63-137 특성**
   - 고양력 에어포일 (high camber)
   - 저속 글라이더/UAV용
   - Re > 150,000에서 더 안정적

3. **대안**
   - **Re를 100,000 이상으로 상향** 권장
   - 또는 **풍동 실험 데이터** 확보
   - **CFD (SU2, OpenFOAM)** 시뮬레이션 활용

---

## 📚 참고 자료

- NeuralFoil 공식 문서: `/home/peterwon/airfoil-optim/neuralfoil/README.md`
- Training examples: `/home/peterwon/airfoil-optim/neuralfoil/training/`
- Benchmarking scripts: `/home/peterwon/airfoil-optim/neuralfoil/benchmarking/`

---

## 💡 즉시 실행 가능한 대안

Re=80,000이 필수가 아니라면:

```bash
# Re=200,000으로 실행 (훨씬 안정적)
python scripts/unified_analysis.py \
    input/airfoil/FX63-137_normalized.dat \
    --re 200000 \
    --mach 0.022 \
    --aoa-sweep -15 20 1 \
    --solver xfoil \
    --ncrit 7.2

# 또는 NeuralFoil (Re > 100k)
python scripts/unified_analysis.py \
    input/airfoil/FX63-137_normalized.dat \
    --re 200000 \
    --mach 0.022 \
    --aoa-sweep -15 20 1 \
    --solver neuralfoil \
    --ncrit 7.2
```

이렇게 하면 즉시 결과를 얻을 수 있습니다! 🚀
