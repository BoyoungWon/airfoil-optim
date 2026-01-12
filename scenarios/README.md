# Optimization Scenarios

이 디렉토리는 다양한 airfoil 최적화 시나리오 설정 파일을 포함합니다.

## 시나리오 분류

### A. 고정익 항공기 (Fixed-Wing Aircraft)

- `cruise_wing.yaml` - 순항 기준 최적화
- `high_lift.yaml` - 고양력 익형
- `low_speed.yaml` - 저속 고효율 (UAV)
- `transonic.yaml` - 천이 환경

### B. 회전익 시스템 (Rotorcraft/Propeller)

- `propeller.yaml` - 프로펠러 익형
- `helicopter_rotor.yaml` - 헬리콥터 로터
- `wind_turbine.yaml` - 풍력터빈

### C. 제어면 (Control Surfaces)

- `control_surface.yaml` - 방향타/승강타
- `high_lift_device.yaml` - 플랩/슬랫

## YAML 파일 구조

```yaml
# 시나리오 메타데이터
name: "시나리오 이름"
description: "상세 설명"
category: "fixed_wing" | "rotorcraft" | "control_surface"
application: "cruise_wing" | "propeller" | etc.

# 형상 매개변수화 방법
parametrization:
  method: "naca" | "cst" | "ffd"
  baseline: "naca0012" | "path/to/baseline.dat"
  parameters:
    # method별로 다름

# 설계점 (운용 조건)
design_points:
  - name: "cruise"
    reynolds: 3.0e6
    aoa: 3.0
    mach: 0.2
    weight: 1.0  # 다점 최적화시 가중치

# 목적함수
objectives:
  - type: "maximize" | "minimize"
    metric: "CL/CD" | "CL" | "CD" | "efficiency"
    weight: 1.0  # 다목적 최적화시

# 제약 조건
constraints:
  geometry:
    - param: "thickness_ratio"
      min: 0.10
      max: 0.20
  aerodynamic:
    - param: "CL"
      min: 0.5
    - param: "CM"
      min: -0.10
      max: 0.05

# 최적화 설정
optimization:
  algorithm: "scipy" | "genetic" | "bayesian"
  max_iterations: 100
  convergence_tol: 1e-6

# Surrogate model 설정
surrogate:
  method: "kriging" | "neural_network" | "polynomial"
  training_samples: 200
  validation_split: 0.2
```

## 사용 예제

### 단일 시나리오 최적화

```bash
python scripts/optimize_airfoil.py --scenario scenarios/cruise_wing.yaml
```

### 여러 시나리오 비교

```bash
python scripts/compare_scenarios.py --scenarios scenarios/cruise_wing.yaml scenarios/high_lift.yaml
```

### 시나리오 검증

```bash
python scripts/validate_scenario.py scenarios/cruise_wing.yaml
```

## 커스텀 시나리오 생성

1. 기존 시나리오 복사
2. 파라미터 수정
3. 검증 실행
4. 최적화 실행
