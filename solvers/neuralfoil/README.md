# NeuralFoil Solver

Neural network surrogate for fast airfoil aerodynamics.

## Features

- **Very fast**: ~0.004s per analysis (100x faster than XFoil)
- **Stable**: No convergence issues
- **Confidence scores**: Know when predictions are reliable
- **Boundary layer**: Full BL parameter prediction

## Limitations

- Incompressible only (Mach < 0.5)
- Trained range: Re 1e4 - 1e7
- Approximation (not exact solution)

## Usage

### Python API

```python
from solvers.neuralfoil_solver import NeuralFoilSolver

solver = NeuralFoilSolver(model_size='xlarge')
result = solver.analyze(
    airfoil_file='naca0012.dat',
    reynolds=5e5,
    alpha=5.0
)

print(f"CL = {result['CL']:.6f}")
print(f"CD = {result['CD']:.6f}")
print(f"Confidence = {result['analysis_confidence']:.3f}")
```

### Vectorized Sweep (Fast!)

```python
import numpy as np

alphas = np.linspace(-5, 15, 41)  # 41 points
results = solver.analyze_sweep('naca0012.dat', re=5e5, alpha_range=alphas)
# Takes ~0.02s total for all 41 points!
```

### Docker Container

NeuralFoil is included in the unified Docker container.

```bash
# Build and run
docker-compose up -d
docker exec -it airfoil-optim bash

# Test NeuralFoil
python -c "from neuralfoil.main import get_aero_from_dat_file; print('OK')"
```

## Model Sizes

| Size | Speed | Accuracy | Recommended |
|------|-------|----------|-------------|
| xxsmall | Fastest | Lower | Mobile/embedded |
| small | Fast | Good | Quick screening |
| large | Medium | Very Good | General use |
| **xlarge** | Medium | **Excellent** | **Default** |
| xxxlarge | Slower | Best | Validation |

## Output Format

```python
{
    'reynolds': 500000.0,
    'aoa': 5.0,
    'mach': 0.0,
    'CL': 0.621293,
    'CD': 0.016065,
    'CM': -0.002888,
    'Top_Xtr': 0.068,      # Upper surface transition
    'Bot_Xtr': 1.000,      # Lower surface transition
    'analysis_confidence': 0.979,  # 0-1, higher = more reliable
    'converged': True,
    'solver': 'neuralfoil'
}
```

## Confidence Interpretation

- **> 0.8**: High confidence, within training distribution
- **0.5-0.8**: Moderate confidence, extrapolation possible
- **< 0.5**: Low confidence, results may be inaccurate
