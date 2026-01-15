# Multi-Solver Airfoil Analysis Framework

## Overview

이 프레임워크는 Reynolds 수와 Mach 수에 따라 자동으로 최적의 CFD solver를 선택하여 airfoil 해석을 수행합니다.

### Supported Solvers

1. **XFoil** - Panel method with boundary layer analysis

   - Best for: Re = 1e4 ~ 1e6, Mach < 0.5 (incompressible)
   - Pros: Fast, accurate for low-medium Re
   - Cons: Cannot handle compressible flow, struggles at very high Re

2. **SU2 RANS** - High-fidelity CFD solver
   - **SA (Spalart-Allmaras)**: General purpose, good for attached flows
   - **SST (k-ω SST)**: Better for separated flows and adverse pressure gradients
   - **Gamma-Re-theta**: Transition modeling for laminar-turbulent flows
   - Best for: Re > 1e6 or Mach > 0.5
   - Pros: Handles compressible flow, transonic, high Re
   - Cons: Slower, requires mesh generation

### Automatic Solver Selection Logic

```
If Mach >= 0.7 (Transonic):
    → SU2 SST (shock-boundary layer interaction)

Else if Mach >= 0.5 (Subsonic compressible):
    → SU2 SA (compressible flow)

Else if Re >= 1e6 (High Reynolds):
    → SU2 SA (high Re turbulence)

Else if Re >= 1e5 (Medium Reynolds):
    → XFoil (optimal range)

Else (Low Reynolds):
    → XFoil with adjusted Ncrit (laminar effects)
```

## Quick Start

### 1. Check Solver Availability

```bash
python scripts/unified_analysis.py --check
```

### 2. Single Point Analysis

```bash
# Low Re, incompressible (will use XFoil)
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 5e5 --mach 0.2 --aoa 5.0

# High Re, transonic (will use SU2 SST)
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 3e6 --mach 0.75 --aoa 2.5

# Large aircraft cruise (will use SU2 SA)
python scripts/unified_analysis.py input/airfoil/rae2822.dat \
    --re 6.5e6 --mach 0.729 --aoa 2.31
```

### 3. AoA Sweep

```bash
# Auto-select solver based on Re and Mach
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 1e6 --mach 0.3 --aoa-sweep -5 15 0.5
```

### 4. Force Specific Solver

```bash
# Force XFoil even for high Re (not recommended but possible)
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 3e6 --mach 0.3 --aoa 5.0 --solver xfoil

# Force SU2 SA for better accuracy
python scripts/unified_analysis.py input/airfoil/naca0012.dat \
    --re 5e5 --mach 0.2 --aoa 5.0 --solver su2_sa
```

## Detailed Usage

### Command-Line Options

```
python scripts/unified_analysis.py AIRFOIL_FILE --re RE --mach MACH [OPTIONS]

Required Arguments:
  AIRFOIL_FILE          Path to airfoil coordinate file (.dat)
  --re RE               Reynolds number

Optional Arguments:
  --mach MACH           Mach number (default: 0.0)
  --aoa AOA             Single angle of attack (degrees)
  --aoa-sweep MIN MAX STEP
                        AoA sweep: min max step (degrees)
  --solver {xfoil,su2_sa,su2_sst,su2_gamma_retheta}
                        Force specific solver (default: auto-select)
  --output-dir DIR      Output directory (default: output/analysis/unified)
  --check               Check solver availability and exit
```

## Use Cases

### Case 1: Small UAV / RC Aircraft

**Conditions**: Re ~ 1e5 - 5e5, Mach < 0.2

```bash
python scripts/unified_analysis.py input/airfoil/sd7037.dat \
    --re 200000 --mach 0.1 --aoa-sweep 0 12 0.5
```

→ **Auto-selects**: XFoil (optimal for this regime)

### Case 2: General Aviation

**Conditions**: Re ~ 5e5 - 2e6, Mach 0.2 - 0.4

```bash
python scripts/unified_analysis.py input/airfoil/naca2412.dat \
    --re 1e6 --mach 0.25 --aoa-sweep -5 15 0.5
```

→ **Auto-selects**: XFoil (still good range)

### Case 3: Commercial Transport (Subsonic)

**Conditions**: Re ~ 1e7, Mach 0.5 - 0.7

```bash
python scripts/unified_analysis.py input/airfoil/rae2822.dat \
    --re 1e7 --mach 0.65 --aoa 2.5
```

→ **Auto-selects**: SU2 SA (compressible, high Re)

### Case 4: Commercial Transport (Transonic)

**Conditions**: Re ~ 1e7, Mach 0.7 - 0.85

```bash
python scripts/unified_analysis.py input/airfoil/rae2822.dat \
    --re 6.5e6 --mach 0.729 --aoa 2.31
```

→ **Auto-selects**: SU2 SST (transonic shock waves)

### Case 5: High-Speed Business Jet

**Conditions**: Re ~ 5e6, Mach 0.85

```bash
python scripts/unified_analysis.py input/airfoil/supercritical.dat \
    --re 5e6 --mach 0.85 --aoa 1.0
```

→ **Auto-selects**: SU2 SST (transonic)

## Solver-Specific Settings

### XFoil Settings (Auto-adjusted)

| Reynolds Range | Ncrit | Notes             |
| -------------- | ----- | ----------------- |
| Re < 1e5       | 5.0   | More laminar flow |
| 1e5 ≤ Re < 5e5 | 7.5   | Transitional      |
| Re ≥ 5e5       | 9.0   | Fully turbulent   |

### SU2 Settings (Auto-adjusted)

| Condition             | Turbulence Model | CFL | Iterations |
| --------------------- | ---------------- | --- | ---------- |
| Mach < 0.7, Attached  | SA               | 5.0 | 5,000      |
| Mach < 0.7, Separated | SST              | 3.0 | 10,000     |
| Mach ≥ 0.7, Transonic | SST              | 1.0 | 10,000     |
| Transition Critical   | Gamma-Re-theta   | 2.0 | 15,000     |

## Output Files

### XFoil Output

```
output/analysis/unified/
  └── {airfoil}_Re{re}_M{mach}_aoa{aoa}.csv
      - alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
```

### SU2 Output

```
output/analysis/unified/{case_name}/
  ├── {case_name}.cfg          # SU2 configuration file
  ├── history.csv              # Convergence history
  ├── surface.csv              # Surface data
  └── restart_flow.dat         # Restart file
```

## Performance Comparison

| Solver  | Re Range | Mach Range | Speed       | Accuracy      | Setup     |
| ------- | -------- | ---------- | ----------- | ------------- | --------- |
| XFoil   | 1e4-1e6  | 0-0.5      | ⚡⚡⚡ Fast | ✓✓ Good       | ✓✓✓ Easy  |
| SU2 SA  | Any      | Any        | ⚡ Slow     | ✓✓✓ Excellent | ✓ Complex |
| SU2 SST | Any      | Any        | ⚡ Slower   | ✓✓✓ Excellent | ✓ Complex |

## Installation

### XFoil

```bash
# Already available in Docker container
# Or install locally: https://web.mit.edu/drela/Public/web/xfoil/
```

### SU2

```bash
# Docker container (recommended)
docker pull su2code/su2:latest

# Or install locally
# Ubuntu/Debian:
sudo apt-get install su2

# From source: https://su2code.github.io/
```

## Known Limitations

### XFoil

- ❌ Cannot handle compressible flow (Mach > 0.5)
- ❌ May fail at very high Re (> 1e7)
- ❌ No shock wave modeling
- ❌ Panel method limitations

### SU2

- ❌ Requires mesh generation (not automated yet)
- ❌ Slower than XFoil for simple cases
- ❌ More complex setup
- ❌ Higher computational cost

## Troubleshooting

### "XFoil not found"

```bash
# Check if xfoil is in PATH
which xfoil

# If using Docker, ensure xfoil is installed in container
docker exec -it airfoil-container which xfoil
```

### "SU2 not found"

```bash
# Check if SU2_CFD is in PATH
which SU2_CFD

# Install SU2 or use Docker container
```

### XFoil fails to converge

- Try lower Ncrit (5-7) for low Re
- Reduce AoA step size (e.g., 0.25° instead of 0.5°)
- Increase iteration limit
- Consider using SU2 for problematic cases

### SU2 requires mesh

- Currently, SU2 configuration is generated but mesh generation is manual
- Use gmsh or other mesh generator to create airfoil mesh
- Or wait for automated mesh generation feature (coming soon)

## Examples

See `examples/` directory for complete example cases:

- `examples/low_re_uav.sh` - UAV case with XFoil
- `examples/commercial_cruise.sh` - Boeing 737 cruise with SU2
- `examples/transonic_comparison.sh` - XFoil vs SU2 comparison

## References

1. Drela, M. "XFOIL: An Analysis and Design System for Low Reynolds Number Airfoils"
2. Economon, T.D. et al. "SU2: An Open-Source Suite for Multiphysics Simulation and Design"
3. Spalart, P.R. and Allmaras, S.R. "A One-Equation Turbulence Model for Aerodynamic Flows"
4. Menter, F.R. "Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications"
