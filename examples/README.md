# Examples - Multi-Solver Analysis Cases

이 디렉토리는 다양한 비행 조건에서의 airfoil 해석 예시를 포함합니다.

## Available Examples

### 1. `low_re_uav.sh`

**Small UAV / RC Aircraft**

- Reynolds: 2e5
- Mach: 0.1
- Solver: XFoil (optimal)
- Use case: Model aircraft, small drones

```bash
bash examples/low_re_uav.sh
```

### 2. `commercial_cruise.sh`

**Commercial Transport Aircraft**

- Reynolds: 6.5e6
- Mach: 0.729
- Solver: SU2 SST (transonic)
- Use case: Boeing 737, Airbus A320

```bash
bash examples/commercial_cruise.sh
```

### 3. `transonic_comparison.sh`

**Solver Comparison Study**

- Three cases comparing XFoil and SU2
- Shows transition from XFoil to SU2 validity range
- Demonstrates automatic solver selection

```bash
bash examples/transonic_comparison.sh
```

## Running Examples

### On Linux/Mac:

```bash
chmod +x examples/*.sh
bash examples/low_re_uav.sh
```

### On Windows (PowerShell):

```powershell
# Convert to PowerShell or run via WSL
wsl bash examples/low_re_uav.sh
```

### Or run Python directly:

```bash
python scripts/unified_analysis.py input/airfoil/naca0012.dat --re 2e5 --mach 0.1 --aoa 5.0
```

## Creating Custom Examples

Copy and modify the example scripts for your specific use case:

```bash
cp examples/low_re_uav.sh examples/my_case.sh
# Edit my_case.sh with your Re, Mach, airfoil
bash examples/my_case.sh
```

## Output

All example results are saved to `output/examples/` directory:

```
output/examples/
├── uav_low_re/
├── commercial_cruise/
└── comparison/
    ├── case1_xfoil/
    ├── case2_su2/
    └── case3_transonic/
```

## Notes

- XFoil examples will run immediately
- SU2 examples generate configuration files but require mesh generation
- Modify Re, Mach, AoA parameters for your specific aircraft
