# SU2 Solver for Airfoil Analysis

SU2 RANS CFD solver optimized for 2D airfoil aerodynamics.

## Features

- **Compressible flow** (subsonic, transonic, supersonic)
- **Turbulence models**: SA, SST, Gamma-Re-theta
- **High Reynolds number** capability (Re > 1e6)
- **MPI + OpenMP** parallel support
- **Mesh formats**: SU2, CGNS
- **Output formats**: VTK/ParaView, TecIO, CSV

## Configuration Templates

Pre-configured templates in `config_templates/`:

| Template | Mach | Use Case |
|----------|------|----------|
| `subsonic_airfoil.cfg` | < 0.5 | General low-speed analysis with SA model |
| `transonic_airfoil.cfg` | 0.7-0.9 | Shock-boundary layer interaction with SST |
| `transition_airfoil.cfg` | < 0.3 | Transition prediction with Gamma-Re-theta |

## Quick Start

SU2 is included in the unified Docker container.

```bash
# Build (includes XFoil, NeuralFoil, SU2)
docker-compose build

# Run container
docker-compose up -d
docker exec -it airfoil-optim bash

# Verify SU2 installation
SU2_CFD -h
```

### Run Analysis

```bash
# Inside container
SU2_CFD config.cfg
```

## Configuration

See `../../scripts/su2_interface.py` for Python configuration generator.

Example:
```python
from solvers.su2_solver import SU2Config, SU2Solver

# Create configuration
config = SU2Config("transonic_airfoil")
config.set_flow_conditions(reynolds=3e6, mach=0.75, aoa=2.0)
config.set_turbulence_model('SST')
config.set_mesh("airfoil_mesh.su2")
config.set_boundary_conditions()
config.write("output/config.cfg")

# Run analysis
solver = SU2Solver()
success, results = solver.analyze("output/config.cfg", "output/")
```

## Mesh Requirements

SU2 requires a computational mesh. Supported formats:

- `.su2` (native)
- `.cgns` (CGNS)

**Recommended tools:**
- Gmsh (open source)
- Pointwise (commercial)
- ICEM CFD (commercial)

### Mesh Guidelines for Airfoil

- **Farfield distance**: 20-50 chord lengths
- **First cell height**: y+ < 1 for wall-resolved RANS
- **Growth ratio**: 1.1-1.2 in boundary layer
- **Minimum cells**: 50,000+ for 2D RANS

## Turbulence Models

| Model | Use Case | Accuracy | Cost |
|-------|----------|----------|------|
| SA | Attached flow, subsonic | Good | Low |
| SST | Separated flow, transonic | Very Good | Medium |
| Gamma-Re-theta | Transition | Excellent | High |

## Output

- `history.csv`: Convergence history (CL, CD, residuals)
- `surface.csv`: Surface pressure distribution
- `restart.dat`: Restart file for continuation
- `volume.vtu`: ParaView/VTK visualization
- `volume.plt`: TecPlot visualization (optional)
