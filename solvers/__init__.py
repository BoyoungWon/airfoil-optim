"""
Airfoil Analysis Solvers

통합 solver 패키지: XFoil, NeuralFoil, SU2

Usage:
    from solvers import SolverType, run_analysis
    
    # Auto-select solver based on conditions
    result = run_analysis(
        airfoil_file="input/airfoil/naca0012.dat",
        reynolds=5e5,
        mach=0.2,
        aoa=5.0
    )
    
    # Use specific solver
    from solvers.neuralfoil_solver import NeuralFoilSolver
    result = NeuralFoilSolver.analyze("naca0012.dat", re=5e5, alpha=5.0)
"""

from enum import Enum
from typing import Optional, Dict, Tuple
from pathlib import Path


class SolverType(Enum):
    """Available solver types"""
    XFOIL = "xfoil"
    NEURALFOIL = "neuralfoil"
    SU2_SA = "su2_sa"
    SU2_SST = "su2_sst"
    SU2_TRANSITION = "su2_transition"


class SolverCapabilities:
    """Solver capability matrix"""
    
    CAPABILITIES = {
        SolverType.XFOIL: {
            'reynolds_range': (1e4, 1e6),
            'mach_range': (0.0, 0.5),
            'compressible': False,
            'transition': True,
            'speed': 'fast',
            'accuracy': 'high',
        },
        SolverType.NEURALFOIL: {
            'reynolds_range': (1e4, 1e7),
            'mach_range': (0.0, 0.5),
            'compressible': False,
            'transition': True,
            'speed': 'very_fast',
            'accuracy': 'medium',
        },
        SolverType.SU2_SA: {
            'reynolds_range': (1e5, 1e8),
            'mach_range': (0.0, 2.0),
            'compressible': True,
            'transition': False,
            'speed': 'slow',
            'accuracy': 'high',
        },
        SolverType.SU2_SST: {
            'reynolds_range': (1e5, 1e8),
            'mach_range': (0.0, 2.0),
            'compressible': True,
            'transition': False,
            'speed': 'slow',
            'accuracy': 'very_high',
        },
        SolverType.SU2_TRANSITION: {
            'reynolds_range': (1e5, 1e8),
            'mach_range': (0.0, 2.0),
            'compressible': True,
            'transition': True,
            'speed': 'very_slow',
            'accuracy': 'very_high',
        },
    }
    
    @classmethod
    def is_suitable(cls, solver: SolverType, reynolds: float, mach: float) -> bool:
        """Check if solver is suitable for given conditions"""
        caps = cls.CAPABILITIES.get(solver, {})
        re_range = caps.get('reynolds_range', (0, float('inf')))
        mach_range = caps.get('mach_range', (0, float('inf')))
        
        return (re_range[0] <= reynolds <= re_range[1] and 
                mach_range[0] <= mach <= mach_range[1])


def get_available_solvers() -> Dict[str, bool]:
    """Check which solvers are available"""
    availability = {}
    
    # Check XFoil
    try:
        import subprocess
        result = subprocess.run(['xfoil'], capture_output=True, timeout=2, 
                               input=b'quit\n')
        availability['xfoil'] = True
    except:
        availability['xfoil'] = False
    
    # Check NeuralFoil
    try:
        import neuralfoil
        availability['neuralfoil'] = True
    except ImportError:
        availability['neuralfoil'] = False
    
    # Check SU2
    try:
        import subprocess
        result = subprocess.run(['SU2_CFD', '-h'], capture_output=True, timeout=5)
        availability['su2'] = True
    except:
        availability['su2'] = False
    
    return availability


def select_solver(reynolds: float, mach: float, 
                 preference: Optional[SolverType] = None) -> SolverType:
    """
    Auto-select best solver for given conditions
    
    Selection Logic:
    1. If user preference is valid for conditions, use it
    2. Mach >= 0.5 → SU2 (compressible required)
    3. Re >= 1e6 → SU2 SA (high Re RANS)
    4. Default → NeuralFoil (fast, reliable)
    """
    avail = get_available_solvers()
    
    # User preference
    if preference is not None:
        if SolverCapabilities.is_suitable(preference, reynolds, mach):
            return preference
    
    # Compressible flow requires SU2
    if mach >= 0.5:
        if avail.get('su2'):
            return SolverType.SU2_SST if mach >= 0.7 else SolverType.SU2_SA
        else:
            raise ValueError(f"SU2 required for Mach={mach} but not available")
    
    # High Reynolds
    if reynolds >= 1e6:
        if avail.get('su2'):
            return SolverType.SU2_SA
        elif avail.get('neuralfoil'):
            return SolverType.NEURALFOIL
        elif avail.get('xfoil'):
            return SolverType.XFOIL
    
    # Default: NeuralFoil (fastest, most reliable)
    if avail.get('neuralfoil'):
        return SolverType.NEURALFOIL
    elif avail.get('xfoil'):
        return SolverType.XFOIL
    else:
        raise ValueError("No suitable solver available")


# Version info
__version__ = "1.0.0"
__all__ = [
    'SolverType',
    'SolverCapabilities', 
    'get_available_solvers',
    'select_solver',
]
