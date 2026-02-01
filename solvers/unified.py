"""
Unified Airfoil Analysis Interface

통합 airfoil 공력 해석 인터페이스
- 자동 solver 선택 (Re, Mach 기반)
- NeuralFoil 기본, XFoil/SU2 fallback
- 단일 포인트 및 sweep 해석 지원

Usage:
    from solvers.unified import analyze
    
    # Auto-select best solver
    result = analyze("naca0012.dat", re=5e5, alpha=5.0)
    
    # Force specific solver
    result = analyze("naca0012.dat", re=5e5, alpha=5.0, solver='xfoil')
    
    # AoA sweep
    results = analyze_sweep("naca0012.dat", re=5e5, alpha_range=(-5, 15, 1.0))
"""

from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import numpy as np

from . import SolverType, get_available_solvers, select_solver
from .neuralfoil_solver import NeuralFoilSolver
from .xfoil_solver import XFoilSolver
from .su2_solver import SU2Solver, SU2Config


def analyze(airfoil_file: str,
            reynolds: float,
            alpha: float,
            mach: float = 0.0,
            solver: Optional[str] = None,
            ncrit: float = 9.0,
            use_fallback: bool = True,
            **kwargs) -> Dict:
    """
    Run airfoil analysis with automatic solver selection
    
    Parameters:
    -----------
    airfoil_file : str
        Path to airfoil coordinate file (.dat)
    reynolds : float
        Reynolds number
    alpha : float
        Angle of attack (degrees)
    mach : float
        Mach number (default: 0.0)
    solver : Optional[str]
        Force specific solver: 'neuralfoil', 'xfoil', 'su2_sa', 'su2_sst'
        None = auto-select (default)
    ncrit : float
        Critical amplification factor for transition (default: 9.0)
    use_fallback : bool
        Use fallback solver if primary fails (default: True)
        
    Returns:
    --------
    Dict with keys: CL, CD, CM, converged, solver, ...
    """
    
    # Check file exists
    airfoil_path = Path(airfoil_file)
    if not airfoil_path.exists():
        raise FileNotFoundError(f"Airfoil file not found: {airfoil_file}")
    
    # Parse solver preference
    solver_type = None
    if solver is not None:
        solver_map = {
            'neuralfoil': SolverType.NEURALFOIL,
            'xfoil': SolverType.XFOIL,
            'su2': SolverType.SU2_SA,
            'su2_sa': SolverType.SU2_SA,
            'su2_sst': SolverType.SU2_SST,
            'su2_transition': SolverType.SU2_TRANSITION,
        }
        solver_type = solver_map.get(solver.lower())
    
    # Select solver
    if solver_type is None:
        solver_type = select_solver(reynolds, mach)
    
    # Run analysis
    result = _run_solver(solver_type, airfoil_path, reynolds, alpha, mach, ncrit, **kwargs)
    
    # Fallback if needed
    if not result.get('converged', False) and use_fallback:
        result = _try_fallback(airfoil_path, reynolds, alpha, mach, ncrit, **kwargs)
    
    return result


def _run_solver(solver_type: SolverType, 
                airfoil_path: Path,
                reynolds: float,
                alpha: float,
                mach: float,
                ncrit: float,
                **kwargs) -> Dict:
    """Run specific solver"""
    
    if solver_type == SolverType.NEURALFOIL:
        if not NeuralFoilSolver.is_available():
            return {'converged': False, 'error': 'NeuralFoil not available'}
        
        solver = NeuralFoilSolver(model_size=kwargs.get('model_size', 'xlarge'))
        return solver.analyze(str(airfoil_path), reynolds, alpha, ncrit)
    
    elif solver_type == SolverType.XFOIL:
        if not XFoilSolver.is_available():
            return {'converged': False, 'error': 'XFoil not available'}
        
        solver = XFoilSolver()
        return solver.analyze(str(airfoil_path), reynolds, alpha, mach, ncrit)
    
    elif solver_type in [SolverType.SU2_SA, SolverType.SU2_SST, SolverType.SU2_TRANSITION]:
        if not SU2Solver.is_available():
            return {'converged': False, 'error': 'SU2 not available'}
        
        # SU2 requires mesh - return config info only
        turb_model = {
            SolverType.SU2_SA: 'SA',
            SolverType.SU2_SST: 'SST',
            SolverType.SU2_TRANSITION: 'TRANSITION',
        }.get(solver_type, 'SA')
        
        return {
            'converged': False,
            'error': 'SU2 requires mesh file. Use SU2Config to generate configuration.',
            'solver': 'su2',
            'suggested_config': {
                'reynolds': reynolds,
                'mach': mach,
                'alpha': alpha,
                'turbulence_model': turb_model,
            }
        }
    
    return {'converged': False, 'error': f'Unknown solver type: {solver_type}'}


def _try_fallback(airfoil_path: Path,
                  reynolds: float,
                  alpha: float,
                  mach: float,
                  ncrit: float,
                  **kwargs) -> Dict:
    """Try fallback solvers in order: NeuralFoil -> XFoil"""
    
    # Only use incompressible solvers for fallback
    if mach >= 0.5:
        return {'converged': False, 'error': 'No fallback available for compressible flow'}
    
    # Try NeuralFoil first (fastest, most reliable)
    if NeuralFoilSolver.is_available():
        solver = NeuralFoilSolver()
        result = solver.analyze(str(airfoil_path), reynolds, alpha, ncrit)
        if result.get('converged'):
            result['fallback'] = True
            return result
    
    # Try XFoil
    if XFoilSolver.is_available():
        solver = XFoilSolver()
        result = solver.analyze(str(airfoil_path), reynolds, alpha, mach, ncrit)
        if result.get('converged'):
            result['fallback'] = True
            return result
    
    return {'converged': False, 'error': 'All fallback solvers failed'}


def analyze_sweep(airfoil_file: str,
                  reynolds: float,
                  alpha_range: Union[Tuple[float, float, float], List[float], np.ndarray],
                  mach: float = 0.0,
                  solver: Optional[str] = None,
                  ncrit: float = 9.0,
                  **kwargs) -> List[Dict]:
    """
    Run AoA sweep analysis
    
    Parameters:
    -----------
    airfoil_file : str
        Path to airfoil coordinate file
    reynolds : float
        Reynolds number
    alpha_range : tuple or array
        If tuple: (min, max, step) in degrees
        If array: list of alpha values
    mach : float
        Mach number
    solver : Optional[str]
        Force specific solver
    ncrit : float
        Critical amplification factor
        
    Returns:
    --------
    List of result dictionaries
    """
    
    # Convert alpha_range to array
    if isinstance(alpha_range, tuple):
        alpha_min, alpha_max, alpha_step = alpha_range
        alphas = np.arange(alpha_min, alpha_max + alpha_step/2, alpha_step)
    else:
        alphas = np.array(alpha_range)
    
    # Select solver
    solver_type = None
    if solver is not None:
        solver_map = {
            'neuralfoil': SolverType.NEURALFOIL,
            'xfoil': SolverType.XFOIL,
        }
        solver_type = solver_map.get(solver.lower())
    
    if solver_type is None:
        solver_type = select_solver(reynolds, mach)
    
    # Prefer NeuralFoil for sweeps (vectorized)
    if solver_type == SolverType.NEURALFOIL or NeuralFoilSolver.is_available():
        try:
            nf_solver = NeuralFoilSolver(model_size=kwargs.get('model_size', 'xlarge'))
            return nf_solver.analyze_sweep(airfoil_file, reynolds, alphas, ncrit)
        except:
            pass
    
    # Fallback to sequential analysis
    results = []
    for alpha in alphas:
        result = analyze(airfoil_file, reynolds, alpha, mach, 
                        solver=solver, ncrit=ncrit, **kwargs)
        results.append(result)
    
    return results


def compare_solvers(airfoil_file: str,
                   reynolds: float,
                   alpha: float,
                   mach: float = 0.0) -> Dict[str, Dict]:
    """
    Run analysis with all available solvers and compare
    
    Returns:
    --------
    Dict mapping solver name to results
    """
    
    results = {}
    avail = get_available_solvers()
    
    if avail.get('neuralfoil'):
        try:
            solver = NeuralFoilSolver()
            results['neuralfoil'] = solver.analyze(airfoil_file, reynolds, alpha)
        except Exception as e:
            results['neuralfoil'] = {'error': str(e)}
    
    if avail.get('xfoil') and mach < 0.5:
        try:
            solver = XFoilSolver()
            results['xfoil'] = solver.analyze(airfoil_file, reynolds, alpha, mach)
        except Exception as e:
            results['xfoil'] = {'error': str(e)}
    
    return results


def print_comparison(comparison: Dict[str, Dict]):
    """Pretty print solver comparison"""
    
    print("\n" + "="*70)
    print("SOLVER COMPARISON")
    print("="*70)
    
    headers = ['Solver', 'CL', 'CD', 'CM', 'L/D', 'Status']
    row_fmt = "{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}"
    
    print(row_fmt.format(*headers))
    print("-"*70)
    
    for solver_name, result in comparison.items():
        if result.get('converged'):
            cl = result.get('CL', 0)
            cd = result.get('CD', 1)
            cm = result.get('CM', 0)
            ld = cl / cd if cd != 0 else 0
            status = "✓"
            
            print(row_fmt.format(
                solver_name,
                f"{cl:.6f}",
                f"{cd:.6f}",
                f"{cm:.6f}",
                f"{ld:.2f}",
                status
            ))
        else:
            error = result.get('error', 'Failed')[:10]
            print(row_fmt.format(solver_name, '-', '-', '-', '-', f"✗ {error}"))
    
    print("="*70)
