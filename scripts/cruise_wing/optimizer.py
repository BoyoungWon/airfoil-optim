"""
SLSQP Optimizer for Cruise Wing

Sequential Least Squares Programming 최적화
- Gradient-based optimization
- Constraint handling
- Scipy interface
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from scipy.optimize import minimize, OptimizeResult
from scipy.optimize import differential_evolution
import json
from pathlib import Path


@dataclass
class OptimizationConfig:
    """최적화 설정"""
    # Algorithm settings
    method: str = 'SLSQP'
    max_iterations: int = 50
    ftol: float = 1e-6
    eps: float = 1e-8
    
    # Objective
    objective_type: str = 'maximize'  # 'maximize' or 'minimize'
    objective_metric: str = 'L/D'
    
    # Constraints
    constraints: List[Dict] = field(default_factory=list)
    
    # Bounds
    bounds: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    success: bool
    optimal_params: np.ndarray
    optimal_value: float
    n_iterations: int
    n_evaluations: int
    message: str
    history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'optimal_params': self.optimal_params.tolist(),
            'optimal_value': self.optimal_value,
            'n_iterations': self.n_iterations,
            'n_evaluations': self.n_evaluations,
            'message': self.message
        }


class CruiseWingOptimizer:
    """
    SLSQP Optimizer for Cruise Wing Optimization
    
    Objective: Maximize L/D at cruise
    Variables: 3 (NACA m, p, t)
    Constraints: CL_min, CM limits, thickness limits
    """
    
    # Default bounds for NACA parameters
    DEFAULT_BOUNDS = [
        (0.0, 0.06),   # m: max camber (0-6%)
        (0.2, 0.5),    # p: camber position (20-50% chord)
        (0.09, 0.18)   # t: thickness (9-18%)
    ]
    
    def __init__(self, 
                 objective_func: Callable,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 constraints: Optional[List[Dict]] = None,
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer
        
        Parameters
        ----------
        objective_func : callable
            Function that takes params array and returns dict with metrics
        bounds : list, optional
            Parameter bounds [(min, max), ...]
        constraints : list, optional
            Constraint definitions
        config : OptimizationConfig, optional
            Optimization configuration
        """
        self.objective_func = objective_func
        self.bounds = bounds or self.DEFAULT_BOUNDS
        self.config = config or OptimizationConfig(bounds=self.bounds)
        
        # Set up constraints
        self.constraints_list = constraints or []
        self._scipy_constraints = self._build_constraints()
        
        # History tracking
        self.history: List[Dict] = []
        self.n_evaluations = 0
        self.best_value = -np.inf
        self.best_params = None
    
    def _build_constraints(self) -> List[Dict]:
        """Build scipy constraint dictionaries"""
        scipy_constraints = []
        
        for c in self.constraints_list:
            param = c['param']
            
            if 'min' in c:
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, p=param, v=c['min']: self._constraint_func(x, p, 'min', v)
                })
            
            if 'max' in c:
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, p=param, v=c['max']: self._constraint_func(x, p, 'max', v)
                })
        
        return scipy_constraints
    
    def _constraint_func(self, params: np.ndarray, 
                         metric: str, 
                         constraint_type: str, 
                         value: float) -> float:
        """
        Constraint function for scipy
        
        Returns >= 0 if constraint is satisfied
        """
        result = self.objective_func(params)
        
        if result is None:
            return -1e10  # Invalid design
        
        # Get metric value
        if metric == 'L/D' and 'L/D' not in result:
            if 'CL' in result and 'CD' in result and result['CD'] > 0:
                metric_value = result['CL'] / result['CD']
            else:
                return -1e10
        else:
            metric_value = result.get(metric, 0)
        
        if constraint_type == 'min':
            return metric_value - value  # >= 0 when metric >= min
        else:  # max
            return value - metric_value  # >= 0 when metric <= max
    
    def _objective_wrapper(self, params: np.ndarray) -> float:
        """
        Wrapper for objective function
        
        Handles maximization by negating
        """
        self.n_evaluations += 1
        
        result = self.objective_func(params)
        
        if result is None:
            return 1e10  # Large penalty for failed evaluations
        
        # Get objective metric
        metric = self.config.objective_metric
        
        if metric == 'L/D':
            if 'L/D' in result:
                value = result['L/D']
            elif 'CL' in result and 'CD' in result and result['CD'] > 0:
                value = result['CL'] / result['CD']
            else:
                return 1e10
        elif metric == 'CL^1.5/CD':
            if 'CL' in result and 'CD' in result and result['CD'] > 0 and result['CL'] > 0:
                value = (result['CL'] ** 1.5) / result['CD']
            else:
                return 1e10
        else:
            value = result.get(metric, 0)
        
        # Track history
        self.history.append({
            'iteration': self.n_evaluations,
            'params': params.copy().tolist(),
            'objective': value,
            'result': {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        })
        
        # Track best
        if value > self.best_value:
            self.best_value = value
            self.best_params = params.copy()
        
        # Negate for maximization (scipy minimizes)
        if self.config.objective_type == 'maximize':
            return -value
        return value
    
    def optimize(self, x0: Optional[np.ndarray] = None,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run optimization
        
        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial guess (default: center of bounds)
        verbose : bool
            Print progress
            
        Returns
        -------
        OptimizationResult
            Optimization results
        """
        # Initial guess
        if x0 is None:
            x0 = np.array([(b[0] + b[1]) / 2 for b in self.bounds])
        
        if verbose:
            print(f"\n🎯 Starting SLSQP Optimization")
            print(f"   Method: {self.config.method}")
            print(f"   Objective: {self.config.objective_type} {self.config.objective_metric}")
            print(f"   Variables: {len(self.bounds)}")
            print(f"   Constraints: {len(self._scipy_constraints)}")
            print(f"   Initial: {x0}")
        
        # Reset tracking
        self.history = []
        self.n_evaluations = 0
        self.best_value = -np.inf
        self.best_params = None
        
        # Callback for verbose output
        def callback(xk):
            if verbose and len(self.history) % 10 == 0:
                print(f"   Iteration {len(self.history)}: "
                      f"best {self.config.objective_metric}={self.best_value:.4f}")
        
        # Run optimization
        result = minimize(
            self._objective_wrapper,
            x0,
            method=self.config.method,
            bounds=self.bounds,
            constraints=self._scipy_constraints,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.ftol,
                'eps': self.config.eps,
                'disp': False
            },
            callback=callback
        )
        
        # Get final objective value (un-negated)
        final_value = -result.fun if self.config.objective_type == 'maximize' else result.fun
        
        if verbose:
            print(f"\n✓ Optimization Complete")
            print(f"   Success: {result.success}")
            print(f"   {self.config.objective_metric}: {final_value:.4f}")
            print(f"   Evaluations: {self.n_evaluations}")
            print(f"   Optimal params: {result.x}")
        
        return OptimizationResult(
            success=result.success,
            optimal_params=result.x,
            optimal_value=final_value,
            n_iterations=result.nit if hasattr(result, 'nit') else len(self.history),
            n_evaluations=self.n_evaluations,
            message=result.message if hasattr(result, 'message') else str(result),
            history=self.history
        )
    
    def optimize_multistart(self, n_starts: int = 5,
                            verbose: bool = True) -> OptimizationResult:
        """
        Multi-start optimization
        
        Run optimization from multiple starting points
        
        Parameters
        ----------
        n_starts : int
            Number of starting points
        verbose : bool
            Print progress
            
        Returns
        -------
        OptimizationResult
            Best result from all starts
        """
        from scipy.stats import qmc
        
        if verbose:
            print(f"\n🎯 Multi-start Optimization ({n_starts} starts)")
        
        # Generate starting points
        sampler = qmc.LatinHypercube(d=len(self.bounds), seed=42)
        samples = sampler.random(n=n_starts)
        
        lower = np.array([b[0] for b in self.bounds])
        upper = np.array([b[1] for b in self.bounds])
        starting_points = lower + samples * (upper - lower)
        
        best_result = None
        
        for i, x0 in enumerate(starting_points):
            if verbose:
                print(f"\n   Start {i+1}/{n_starts}")
            
            result = self.optimize(x0=x0, verbose=False)
            
            if best_result is None or result.optimal_value > best_result.optimal_value:
                best_result = result
            
            if verbose:
                print(f"   → {self.config.objective_metric}={result.optimal_value:.4f}")
        
        if verbose:
            print(f"\n✓ Best result: {self.config.objective_metric}={best_result.optimal_value:.4f}")
            print(f"   Optimal params: {best_result.optimal_params}")
        
        return best_result


class SurrogateOptimizer(CruiseWingOptimizer):
    """
    Optimizer using surrogate model
    
    Uses Kriging predictions for fast optimization
    """
    
    def __init__(self,
                 surrogate_model,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 constraints: Optional[List[Dict]] = None,
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize surrogate-based optimizer
        
        Parameters
        ----------
        surrogate_model : CruiseWingKriging
            Trained Kriging model
        bounds : list, optional
            Parameter bounds
        constraints : list, optional
            Constraints
        config : OptimizationConfig, optional
            Configuration
        """
        self.surrogate = surrogate_model
        
        # Objective function using surrogate
        def surrogate_objective(params):
            return self.surrogate.predict_single(params)
        
        super().__init__(
            objective_func=surrogate_objective,
            bounds=bounds,
            constraints=constraints,
            config=config
        )
    
    def optimize_with_uncertainty(self, x0: Optional[np.ndarray] = None,
                                   exploration_weight: float = 0.1,
                                   verbose: bool = True) -> OptimizationResult:
        """
        Optimization with uncertainty consideration (Expected Improvement-like)
        
        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial guess
        exploration_weight : float
            Weight for uncertainty term (higher = more exploration)
        verbose : bool
            Print progress
            
        Returns
        -------
        OptimizationResult
        """
        # Modify objective to include uncertainty
        original_metric = self.config.objective_metric
        
        def uncertainty_objective(params):
            pred, std = self.surrogate.predict_single(params.reshape(1, -1), return_std=True)
            
            # Expected Improvement-like: value + exploration * uncertainty
            if original_metric == 'L/D':
                value = pred.get('L/D', pred.get('CL', 0) / max(pred.get('CD', 1), 1e-10))
                uncertainty = std.get('L/D', std.get('CL', 0))
            else:
                value = pred.get(original_metric, 0)
                uncertainty = std.get(original_metric, 0)
            
            # For maximization, add uncertainty
            return {'value': value + exploration_weight * uncertainty}
        
        # Temporarily replace objective
        old_func = self.objective_func
        old_metric = self.config.objective_metric
        
        self.objective_func = uncertainty_objective
        self.config.objective_metric = 'value'
        
        result = self.optimize(x0=x0, verbose=verbose)
        
        # Restore
        self.objective_func = old_func
        self.config.objective_metric = old_metric
        
        return result


class DirectXFOILOptimizer(CruiseWingOptimizer):
    """
    Optimizer using direct XFOIL calls
    
    For small-scale optimization without surrogate
    """
    
    def __init__(self,
                 reynolds: float,
                 aoa: float,
                 mach: float = 0.0,
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 constraints: Optional[List[Dict]] = None,
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize direct XFOIL optimizer
        
        Parameters
        ----------
        reynolds : float
            Reynolds number
        aoa : float
            Angle of attack
        mach : float
            Mach number
        bounds : list, optional
            Parameter bounds
        constraints : list, optional
            Constraints
        config : OptimizationConfig, optional
            Configuration
        """
        self.reynolds = reynolds
        self.aoa = aoa
        self.mach = mach
        
        # Import analyzer
        from .analyzer import AirfoilAnalyzer
        from .database import NACADatabase
        
        self.analyzer = AirfoilAnalyzer()
        self.db = NACADatabase()
        
        def xfoil_objective(params):
            # Generate NACA coordinates
            m, p, t = params
            coords = self.db.generate_naca_coords(m, p, t)
            
            # Analyze with XFOIL
            result = self.analyzer.analyze_single(
                coords, self.reynolds, self.aoa, self.mach
            )
            
            return result
        
        super().__init__(
            objective_func=xfoil_objective,
            bounds=bounds,
            constraints=constraints,
            config=config
        )


def create_cruise_constraints(cl_min: float = 0.4,
                              cm_min: float = -0.1,
                              cm_max: float = 0.0,
                              ld_min: float = 50.0) -> List[Dict]:
    """
    Create default cruise wing constraints
    
    Parameters
    ----------
    cl_min : float
        Minimum lift coefficient
    cm_min : float
        Minimum (most negative) pitching moment
    cm_max : float
        Maximum pitching moment
    ld_min : float
        Minimum L/D ratio
        
    Returns
    -------
    list
        Constraint definitions
    """
    return [
        {'param': 'CL', 'min': cl_min},
        {'param': 'CM', 'min': cm_min, 'max': cm_max},
        {'param': 'L/D', 'min': ld_min}
    ]
