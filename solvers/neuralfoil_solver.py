"""
NeuralFoil Solver Interface

NeuralFoil: Neural network surrogate for airfoil aerodynamics

Features:
- Very fast predictions (~100x faster than XFoil)
- Stable convergence (no iteration failures)
- Confidence scores for predictions
- Boundary layer parameters

Trained on:
- Reynolds: 1e4 - 1e7
- Mach: Incompressible (< 0.5)
- Alpha: -20° to 30°

Limitations:
- Approximation (not exact solution)
- Limited to training data distribution
- No compressible flow support
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Union, List
import numpy as np


# NeuralFoil import utilities
def _import_neuralfoil_functions():
    """Import NeuralFoil functions with proper module resolution"""
    try:
        # Direct import from neuralfoil.main to avoid namespace issues
        from neuralfoil.main import (
            get_aero_from_kulfan_parameters,
            get_aero_from_airfoil,
            get_aero_from_coordinates,
            get_aero_from_dat_file,
        )
        return {
            'get_aero_from_kulfan_parameters': get_aero_from_kulfan_parameters,
            'get_aero_from_airfoil': get_aero_from_airfoil,
            'get_aero_from_coordinates': get_aero_from_coordinates,
            'get_aero_from_dat_file': get_aero_from_dat_file,
        }
    except ImportError:
        return None


class NeuralFoilSolver:
    """NeuralFoil solver interface"""
    
    def __init__(self, model_size: str = "xlarge"):
        """
        Initialize NeuralFoil solver
        
        Parameters:
        -----------
        model_size : str
            Model size: "xxsmall", "xsmall", "small", "medium", 
                       "large", "xlarge", "xxlarge", "xxxlarge"
        """
        self.model_size = model_size
        self._nf_funcs = _import_neuralfoil_functions()
    
    @staticmethod
    def is_available() -> bool:
        """Check if NeuralFoil is available"""
        try:
            from neuralfoil.main import get_aero_from_dat_file
            return True
        except ImportError:
            return False
    
    def analyze(self,
                airfoil_file: str,
                reynolds: float,
                alpha: float,
                ncrit: float = 9.0,
                xtr_upper: float = 1.0,
                xtr_lower: float = 1.0) -> Optional[Dict]:
        """
        Run NeuralFoil analysis
        
        Parameters:
        -----------
        airfoil_file : str
            Path to airfoil coordinate file (.dat)
        reynolds : float
            Reynolds number
        alpha : float
            Angle of attack (degrees)
        ncrit : float
            Critical amplification factor (default: 9.0)
        xtr_upper : float
            Forced transition on upper surface (0-1, default: 1.0 = natural)
        xtr_lower : float
            Forced transition on lower surface (0-1, default: 1.0 = natural)
            
        Returns:
        --------
        Dict with keys: CL, CD, CM, Top_Xtr, Bot_Xtr, analysis_confidence, converged
        """
        
        if self._nf_funcs is None:
            raise ImportError("NeuralFoil not available. Install: pip install neuralfoil")
        
        airfoil_path = Path(airfoil_file)
        if not airfoil_path.exists():
            raise FileNotFoundError(f"Airfoil file not found: {airfoil_file}")
        
        try:
            # Run NeuralFoil
            raw_result = self._nf_funcs['get_aero_from_dat_file'](
                str(airfoil_path),
                alpha=alpha,
                Re=reynolds,
                n_crit=ncrit,
                xtr_upper=xtr_upper,
                xtr_lower=xtr_lower,
                model_size=self.model_size
            )
            
            # Extract scalar values (NeuralFoil returns arrays)
            def to_scalar(val):
                if hasattr(val, '__len__') and len(val) == 1:
                    return float(val[0])
                return float(val)
            
            result = {
                'reynolds': reynolds,
                'aoa': alpha,
                'mach': 0.0,
                'CL': to_scalar(raw_result['CL']),
                'CD': to_scalar(raw_result['CD']),
                'CM': to_scalar(raw_result['CM']),
                'Top_Xtr': to_scalar(raw_result['Top_Xtr']),
                'Bot_Xtr': to_scalar(raw_result['Bot_Xtr']),
                'analysis_confidence': to_scalar(raw_result['analysis_confidence']),
                'converged': True,
                'solver': 'neuralfoil'
            }
            
            return result
            
        except Exception as e:
            return {
                'converged': False,
                'error': str(e),
                'solver': 'neuralfoil'
            }
    
    def analyze_sweep(self,
                      airfoil_file: str,
                      reynolds: float,
                      alpha_range: Union[List[float], np.ndarray],
                      ncrit: float = 9.0) -> List[Dict]:
        """
        Run AoA sweep analysis (vectorized for speed)
        
        Parameters:
        -----------
        alpha_range : array-like
            List of angles of attack (degrees)
            
        Returns:
        --------
        List of result dictionaries
        """
        
        if self._nf_funcs is None:
            raise ImportError("NeuralFoil not available")
        
        airfoil_path = Path(airfoil_file)
        alphas = np.array(alpha_range)
        
        try:
            # Vectorized analysis (very fast)
            raw_result = self._nf_funcs['get_aero_from_dat_file'](
                str(airfoil_path),
                alpha=alphas,
                Re=reynolds,
                n_crit=ncrit,
                model_size=self.model_size
            )
            
            results = []
            for i, alpha in enumerate(alphas):
                result = {
                    'reynolds': reynolds,
                    'aoa': alpha,
                    'mach': 0.0,
                    'CL': float(raw_result['CL'][i]),
                    'CD': float(raw_result['CD'][i]),
                    'CM': float(raw_result['CM'][i]),
                    'Top_Xtr': float(raw_result['Top_Xtr'][i]),
                    'Bot_Xtr': float(raw_result['Bot_Xtr'][i]),
                    'analysis_confidence': float(raw_result['analysis_confidence'][i]),
                    'converged': True,
                    'solver': 'neuralfoil'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            return [{'converged': False, 'error': str(e), 'solver': 'neuralfoil'}]
    
    def analyze_coordinates(self,
                           coordinates: np.ndarray,
                           reynolds: float,
                           alpha: float,
                           ncrit: float = 9.0) -> Optional[Dict]:
        """
        Analyze airfoil from coordinates directly
        
        Parameters:
        -----------
        coordinates : np.ndarray
            Shape (N, 2) array of (x, y) coordinates in Selig format
        """
        
        if self._nf_funcs is None:
            raise ImportError("NeuralFoil not available")
        
        try:
            raw_result = self._nf_funcs['get_aero_from_coordinates'](
                coordinates,
                alpha=alpha,
                Re=reynolds,
                n_crit=ncrit,
                model_size=self.model_size
            )
            
            def to_scalar(val):
                if hasattr(val, '__len__') and len(val) == 1:
                    return float(val[0])
                return float(val)
            
            return {
                'reynolds': reynolds,
                'aoa': alpha,
                'mach': 0.0,
                'CL': to_scalar(raw_result['CL']),
                'CD': to_scalar(raw_result['CD']),
                'CM': to_scalar(raw_result['CM']),
                'Top_Xtr': to_scalar(raw_result['Top_Xtr']),
                'Bot_Xtr': to_scalar(raw_result['Bot_Xtr']),
                'analysis_confidence': to_scalar(raw_result['analysis_confidence']),
                'converged': True,
                'solver': 'neuralfoil'
            }
            
        except Exception as e:
            return {'converged': False, 'error': str(e), 'solver': 'neuralfoil'}


# Convenience function
def analyze(airfoil_file: str, reynolds: float, alpha: float, 
            ncrit: float = 9.0, model_size: str = "xlarge", 
            **kwargs) -> Optional[Dict]:
    """Quick analysis using NeuralFoil"""
    solver = NeuralFoilSolver(model_size=model_size)
    return solver.analyze(airfoil_file, reynolds, alpha, ncrit, **kwargs)
