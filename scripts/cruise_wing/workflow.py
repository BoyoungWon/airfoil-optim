"""
Cruise Wing Optimization Workflow

전체 최적화 워크플로우 오케스트레이터:
Phase 1: Database screening (30분)
Phase 2: NACA optimization (2-4시간)
Phase 3: Validation (1일)

Application: 일반 항공기, 글라이더, UAV
Operating Condition (XFOIL Valid Range):
  - Re: 50k - 50M
  - Ma: 0 - 0.5 (with compressibility correction)
  - α: -5° - 15° (pre-stall region)
Design Objective: Maximize L/D at cruise
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import json
import yaml
import time
from datetime import datetime

from .database import NACADatabase
from .analyzer import AirfoilAnalyzer
from .kriging import CruiseWingKriging, LHSSampler, generate_training_data
from .optimizer import (
    CruiseWingOptimizer as SLSQPOptimizer,
    SurrogateOptimizer,
    DirectXFOILOptimizer,
    OptimizationConfig,
    OptimizationResult,
    create_cruise_constraints
)
from .visualizer import OptimizationVisualizer


@dataclass
class DesignPoint:
    """단일 설계점 정의"""
    reynolds: float
    aoa: float
    mach: float = 0.0
    weight: float = 1.0
    name: str = "design_point"


@dataclass
class CruiseWingConfig:
    """Cruise Wing 최적화 설정"""
    # Design point
    design_points: List[DesignPoint] = field(default_factory=list)
    
    # Parameter bounds
    m_bounds: Tuple[float, float] = (0.0, 0.06)    # Max camber
    p_bounds: Tuple[float, float] = (0.2, 0.5)     # Camber position
    t_bounds: Tuple[float, float] = (0.09, 0.18)   # Thickness
    
    # Constraints
    cl_min: float = 0.4
    cm_min: float = -0.1
    cm_max: float = 0.0
    ld_min: float = 50.0
    t_min: float = 0.10  # Minimum thickness (structural)
    
    # Surrogate settings
    use_surrogate: bool = True
    n_training_samples: int = 80
    kriging_kernel: str = 'matern'
    
    # Optimization settings
    max_iterations: int = 50
    n_multistart: int = 5
    convergence_tol: float = 1e-6
    
    # Output
    output_dir: str = "output/optimization/cruise_wing"
    save_history: bool = True
    create_plots: bool = True
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'CruiseWingConfig':
        """YAML 파일에서 설정 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Parse design points
        design_points = []
        for dp in data.get('design_points', []):
            design_points.append(DesignPoint(
                reynolds=float(dp['reynolds']),
                aoa=float(dp['aoa']),
                mach=float(dp.get('mach', 0.0)),
                weight=float(dp.get('weight', 1.0)),
                name=dp.get('name', 'design_point')
            ))
        
        # Parse bounds
        param_config = data.get('parametrization', {}).get('parameters', {})
        bounds = param_config.get('bounds', {})
        
        m_bounds = tuple(b / 100 for b in bounds.get('max_camber', [0, 6]))
        p_bounds = tuple(b / 10 for b in bounds.get('camber_pos', [2, 5]))
        t_bounds = tuple(b / 100 for b in bounds.get('thickness', [9, 18]))
        
        # Parse constraints
        constraints = data.get('constraints', {})
        aero_constraints = constraints.get('aerodynamic', [])
        
        cl_min = 0.4
        cm_min, cm_max = -0.1, 0.0
        ld_min = 50.0
        
        for c in aero_constraints:
            if c['param'] == 'CL':
                cl_min = c.get('min', cl_min)
            elif c['param'] == 'CM':
                cm_min = c.get('min', cm_min)
                cm_max = c.get('max', cm_max)
            elif c['param'] == 'CL/CD':
                ld_min = c.get('min', ld_min)
        
        # Parse surrogate settings
        surrogate = data.get('surrogate', {})
        
        # Parse optimization settings
        optimization = data.get('optimization', {})
        
        return cls(
            design_points=design_points,
            m_bounds=m_bounds,
            p_bounds=p_bounds,
            t_bounds=t_bounds,
            cl_min=cl_min,
            cm_min=cm_min,
            cm_max=cm_max,
            ld_min=ld_min,
            use_surrogate=surrogate.get('method', 'kriging') != 'none',
            n_training_samples=surrogate.get('training_samples', 80),
            kriging_kernel=surrogate.get('kernel', 'matern'),
            max_iterations=optimization.get('max_iterations', 50),
            convergence_tol=optimization.get('convergence_tol', 1e-6),
            output_dir=data.get('output', {}).get('directory', 'output/optimization/cruise_wing'),
            save_history=data.get('output', {}).get('save_history', True),
            create_plots=data.get('output', {}).get('plot_convergence', True)
        )


@dataclass
class WorkflowResult:
    """워크플로우 결과"""
    success: bool
    initial_airfoil: Dict
    optimal_airfoil: Dict
    optimization_result: OptimizationResult
    validation_result: Optional[Dict]
    total_time: float
    phases: Dict
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'initial_airfoil': self.initial_airfoil,
            'optimal_airfoil': self.optimal_airfoil,
            'optimization_result': self.optimization_result.to_dict(),
            'validation_result': self.validation_result,
            'total_time_seconds': self.total_time,
            'phases': self.phases
        }


class CruiseWingOptimizer:
    """
    Cruise Wing 최적화 워크플로우 오케스트레이터
    
    Complete workflow:
    1. Database Screening - Find best initial NACA airfoil
    2. Surrogate Training - Build Kriging model (optional)
    3. Optimization - SLSQP with surrogate or direct XFOIL
    4. Validation - Verify with XFOIL polar analysis
    """
    
    def __init__(self, config: Optional[CruiseWingConfig] = None,
                 scenario_file: Optional[str] = None):
        """
        Initialize optimizer
        
        Parameters
        ----------
        config : CruiseWingConfig, optional
            Configuration object
        scenario_file : str, optional
            Path to YAML scenario file
        """
        if scenario_file:
            self.config = CruiseWingConfig.from_yaml(scenario_file)
        elif config:
            self.config = config
        else:
            # Default single-point cruise optimization
            self.config = CruiseWingConfig(
                design_points=[DesignPoint(reynolds=3e6, aoa=3.0, mach=0.2)]
            )
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.database = NACADatabase(cache_dir=str(self.output_dir / "database"))
        self.analyzer = AirfoilAnalyzer()
        self.visualizer = OptimizationVisualizer(output_dir=str(self.output_dir / "figures"))
        
        # Bounds
        self.bounds = [
            self.config.m_bounds,
            self.config.p_bounds,
            self.config.t_bounds
        ]
        
        # Results storage
        self.initial_airfoil = None
        self.surrogate = None
        self.optimization_result = None
        self.optimal_airfoil = None
    
    def _evaluate_design(self, params: np.ndarray) -> Optional[Dict]:
        """
        Evaluate design at all design points
        
        Parameters
        ----------
        params : np.ndarray
            [m, p, t] NACA parameters
            
        Returns
        -------
        dict or None
            Weighted average results
        """
        m, p, t = params
        
        # Validate parameters
        if p <= 0 or p >= 1:
            return None
        if t <= 0:
            return None
        
        # Generate coordinates
        coords = self.database.generate_naca_coords(m, p, t)
        
        # Multi-point analysis
        design_points = [
            {
                'reynolds': dp.reynolds,
                'aoa': dp.aoa,
                'mach': dp.mach,
                'weight': dp.weight
            }
            for dp in self.config.design_points
        ]
        
        return self.analyzer.analyze_multi_point(coords, design_points)
    
    def phase1_database_screening(self, verbose: bool = True) -> Dict:
        """
        Phase 1: Database Screening
        
        Scan NACA database to find best initial design
        
        Returns
        -------
        dict
            Best initial airfoil info
        """
        if verbose:
            print("\n" + "="*60)
            print("PHASE 1: DATABASE SCREENING")
            print("="*60)
        
        start_time = time.time()
        
        # Use primary design point for screening
        primary_dp = self.config.design_points[0]
        
        # Scan database
        results = self.database.scan_database(
            reynolds=primary_dp.reynolds,
            aoa=primary_dp.aoa,
            mach=primary_dp.mach,
            target_cl=self.config.cl_min,
            verbose=verbose
        )
        
        # Filter by minimum thickness
        results = [r for r in results if r.get('t', 0) >= self.config.t_min]
        
        if not results:
            # Fallback to NACA 2412
            if verbose:
                print("   No suitable airfoil found, using NACA 2412")
            initial = {
                'naca': '2412',
                'm': 0.02,
                'p': 0.4,
                't': 0.12,
                'L/D': None
            }
        else:
            initial = results[0]
        
        # Generate coordinates for initial
        initial['coords'] = self.database.generate_naca_coords(
            initial['m'], initial['p'], initial['t']
        )
        initial['name'] = f"NACA {initial['naca']}"
        
        # Calculate L/D if not available
        if initial.get('L/D') is None:
            result = self._evaluate_design(np.array([initial['m'], initial['p'], initial['t']]))
            if result and 'L/D' in result:
                initial['L/D'] = result['L/D']
                initial['CL'] = result.get('CL')
                initial['CD'] = result.get('CD')
                initial['CM'] = result.get('CM')
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Phase 1 Complete ({elapsed:.1f}s)")
            print(f"   Selected: NACA {initial['naca']}")
            print(f"   L/D: {initial.get('L/D', 'N/A')}")
        
        self.initial_airfoil = initial
        
        return {
            'initial': initial,
            'elapsed_time': elapsed,
            'n_scanned': len(self.database.COMMON_NACA)
        }
    
    def phase2_surrogate_training(self, verbose: bool = True) -> Optional[Dict]:
        """
        Phase 2: Surrogate Model Training
        
        Build Kriging surrogate using LHS samples
        
        Returns
        -------
        dict or None
            Training statistics
        """
        if not self.config.use_surrogate:
            if verbose:
                print("\n⏭ Skipping surrogate training (direct optimization)")
            return None
        
        if verbose:
            print("\n" + "="*60)
            print("PHASE 2: SURROGATE TRAINING")
            print("="*60)
        
        start_time = time.time()
        
        # Generate training data
        X_train, y_train = generate_training_data(
            bounds=self.bounds,
            n_samples=self.config.n_training_samples,
            evaluate_func=self._evaluate_design,
            verbose=verbose
        )
        
        if len(X_train) < 20:
            if verbose:
                print("   ⚠ Insufficient training data, falling back to direct optimization")
            self.config.use_surrogate = False
            return None
        
        # Train Kriging model
        self.surrogate = CruiseWingKriging(
            kernel=self.config.kriging_kernel,
            normalize=True
        )
        
        stats = self.surrogate.train(X_train, y_train, verbose=verbose)
        
        # Cross-validation
        if verbose:
            print("\n📊 Cross-validation...")
        
        cv_scores = self.surrogate.cross_validate(n_folds=5)
        
        if verbose:
            for metric, scores in cv_scores.items():
                print(f"   {metric}: CV R² = {scores['mean']:.4f} ± {scores['std']:.4f}")
        
        # Save surrogate model
        model_path = self.output_dir / "surrogate_model.pkl"
        self.surrogate.save(model_path)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Phase 2 Complete ({elapsed:.1f}s)")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Model saved: {model_path}")
        
        # Create validation plot
        if self.config.create_plots:
            # Predict on training data
            y_pred = self.surrogate.predict(X_train)
            
            if 'L/D' in y_train[0]:
                y_true = np.array([y['L/D'] for y in y_train])
                y_pred_ld = y_pred['L/D']
                self.visualizer.plot_surrogate_validation(y_true, y_pred_ld, 'L/D')
            
            self.visualizer.plot_design_space(X_train, y_train)
        
        return {
            'n_samples': len(X_train),
            'training_stats': stats,
            'cv_scores': cv_scores,
            'elapsed_time': elapsed
        }
    
    def phase3_optimization(self, verbose: bool = True) -> OptimizationResult:
        """
        Phase 3: Optimization
        
        SLSQP optimization using surrogate or direct XFOIL
        
        Returns
        -------
        OptimizationResult
            Optimization results
        """
        if verbose:
            print("\n" + "="*60)
            print("PHASE 3: OPTIMIZATION")
            print("="*60)
        
        start_time = time.time()
        
        # Setup constraints
        constraints = create_cruise_constraints(
            cl_min=self.config.cl_min,
            cm_min=self.config.cm_min,
            cm_max=self.config.cm_max,
            ld_min=self.config.ld_min
        )
        
        # Configuration
        opt_config = OptimizationConfig(
            method='SLSQP',
            max_iterations=self.config.max_iterations,
            ftol=self.config.convergence_tol,
            objective_type='maximize',
            objective_metric='L/D',
            bounds=self.bounds,
            constraints=constraints
        )
        
        # Initial guess from Phase 1
        if self.initial_airfoil:
            x0 = np.array([
                self.initial_airfoil['m'],
                self.initial_airfoil['p'],
                self.initial_airfoil['t']
            ])
        else:
            x0 = np.array([0.02, 0.4, 0.12])  # NACA 2412
        
        # Create optimizer
        if self.config.use_surrogate and self.surrogate is not None:
            if verbose:
                print("   Using surrogate-based optimization")
            
            optimizer = SurrogateOptimizer(
                surrogate_model=self.surrogate,
                bounds=self.bounds,
                constraints=constraints,
                config=opt_config
            )
        else:
            if verbose:
                print("   Using direct XFOIL optimization")
            
            primary_dp = self.config.design_points[0]
            optimizer = DirectXFOILOptimizer(
                reynolds=primary_dp.reynolds,
                aoa=primary_dp.aoa,
                mach=primary_dp.mach,
                bounds=self.bounds,
                constraints=constraints,
                config=opt_config
            )
        
        # Run multi-start optimization
        result = optimizer.optimize_multistart(
            n_starts=self.config.n_multistart,
            verbose=verbose
        )
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Phase 3 Complete ({elapsed:.1f}s)")
            print(f"   Optimal L/D: {result.optimal_value:.2f}")
            print(f"   Parameters: m={result.optimal_params[0]:.4f}, "
                  f"p={result.optimal_params[1]:.4f}, "
                  f"t={result.optimal_params[2]:.4f}")
        
        self.optimization_result = result
        
        # Generate optimal airfoil info
        m, p, t = result.optimal_params
        naca_code = self.database.encode_naca_code(m, p, t)
        
        self.optimal_airfoil = {
            'naca': naca_code,
            'm': m,
            'p': p,
            't': t,
            'coords': self.database.generate_naca_coords(m, p, t),
            'name': f"NACA {naca_code}",
            'L/D': result.optimal_value
        }
        
        # Create convergence plot
        if self.config.create_plots and result.history:
            self.visualizer.plot_convergence(result.history)
        
        return result
    
    def phase4_validation(self, verbose: bool = True) -> Dict:
        """
        Phase 4: XFOIL Validation
        
        Validate optimal design with full XFOIL polar analysis
        
        Returns
        -------
        dict
            Validation results
        """
        if verbose:
            print("\n" + "="*60)
            print("PHASE 4: VALIDATION")
            print("="*60)
        
        start_time = time.time()
        
        if self.optimal_airfoil is None:
            raise RuntimeError("No optimal airfoil to validate")
        
        coords = self.optimal_airfoil['coords']
        primary_dp = self.config.design_points[0]
        
        # Design point validation
        if verbose:
            print("\n📊 Design Point Validation")
        
        design_result = self.analyzer.analyze_single(
            coords,
            reynolds=primary_dp.reynolds,
            aoa=primary_dp.aoa,
            mach=primary_dp.mach
        )
        
        if design_result:
            self.optimal_airfoil.update(design_result)
            if verbose:
                print(f"   CL: {design_result.get('CL', 'N/A'):.4f}")
                print(f"   CD: {design_result.get('CD', 'N/A'):.5f}")
                print(f"   CM: {design_result.get('CM', 'N/A'):.4f}")
                print(f"   L/D: {design_result.get('L/D', 'N/A'):.2f}")
        
        # Polar analysis
        if verbose:
            print("\n📊 Polar Analysis")
        
        polar = self.analyzer.analyze_polar(
            coords,
            reynolds=primary_dp.reynolds,
            aoa_range=(-2, 12),
            aoa_step=0.5,
            mach=primary_dp.mach
        )
        
        if polar:
            self.optimal_airfoil['polar'] = polar
            
            # Find max L/D
            idx_max = np.argmax(polar['L/D'])
            if verbose:
                print(f"   Max L/D: {polar['L/D'][idx_max]:.2f} at α={polar['alpha'][idx_max]:.1f}°")
                print(f"   CL_max: {max(polar['CL']):.3f}")
        
        # Also get initial airfoil polar for comparison
        if self.initial_airfoil:
            if verbose:
                print("\n📊 Initial Airfoil Polar (for comparison)")
            
            initial_polar = self.analyzer.analyze_polar(
                self.initial_airfoil['coords'],
                reynolds=primary_dp.reynolds,
                aoa_range=(-2, 12),
                aoa_step=0.5,
                mach=primary_dp.mach
            )
            
            if initial_polar:
                self.initial_airfoil['polar'] = initial_polar
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Phase 4 Complete ({elapsed:.1f}s)")
        
        # Create comparison plots
        if self.config.create_plots:
            # Airfoil comparison
            self.visualizer.plot_airfoil_comparison({
                f"Initial: {self.initial_airfoil['name']}": self.initial_airfoil['coords'],
                f"Optimal: {self.optimal_airfoil['name']}": self.optimal_airfoil['coords']
            })
            
            # Polar comparison
            if 'polar' in self.initial_airfoil and 'polar' in self.optimal_airfoil:
                self.visualizer.plot_polar_comparison({
                    self.initial_airfoil['name']: self.initial_airfoil['polar'],
                    self.optimal_airfoil['name']: self.optimal_airfoil['polar']
                })
        
        return {
            'design_point': design_result,
            'polar': polar,
            'elapsed_time': elapsed
        }
    
    def run(self, verbose: bool = True) -> WorkflowResult:
        """
        Execute complete optimization workflow
        
        Parameters
        ----------
        verbose : bool
            Print progress
            
        Returns
        -------
        WorkflowResult
            Complete results
        """
        total_start = time.time()
        
        if verbose:
            print("\n" + "="*60)
            print("CRUISE WING OPTIMIZATION")
            print("="*60)
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Output: {self.output_dir}")
            
            print("\n📋 Configuration:")
            for dp in self.config.design_points:
                print(f"   Design Point: Re={dp.reynolds:.2e}, α={dp.aoa}°, Ma={dp.mach}")
            print(f"   Surrogate: {'Kriging' if self.config.use_surrogate else 'None (direct)'}")
            print(f"   Training samples: {self.config.n_training_samples}")
        
        phases = {}
        
        # Phase 1: Database Screening
        phases['phase1'] = self.phase1_database_screening(verbose=verbose)
        
        # Phase 2: Surrogate Training
        phases['phase2'] = self.phase2_surrogate_training(verbose=verbose)
        
        # Phase 3: Optimization
        phases['phase3'] = {'result': self.phase3_optimization(verbose=verbose).to_dict()}
        
        # Phase 4: Validation
        phases['phase4'] = self.phase4_validation(verbose=verbose)
        
        total_elapsed = time.time() - total_start
        
        # Create summary report
        if self.config.create_plots:
            self.visualizer.create_summary_report(
                self.optimization_result,
                self.initial_airfoil,
                self.optimal_airfoil
            )
        
        # Save results
        if self.config.save_history:
            self._save_results(phases, total_elapsed)
        
        if verbose:
            print("\n" + "="*60)
            print("OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
            print(f"\n📊 Results:")
            
            initial_ld = self.initial_airfoil.get('L/D')
            optimal_ld = self.optimal_airfoil.get('L/D')
            
            initial_str = f"{initial_ld:.2f}" if initial_ld is not None else "N/A"
            optimal_str = f"{optimal_ld:.2f}" if optimal_ld is not None else "N/A"
            
            print(f"   Initial: {self.initial_airfoil['name']}, L/D={initial_str}")
            print(f"   Optimal: {self.optimal_airfoil['name']}, L/D={optimal_str}")
            
            if initial_ld is not None and optimal_ld is not None:
                improvement = (optimal_ld / initial_ld - 1) * 100
                print(f"   Improvement: {improvement:+.1f}%")
            
            print(f"\n📁 Output files saved to: {self.output_dir}")
        
        # Determine overall success
        overall_success = (
            self.optimization_result is not None and 
            self.optimal_airfoil is not None and 
            self.optimal_airfoil.get('L/D') is not None and
            self.optimal_airfoil.get('L/D') > 0
        )
        
        return WorkflowResult(
            success=overall_success,
            initial_airfoil=self.initial_airfoil,
            optimal_airfoil=self.optimal_airfoil,
            optimization_result=self.optimization_result,
            validation_result=phases['phase4'],
            total_time=total_elapsed,
            phases=phases
        )
    
    def _save_results(self, phases: Dict, total_time: float):
        """Save results to files"""
        # Save optimal airfoil coordinates
        coord_file = self.output_dir / "optimal_airfoil.dat"
        with open(coord_file, 'w') as f:
            f.write(f"Optimal {self.optimal_airfoil['name']}\n")
            for x, y in self.optimal_airfoil['coords']:
                f.write(f"{x:12.8f}  {y:12.8f}\n")
        
        # Save optimization history
        history_file = self.output_dir / "optimization_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.optimization_result.history, f, indent=2)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'initial_airfoil': {
                'name': self.initial_airfoil['name'],
                'naca': self.initial_airfoil['naca'],
                'm': self.initial_airfoil['m'],
                'p': self.initial_airfoil['p'],
                't': self.initial_airfoil['t'],
                'L/D': self.initial_airfoil.get('L/D')
            },
            'optimal_airfoil': {
                'name': self.optimal_airfoil['name'],
                'naca': self.optimal_airfoil['naca'],
                'm': self.optimal_airfoil['m'],
                'p': self.optimal_airfoil['p'],
                't': self.optimal_airfoil['t'],
                'L/D': self.optimal_airfoil.get('L/D'),
                'CL': self.optimal_airfoil.get('CL'),
                'CD': self.optimal_airfoil.get('CD'),
                'CM': self.optimal_airfoil.get('CM')
            },
            'optimization': {
                'success': self.optimization_result.success,
                'n_evaluations': self.optimization_result.n_evaluations,
                'message': self.optimization_result.message
            },
            'config': {
                'design_points': [
                    {'reynolds': dp.reynolds, 'aoa': dp.aoa, 'mach': dp.mach}
                    for dp in self.config.design_points
                ],
                'use_surrogate': self.config.use_surrogate,
                'n_training_samples': self.config.n_training_samples
            }
        }
        
        summary_file = self.output_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved:")
        print(f"   {coord_file}")
        print(f"   {history_file}")
        print(f"   {summary_file}")


# Convenience function for quick optimization
def optimize_cruise_wing(reynolds: float = 3e6,
                         aoa: float = 3.0,
                         mach: float = 0.2,
                         use_surrogate: bool = True,
                         n_samples: int = 80,
                         output_dir: str = "output/optimization/cruise_wing",
                         verbose: bool = True) -> WorkflowResult:
    """
    Quick cruise wing optimization
    
    Parameters
    ----------
    reynolds : float
        Reynolds number
    aoa : float
        Design angle of attack (degrees)
    mach : float
        Mach number
    use_surrogate : bool
        Use Kriging surrogate model
    n_samples : int
        Number of training samples (if using surrogate)
    output_dir : str
        Output directory
    verbose : bool
        Print progress
        
    Returns
    -------
    WorkflowResult
        Optimization results
    """
    config = CruiseWingConfig(
        design_points=[DesignPoint(reynolds=reynolds, aoa=aoa, mach=mach)],
        use_surrogate=use_surrogate,
        n_training_samples=n_samples,
        output_dir=output_dir
    )
    
    optimizer = CruiseWingOptimizer(config=config)
    return optimizer.run(verbose=verbose)
