"""
SU2 Solver Interface

SU2: Open-source RANS CFD solver for high-fidelity airfoil analysis

Features:
- Compressible/transonic flow
- Multiple turbulence models (SA, SST, transition)
- High Reynolds number capability
- Optimization support

Models:
- SA (Spalart-Allmaras): Attached flows, efficient
- SST (k-omega SST): Separated flows, transonic
- Gamma-Re-theta: Laminar-turbulent transition

Limitations:
- Requires mesh generation
- Slower than panel methods
- Complex setup
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np


class SU2Config:
    """SU2 configuration file generator for airfoil analysis"""
    
    def __init__(self, case_name: str = "airfoil"):
        self.case_name = case_name
        self.config = {}
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration for 2D airfoil"""
        self.config = {
            # Problem definition
            'SOLVER': 'RANS',
            'KIND_TURB_MODEL': 'SA',
            'MATH_PROBLEM': 'DIRECT',
            'RESTART_SOL': 'NO',
            
            # Freestream (defaults)
            'MACH_NUMBER': 0.3,
            'AOA': 0.0,
            'FREESTREAM_TEMPERATURE': 288.15,
            'REYNOLDS_NUMBER': 1e6,
            'REYNOLDS_LENGTH': 1.0,
            
            # Reference values
            'REF_LENGTH': 1.0,
            'REF_AREA': 1.0,
            
            # Numerical method
            'NUM_METHOD_GRAD': 'WEIGHTED_LEAST_SQUARES',
            'CFL_NUMBER': 5.0,
            'CFL_ADAPT': 'YES',
            'CFL_ADAPT_PARAM': (0.5, 1.5, 5.0, 100.0),
            'ITER': 5000,
            
            # Linear solver
            'LINEAR_SOLVER': 'FGMRES',
            'LINEAR_SOLVER_PREC': 'ILU',
            'LINEAR_SOLVER_ERROR': 1e-6,
            'LINEAR_SOLVER_ITER': 10,
            
            # Multigrid
            'MGLEVEL': 3,
            'MGCYCLE': 'V_CYCLE',
            
            # Convergence
            'CONV_RESIDUAL_MINVAL': -12,
            'CONV_STARTITER': 10,
            'CONV_CAUCHY_ELEMS': 100,
            'CONV_CAUCHY_EPS': 1e-6,
            
            # Output
            'OUTPUT_WRT_FREQ': 100,
            'SCREEN_WRT_FREQ_INNER': 10,
        }
    
    def set_flow_conditions(self, 
                           reynolds: float,
                           mach: float,
                           aoa: float,
                           temperature: float = 288.15):
        """Set flow conditions"""
        self.config['REYNOLDS_NUMBER'] = reynolds
        self.config['MACH_NUMBER'] = mach
        self.config['AOA'] = aoa
        self.config['FREESTREAM_TEMPERATURE'] = temperature
        
        # Adjust CFL for transonic
        if mach >= 0.7:
            self.config['CFL_NUMBER'] = 1.0
            self.config['ITER'] = 10000
        
        return self
    
    def set_turbulence_model(self, model: str = 'SA'):
        """
        Set turbulence model
        
        Parameters:
        -----------
        model : str
            'SA', 'SST', 'SA_NEG', or 'TRANSITION' (LM)
        """
        model_upper = model.upper()
        
        if model_upper in ['SA', 'SA_NEG']:
            self.config['KIND_TURB_MODEL'] = model_upper
        elif model_upper == 'SST':
            self.config['KIND_TURB_MODEL'] = 'SST'
        elif model_upper in ['TRANSITION', 'LM', 'GAMMA_RETHETA']:
            self.config['KIND_TURB_MODEL'] = 'SA'
            self.config['KIND_TRANS_MODEL'] = 'LM'
        
        return self
    
    def set_mesh(self, mesh_file: str):
        """Set mesh file"""
        self.config['MESH_FILENAME'] = mesh_file
        self.config['MESH_FORMAT'] = 'SU2'
        return self
    
    def set_boundary_conditions(self,
                               airfoil_marker: str = 'airfoil',
                               farfield_marker: str = 'farfield'):
        """Set boundary conditions for airfoil"""
        self.config['MARKER_HEATFLUX'] = f'( {airfoil_marker}, 0.0 )'
        self.config['MARKER_FAR'] = f'( {farfield_marker} )'
        self.config['MARKER_PLOTTING'] = f'( {airfoil_marker} )'
        self.config['MARKER_MONITORING'] = f'( {airfoil_marker} )'
        return self
    
    def set_output(self, output_dir: str):
        """Set output configuration"""
        self.config['CONV_FILENAME'] = 'history'
        self.config['RESTART_FILENAME'] = 'restart'
        self.config['VOLUME_FILENAME'] = 'volume'
        self.config['SURFACE_FILENAME'] = 'surface'
        self.config['OUTPUT_FILES'] = ['RESTART', 'PARAVIEW', 'SURFACE_CSV']
        return self
    
    def write(self, output_file: str) -> Path:
        """Write configuration file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"% SU2 Configuration File\n")
            f.write(f"% Case: {self.case_name}\n")
            f.write(f"% Auto-generated for airfoil analysis\n\n")
            
            # Group configs
            sections = {
                'Problem Definition': ['SOLVER', 'KIND_TURB_MODEL', 'KIND_TRANS_MODEL', 
                                      'MATH_PROBLEM', 'RESTART_SOL'],
                'Flow Conditions': ['MACH_NUMBER', 'AOA', 'FREESTREAM_TEMPERATURE',
                                   'REYNOLDS_NUMBER', 'REYNOLDS_LENGTH'],
                'Reference Values': ['REF_LENGTH', 'REF_AREA'],
                'Boundary Conditions': ['MARKER_HEATFLUX', 'MARKER_FAR', 
                                       'MARKER_PLOTTING', 'MARKER_MONITORING'],
                'Numerical Method': ['NUM_METHOD_GRAD', 'CFL_NUMBER', 'CFL_ADAPT',
                                    'CFL_ADAPT_PARAM', 'ITER'],
                'Linear Solver': ['LINEAR_SOLVER', 'LINEAR_SOLVER_PREC',
                                 'LINEAR_SOLVER_ERROR', 'LINEAR_SOLVER_ITER'],
                'Multigrid': ['MGLEVEL', 'MGCYCLE'],
                'Convergence': ['CONV_RESIDUAL_MINVAL', 'CONV_STARTITER',
                               'CONV_CAUCHY_ELEMS', 'CONV_CAUCHY_EPS'],
                'Input/Output': ['MESH_FILENAME', 'MESH_FORMAT', 'CONV_FILENAME',
                                'RESTART_FILENAME', 'VOLUME_FILENAME', 'SURFACE_FILENAME',
                                'OUTPUT_FILES', 'OUTPUT_WRT_FREQ', 'SCREEN_WRT_FREQ_INNER'],
            }
            
            for section, keys in sections.items():
                f.write(f"% {section}\n")
                for key in keys:
                    if key in self.config:
                        val = self.config[key]
                        if isinstance(val, tuple):
                            val = f"( {', '.join(map(str, val))} )"
                        elif isinstance(val, list):
                            val = f"( {', '.join(map(str, val))} )"
                        f.write(f"{key}= {val}\n")
                f.write("\n")
        
        return output_path


class SU2Solver:
    """SU2 solver interface"""
    
    def __init__(self, su2_path: str = "SU2_CFD"):
        self.su2_path = su2_path
    
    @staticmethod
    def is_available() -> bool:
        """Check if SU2 is installed"""
        try:
            result = subprocess.run(
                ['SU2_CFD', '-h'],
                capture_output=True,
                timeout=10
            )
            return True
        except:
            return False
    
    def analyze(self,
                config_file: str,
                working_dir: str,
                timeout: int = 3600) -> Tuple[bool, Dict]:
        """
        Run SU2 analysis
        
        Parameters:
        -----------
        config_file : str
            Path to SU2 configuration file
        working_dir : str
            Working directory for output
        timeout : int
            Timeout in seconds (default: 3600 = 1 hour)
            
        Returns:
        --------
        Tuple[bool, Dict]: (success, results)
        """
        
        config_path = Path(config_file)
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\nRunning SU2_CFD...")
            print(f"  Config: {config_path}")
            print(f"  Working dir: {work_path}")
            
            process = subprocess.run(
                [self.su2_path, str(config_path.absolute())],
                cwd=str(work_path),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if process.returncode == 0:
                # Parse results
                results = self._parse_history(work_path / "history.csv")
                results['converged'] = True
                results['solver'] = 'su2'
                return True, results
            else:
                return False, {
                    'converged': False,
                    'error': f"Return code: {process.returncode}",
                    'stderr': process.stderr,
                    'solver': 'su2'
                }
                
        except subprocess.TimeoutExpired:
            return False, {'converged': False, 'error': 'Timeout', 'solver': 'su2'}
        except FileNotFoundError:
            return False, {'converged': False, 'error': 'SU2_CFD not found', 'solver': 'su2'}
        except Exception as e:
            return False, {'converged': False, 'error': str(e), 'solver': 'su2'}
    
    def _parse_history(self, history_file: Path) -> Dict:
        """Parse SU2 history file for final results"""
        
        results = {
            'CL': None,
            'CD': None,
            'CM': None,
        }
        
        if not history_file.exists():
            return results
        
        try:
            import pandas as pd
            df = pd.read_csv(history_file)
            
            # Get last converged values
            if 'CL' in df.columns:
                results['CL'] = float(df['CL'].iloc[-1])
            if 'CD' in df.columns:
                results['CD'] = float(df['CD'].iloc[-1])
            if 'CMz' in df.columns:
                results['CM'] = float(df['CMz'].iloc[-1])
            elif 'CM' in df.columns:
                results['CM'] = float(df['CM'].iloc[-1])
                
        except Exception as e:
            print(f"Warning: Could not parse history file: {e}")
        
        return results
    
    def generate_mesh(self,
                     airfoil_file: str,
                     output_file: str,
                     farfield_radius: float = 20.0) -> bool:
        """
        Generate simple mesh for airfoil (requires gmsh)
        
        Note: This is a simplified mesh generator. For production use,
        a more sophisticated mesh with proper boundary layer resolution
        is recommended.
        """
        
        # Check for gmsh
        if shutil.which('gmsh') is None:
            print("Warning: gmsh not found. Cannot generate mesh.")
            return False
        
        # TODO: Implement mesh generation
        print("Mesh generation not yet implemented")
        print("Use external tool (Pointwise, ICEM, gmsh) to generate mesh")
        return False


# Convenience functions
def create_config(reynolds: float, mach: float, aoa: float,
                 turbulence_model: str = 'SA',
                 mesh_file: str = 'mesh.su2') -> SU2Config:
    """Create SU2 configuration for airfoil analysis"""
    config = SU2Config(case_name=f"Re{reynolds:.0e}_M{mach:.2f}_aoa{aoa:.1f}")
    config.set_flow_conditions(reynolds, mach, aoa)
    config.set_turbulence_model(turbulence_model)
    config.set_mesh(mesh_file)
    config.set_boundary_conditions()
    config.set_output("output")
    return config


def analyze(config_file: str, working_dir: str, 
            timeout: int = 3600) -> Tuple[bool, Dict]:
    """Quick analysis using SU2"""
    solver = SU2Solver()
    return solver.analyze(config_file, working_dir, timeout)
