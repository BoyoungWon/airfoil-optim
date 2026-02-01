#!/usr/bin/env python3
"""
SU2 Interface Module

SU2 RANS solver를 위한 configuration 생성 및 실행 인터페이스
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


class SU2Config:
    """SU2 configuration file generator"""
    
    def __init__(self, airfoil_file: str, case_name: str):
        self.airfoil_file = Path(airfoil_file)
        self.case_name = case_name
        self.config = {}
        
    def set_physics(self, mach: float, reynolds: float, 
                    aoa: float = 0.0, temperature: float = 288.15):
        """
        물리 조건 설정
        
        Parameters:
        -----------
        mach : float
            Mach number
        reynolds : float
            Reynolds number
        aoa : float
            Angle of attack (degrees)
        temperature : float
            Freestream temperature (K)
        """
        
        # Calculate freestream conditions
        gas_constant = 287.05  # J/(kg·K) for air
        gamma = 1.4
        
        # Speed of sound at given temperature
        speed_of_sound = np.sqrt(gamma * gas_constant * temperature)
        velocity = mach * speed_of_sound
        
        # Dynamic viscosity (Sutherland's law)
        T_ref = 273.15
        mu_ref = 1.716e-5
        S = 110.4
        mu = mu_ref * (temperature / T_ref)**1.5 * (T_ref + S) / (temperature + S)
        
        # Freestream density from Reynolds number
        # Re = rho * V * L / mu, assuming chord length L = 1.0
        chord = 1.0
        rho = reynolds * mu / (velocity * chord)
        
        # Pressure from ideal gas law
        pressure = rho * gas_constant * temperature
        
        self.config['MACH_NUMBER'] = mach
        self.config['AOA'] = aoa
        self.config['FREESTREAM_TEMPERATURE'] = temperature
        self.config['REYNOLDS_NUMBER'] = reynolds
        self.config['REYNOLDS_LENGTH'] = chord
        
        return self
    
    def set_turbulence_model(self, model: str = 'SA'):
        """
        난류 모델 설정
        
        Parameters:
        -----------
        model : str
            'SA' (Spalart-Allmaras), 'SST' (k-omega SST), 
            'SA-neg' (negative SA), 'LM' (Langtry-Menter transition)
        """
        
        model_map = {
            'SA': 'RANS',
            'SST': 'RANS', 
            'SA-neg': 'RANS',
            'LM': 'RANS'
        }
        
        turb_model_map = {
            'SA': 'SA',
            'SST': 'SST',
            'SA-neg': 'SA_NEG',
            'LM': 'SA'
        }
        
        self.config['SOLVER'] = model_map.get(model, 'RANS')
        self.config['KIND_TURB_MODEL'] = turb_model_map.get(model, 'SA')
        
        if model == 'LM':
            self.config['KIND_TRANS_MODEL'] = 'LM'
        
        return self
    
    def set_numerical_settings(self, cfl: float = 5.0, 
                              mg_levels: int = 3,
                              iter_max: int = 5000):
        """수치 해석 설정"""
        
        self.config['CFL_NUMBER'] = cfl
        self.config['CFL_ADAPT'] = 'YES'
        self.config['CFL_ADAPT_PARAM'] = [0.5, 1.5, cfl, 100.0]
        
        self.config['MGLEVEL'] = mg_levels
        self.config['MGCYCLE'] = 'V_CYCLE'
        
        self.config['ITER'] = iter_max
        self.config['CONV_RESIDUAL_MINVAL'] = -12
        
        return self
    
    def set_boundary_conditions(self):
        """경계 조건 설정 (2D airfoil)"""
        
        self.config['MARKER_HEATFLUX'] = '( airfoil, 0.0 )'
        self.config['MARKER_FAR'] = '( farfield )'
        self.config['MARKER_PLOTTING'] = '( airfoil )'
        self.config['MARKER_MONITORING'] = '( airfoil )'
        
        return self
    
    def set_output(self, output_dir: str):
        """출력 설정"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.config['OUTPUT_FILES'] = ['RESTART', 'PARAVIEW', 'SURFACE_CSV']
        self.config['MESH_FILENAME'] = str(self.airfoil_file.absolute())
        self.config['MESH_FORMAT'] = 'SU2'
        
        self.config['CONV_FILENAME'] = f'history'
        self.config['RESTART_FILENAME'] = f'restart'
        self.config['VOLUME_FILENAME'] = f'volume'
        self.config['SURFACE_FILENAME'] = f'surface'
        
        self.config['OUTPUT_WRT_FREQ'] = 100
        self.config['SCREEN_WRT_FREQ_INNER'] = 10
        
        return self
    
    def write_config(self, config_file: str):
        """Configuration 파일 생성"""
        
        config_path = Path(config_file)
        
        with open(config_path, 'w') as f:
            f.write(f"% SU2 Configuration File\n")
            f.write(f"% Case: {self.case_name}\n")
            f.write(f"% Auto-generated configuration\n\n")
            
            # Physics
            f.write(f"% Physical Definition\n")
            f.write(f"PHYSICAL_PROBLEM= {self.config.get('SOLVER', 'RANS')}\n")
            f.write(f"KIND_TURB_MODEL= {self.config.get('KIND_TURB_MODEL', 'SA')}\n")
            if 'KIND_TRANS_MODEL' in self.config:
                f.write(f"KIND_TRANS_MODEL= {self.config['KIND_TRANS_MODEL']}\n")
            f.write(f"MATH_PROBLEM= DIRECT\n")
            f.write(f"RESTART_SOL= NO\n\n")
            
            # Freestream
            f.write(f"% Freestream Conditions\n")
            f.write(f"MACH_NUMBER= {self.config.get('MACH_NUMBER', 0.5)}\n")
            f.write(f"AOA= {self.config.get('AOA', 0.0)}\n")
            f.write(f"FREESTREAM_TEMPERATURE= {self.config.get('FREESTREAM_TEMPERATURE', 288.15)}\n")
            f.write(f"REYNOLDS_NUMBER= {self.config.get('REYNOLDS_NUMBER', 1e6)}\n")
            f.write(f"REYNOLDS_LENGTH= {self.config.get('REYNOLDS_LENGTH', 1.0)}\n\n")
            
            # Reference values
            f.write(f"% Reference Values\n")
            f.write(f"REF_LENGTH= 1.0\n")
            f.write(f"REF_AREA= 1.0\n\n")
            
            # Boundary conditions
            f.write(f"% Boundary Conditions\n")
            f.write(f"MARKER_HEATFLUX= {self.config.get('MARKER_HEATFLUX', '( airfoil, 0.0 )')}\n")
            f.write(f"MARKER_FAR= {self.config.get('MARKER_FAR', '( farfield )')}\n")
            f.write(f"MARKER_PLOTTING= {self.config.get('MARKER_PLOTTING', '( airfoil )')}\n")
            f.write(f"MARKER_MONITORING= {self.config.get('MARKER_MONITORING', '( airfoil )')}\n\n")
            
            # Numerical settings
            f.write(f"% Numerical Method\n")
            f.write(f"NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES\n")
            f.write(f"CFL_NUMBER= {self.config.get('CFL_NUMBER', 5.0)}\n")
            f.write(f"CFL_ADAPT= {self.config.get('CFL_ADAPT', 'YES')}\n")
            if 'CFL_ADAPT_PARAM' in self.config:
                params = self.config['CFL_ADAPT_PARAM']
                f.write(f"CFL_ADAPT_PARAM= ( {params[0]}, {params[1]}, {params[2]}, {params[3]} )\n")
            f.write(f"ITER= {self.config.get('ITER', 5000)}\n\n")
            
            # Linear solver
            f.write(f"% Linear Solver\n")
            f.write(f"LINEAR_SOLVER= FGMRES\n")
            f.write(f"LINEAR_SOLVER_PREC= ILU\n")
            f.write(f"LINEAR_SOLVER_ERROR= 1E-6\n")
            f.write(f"LINEAR_SOLVER_ITER= 10\n\n")
            
            # Multigrid
            f.write(f"% Multigrid\n")
            f.write(f"MGLEVEL= {self.config.get('MGLEVEL', 3)}\n")
            f.write(f"MGCYCLE= {self.config.get('MGCYCLE', 'V_CYCLE')}\n\n")
            
            # Convergence
            f.write(f"% Convergence Criteria\n")
            f.write(f"CONV_RESIDUAL_MINVAL= {self.config.get('CONV_RESIDUAL_MINVAL', -12)}\n")
            f.write(f"CONV_STARTITER= 10\n")
            f.write(f"CONV_CAUCHY_ELEMS= 100\n")
            f.write(f"CONV_CAUCHY_EPS= 1E-6\n\n")
            
            # I/O
            f.write(f"% Input/Output\n")
            f.write(f"MESH_FILENAME= {self.config.get('MESH_FILENAME', 'mesh.su2')}\n")
            f.write(f"MESH_FORMAT= SU2\n")
            f.write(f"CONV_FILENAME= {self.config.get('CONV_FILENAME', 'history')}\n")
            f.write(f"RESTART_FILENAME= {self.config.get('RESTART_FILENAME', 'restart')}\n")
            f.write(f"VOLUME_FILENAME= {self.config.get('VOLUME_FILENAME', 'volume')}\n")
            f.write(f"SURFACE_FILENAME= {self.config.get('SURFACE_FILENAME', 'surface')}\n")
            f.write(f"OUTPUT_WRT_FREQ= {self.config.get('OUTPUT_WRT_FREQ', 100)}\n")
            f.write(f"SCREEN_WRT_FREQ_INNER= {self.config.get('SCREEN_WRT_FREQ_INNER', 10)}\n")
        
        return config_path


class SU2Interface:
    """SU2 solver interface"""
    
    @staticmethod
    def check_installation() -> bool:
        """SU2 설치 확인"""
        try:
            result = subprocess.run(['SU2_CFD', '-h'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    @staticmethod
    def run_analysis(config_file: str, working_dir: str, 
                    timeout: int = 3600) -> Tuple[bool, str]:
        """
        SU2 해석 실행
        
        Parameters:
        -----------
        config_file : str
            SU2 configuration file
        working_dir : str
            Working directory
        timeout : int
            Timeout in seconds (default: 3600 = 1 hour)
            
        Returns:
        --------
        Tuple[bool, str] : (success, output_message)
        """
        
        config_path = Path(config_file)
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\nRunning SU2_CFD...")
            print(f"  Config: {config_path}")
            print(f"  Working dir: {work_path}")
            
            process = subprocess.run(
                ['SU2_CFD', str(config_path.absolute())],
                cwd=str(work_path.absolute()),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if process.returncode == 0:
                return True, "SU2 analysis completed successfully"
            else:
                return False, f"SU2 failed with return code {process.returncode}\n{process.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, f"SU2 analysis timed out (>{timeout}s)"
        except FileNotFoundError:
            return False, "SU2_CFD not found. Please install SU2."
        except Exception as e:
            return False, f"Error running SU2: {str(e)}"
    
    @staticmethod
    def parse_results(surface_file: str) -> Optional[Dict]:
        """
        SU2 surface results 파싱
        
        Parameters:
        -----------
        surface_file : str
            Surface CSV file from SU2
            
        Returns:
        --------
        Dict : {'CL': float, 'CD': float, 'CM': float, ...}
        """
        
        surface_path = Path(surface_file)
        
        if not surface_path.exists():
            print(f"✗ Surface file not found: {surface_file}")
            return None
        
        try:
            # Read SU2 surface CSV
            df = pd.read_csv(surface_path)
            
            # Extract force coefficients (usually in the header or separate file)
            # This is simplified - actual implementation depends on SU2 output format
            
            results = {
                'CL': None,
                'CD': None,
                'CM': None,
                'converged': True
            }
            
            # TODO: Parse actual SU2 output format
            # This requires reading the history file or forces file
            
            return results
            
        except Exception as e:
            print(f"✗ Error parsing SU2 results: {e}")
            return None


if __name__ == "__main__":
    """Test SU2 interface"""
    
    # Check installation
    if SU2Interface.check_installation():
        print("✓ SU2 is installed and available")
    else:
        print("✗ SU2 not found")
        print("  Install from: https://su2code.github.io/")
    
    # Test config generation
    print("\nGenerating test SU2 configuration...")
    
    config = SU2Config("test_airfoil.dat", "test_case")
    config.set_physics(mach=0.75, reynolds=3e6, aoa=2.0)
    config.set_turbulence_model('SA')
    config.set_numerical_settings(cfl=5.0, iter_max=5000)
    config.set_boundary_conditions()
    config.set_output("test_output")
    
    test_config = "test_su2.cfg"
    config.write_config(test_config)
    
    print(f"✓ Test configuration written to: {test_config}")
