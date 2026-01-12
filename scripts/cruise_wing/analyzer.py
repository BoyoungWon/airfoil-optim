"""
Airfoil Analyzer Module

XFOIL을 이용한 익형 분석 모듈
- Single point analysis
- Polar analysis  
- Multi-point analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
import subprocess
import os


class AirfoilAnalyzer:
    """
    XFOIL 기반 익형 분석기
    
    Cruise Wing 최적화에 최적화된 설정:
    - Ncrit: 10-11 (flight condition)
    - Panels: 160
    - Iter: 100-200
    """
    
    def __init__(self, xfoil_path: Optional[str] = None,
                 ncrit: float = 10.0,
                 n_panels: int = 160,
                 max_iter: int = 200,
                 timeout: int = 60):
        """
        Initialize analyzer
        
        Parameters
        ----------
        xfoil_path : str, optional
            XFOIL 실행 파일 경로
        ncrit : float
            Critical amplification factor (9-11 for flight, 3-5 for wind tunnel)
        n_panels : int
            Number of panels for airfoil discretization
        max_iter : int
            Maximum XFOIL iterations
        timeout : int
            XFOIL execution timeout (seconds)
        """
        self.xfoil_path = xfoil_path or "xfoil"
        self.ncrit = ncrit
        self.n_panels = n_panels
        self.max_iter = max_iter
        self.timeout = timeout
        
    def _write_airfoil(self, coords: np.ndarray, filepath: Path) -> None:
        """익형 좌표 파일 저장"""
        with open(filepath, 'w') as f:
            f.write("Airfoil\n")
            for x, y in coords:
                f.write(f"{x:12.8f}  {y:12.8f}\n")
    
    def _parse_polar_file(self, filepath: Path) -> Optional[Dict]:
        """
        XFOIL polar 파일 파싱
        
        Returns
        -------
        dict
            {'alpha': [], 'CL': [], 'CD': [], 'CDp': [], 'CM': [], 'Top_Xtr': [], 'Bot_Xtr': []}
        """
        if not filepath.exists():
            return None
        
        data = {
            'alpha': [], 'CL': [], 'CD': [], 'CDp': [],
            'CM': [], 'Top_Xtr': [], 'Bot_Xtr': []
        }
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find data start (after header)
        data_start = 0
        for i, line in enumerate(lines):
            if '---' in line:
                data_start = i + 1
                break
        
        # Parse data lines
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    data['alpha'].append(float(parts[0]))
                    data['CL'].append(float(parts[1]))
                    data['CD'].append(float(parts[2]))
                    data['CDp'].append(float(parts[3]))
                    data['CM'].append(float(parts[4]))
                    data['Top_Xtr'].append(float(parts[5]))
                    data['Bot_Xtr'].append(float(parts[6]))
                except (ValueError, IndexError):
                    continue
        
        if not data['alpha']:
            return None
        
        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
        
        # Calculate L/D
        data['L/D'] = np.where(data['CD'] > 0, data['CL'] / data['CD'], 0)
        
        return data
    
    def analyze_single(self, coords: np.ndarray, 
                       reynolds: float, aoa: float, mach: float = 0.0,
                       verbose: bool = False) -> Optional[Dict]:
        """
        단일 조건에서 익형 분석
        
        Parameters
        ----------
        coords : np.ndarray
            Airfoil coordinates (N, 2)
        reynolds : float
            Reynolds number
        aoa : float
            Angle of attack (degrees)
        mach : float
            Mach number
        verbose : bool
            Print XFOIL output
            
        Returns
        -------
        dict or None
            {'CL': ..., 'CD': ..., 'CM': ..., 'L/D': ..., 'converged': bool}
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write airfoil file
            coord_file = tmpdir / "airfoil.dat"
            self._write_airfoil(coords, coord_file)
            
            # Output file
            polar_file = tmpdir / "polar.txt"
            
            # XFOIL commands
            commands = f"""PLOP
G

LOAD {coord_file}

PPAR
N
{self.n_panels}


OPER
VISC {reynolds}
MACH {mach}
VPAR
N
{self.ncrit}

ITER {self.max_iter}
PACC
{polar_file}

ALFA {aoa}

QUIT
"""
            
            try:
                result = subprocess.run(
                    [self.xfoil_path],
                    input=commands,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout
                )
                
                if verbose:
                    print(result.stdout)
                
                # Parse results
                polar_data = self._parse_polar_file(polar_file)
                
                if polar_data and len(polar_data['CL']) > 0:
                    return {
                        'CL': polar_data['CL'][0],
                        'CD': polar_data['CD'][0],
                        'CM': polar_data['CM'][0],
                        'L/D': polar_data['L/D'][0],
                        'Top_Xtr': polar_data['Top_Xtr'][0],
                        'Bot_Xtr': polar_data['Bot_Xtr'][0],
                        'converged': True
                    }
                else:
                    return None
                    
            except subprocess.TimeoutExpired:
                if verbose:
                    print("XFOIL timeout")
                return None
            except FileNotFoundError:
                print(f"XFOIL not found at: {self.xfoil_path}")
                return None
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                return None
    
    def analyze_polar(self, coords: np.ndarray, reynolds: float,
                      aoa_range: Tuple[float, float] = (-4, 12),
                      aoa_step: float = 0.5,
                      mach: float = 0.0,
                      verbose: bool = False) -> Optional[Dict]:
        """
        Polar 분석 (여러 받음각)
        
        Parameters
        ----------
        coords : np.ndarray
            Airfoil coordinates
        reynolds : float
            Reynolds number
        aoa_range : tuple
            AoA range (min, max) in degrees
        aoa_step : float
            AoA step in degrees
        mach : float
            Mach number
        verbose : bool
            Print output
            
        Returns
        -------
        dict or None
            Polar data arrays
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write airfoil file
            coord_file = tmpdir / "airfoil.dat"
            self._write_airfoil(coords, coord_file)
            
            # Output file
            polar_file = tmpdir / "polar.txt"
            
            # Generate XFOIL commands for polar sweep
            aoa_min, aoa_max = aoa_range
            
            commands = f"""PLOP
G

LOAD {coord_file}

PPAR
N
{self.n_panels}


OPER
VISC {reynolds}
MACH {mach}
VPAR
N
{self.ncrit}

ITER {self.max_iter}
PACC
{polar_file}

ASEQ {aoa_min} {aoa_max} {aoa_step}

QUIT
"""
            
            try:
                result = subprocess.run(
                    [self.xfoil_path],
                    input=commands,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout * 3  # Longer timeout for polar
                )
                
                if verbose:
                    print(result.stdout)
                
                return self._parse_polar_file(polar_file)
                
            except subprocess.TimeoutExpired:
                if verbose:
                    print("XFOIL timeout during polar sweep")
                return None
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                return None
    
    def analyze_multi_point(self, coords: np.ndarray,
                            design_points: List[Dict],
                            verbose: bool = False) -> Optional[Dict]:
        """
        다중 설계점 분석
        
        Parameters
        ----------
        coords : np.ndarray
            Airfoil coordinates
        design_points : list
            List of dicts with 'reynolds', 'aoa', 'mach', 'weight'
        verbose : bool
            Print output
            
        Returns
        -------
        dict or None
            Weighted average results
        """
        results_list = []
        weights = []
        
        for dp in design_points:
            result = self.analyze_single(
                coords,
                reynolds=dp['reynolds'],
                aoa=dp['aoa'],
                mach=dp.get('mach', 0.0),
                verbose=verbose
            )
            
            if result is not None and result.get('converged', False):
                results_list.append(result)
                weights.append(dp.get('weight', 1.0))
        
        if not results_list:
            return None
        
        # Weighted average
        weights = np.array(weights) / np.sum(weights)
        
        combined = {}
        for key in ['CL', 'CD', 'CM', 'L/D']:
            if key in results_list[0]:
                values = [r[key] for r in results_list]
                combined[key] = np.average(values, weights=weights)
        
        combined['n_converged'] = len(results_list)
        combined['n_total'] = len(design_points)
        
        return combined
    
    def find_max_ld(self, coords: np.ndarray, reynolds: float,
                    aoa_range: Tuple[float, float] = (0, 10),
                    mach: float = 0.0) -> Optional[Dict]:
        """
        최대 L/D와 해당 받음각 찾기
        
        Parameters
        ----------
        coords : np.ndarray
            Airfoil coordinates
        reynolds : float
            Reynolds number
        aoa_range : tuple
            Search range for AoA
        mach : float
            Mach number
            
        Returns
        -------
        dict or None
            {'alpha_opt': ..., 'CL': ..., 'CD': ..., 'L/D_max': ...}
        """
        polar = self.analyze_polar(coords, reynolds, aoa_range, 0.5, mach)
        
        if polar is None or len(polar['L/D']) == 0:
            return None
        
        # Find maximum L/D
        idx_max = np.argmax(polar['L/D'])
        
        return {
            'alpha_opt': polar['alpha'][idx_max],
            'CL': polar['CL'][idx_max],
            'CD': polar['CD'][idx_max],
            'L/D_max': polar['L/D'][idx_max],
            'CM': polar['CM'][idx_max]
        }
    
    def find_design_alpha(self, coords: np.ndarray, reynolds: float,
                          target_cl: float, mach: float = 0.0,
                          aoa_range: Tuple[float, float] = (-4, 12)) -> Optional[Dict]:
        """
        목표 CL에 해당하는 받음각 찾기
        
        Parameters
        ----------
        coords : np.ndarray
            Airfoil coordinates
        reynolds : float
            Reynolds number
        target_cl : float
            Target lift coefficient
        mach : float
            Mach number
        aoa_range : tuple
            Search range
            
        Returns
        -------
        dict or None
            {'alpha': ..., 'CL': ..., 'CD': ..., 'L/D': ...}
        """
        polar = self.analyze_polar(coords, reynolds, aoa_range, 0.5, mach)
        
        if polar is None or len(polar['CL']) == 0:
            return None
        
        # Interpolate to find alpha for target CL
        from scipy import interpolate
        
        try:
            f = interpolate.interp1d(polar['CL'], polar['alpha'], 
                                    bounds_error=False, fill_value='extrapolate')
            alpha_target = float(f(target_cl))
            
            # Get other values at this alpha
            f_cd = interpolate.interp1d(polar['alpha'], polar['CD'])
            f_cm = interpolate.interp1d(polar['alpha'], polar['CM'])
            
            cd = float(f_cd(alpha_target))
            cm = float(f_cm(alpha_target))
            ld = target_cl / cd if cd > 0 else 0
            
            return {
                'alpha': alpha_target,
                'CL': target_cl,
                'CD': cd,
                'CM': cm,
                'L/D': ld
            }
        except Exception:
            return None


def run_xfoil_analysis(coords: np.ndarray, reynolds: float, aoa: float,
                       mach: float = 0.0, n_iter: int = 200,
                       verbose: bool = False) -> Optional[Dict]:
    """
    Convenience function for single-point XFOIL analysis
    
    Backward compatible with existing interface
    """
    analyzer = AirfoilAnalyzer(max_iter=n_iter)
    return analyzer.analyze_single(coords, reynolds, aoa, mach, verbose)


def run_polar_analysis(coords: np.ndarray, reynolds: float,
                       aoa_list: Union[np.ndarray, List[float]],
                       mach: float = 0.0) -> Optional[Dict]:
    """
    Convenience function for polar analysis
    """
    aoa_array = np.array(aoa_list)
    aoa_range = (aoa_array.min(), aoa_array.max())
    aoa_step = np.diff(aoa_array).mean() if len(aoa_array) > 1 else 0.5
    
    analyzer = AirfoilAnalyzer()
    return analyzer.analyze_polar(coords, reynolds, aoa_range, aoa_step, mach)
