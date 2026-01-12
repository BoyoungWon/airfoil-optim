"""
XFOIL Interface Module

XFOIL과의 인터페이스를 제공하는 모듈
"""

import subprocess
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, Optional


def run_xfoil_analysis(coords: np.ndarray, 
                       reynolds: float, 
                       aoa: float, 
                       mach: float = 0.0,
                       n_iter: int = 200,
                       verbose: bool = False) -> Optional[Dict]:
    """
    Run XFOIL analysis for given airfoil coordinates
    
    Parameters:
    -----------
    coords : np.ndarray
        Airfoil coordinates (N, 2)
    reynolds : float
        Reynolds number
    aoa : float
        Angle of attack in degrees
    mach : float
        Mach number (default 0.0)
    n_iter : int
        Maximum iterations (default 200)
    verbose : bool
        Print output (default False)
    
    Returns:
    --------
    dict or None
        Analysis results with CL, CD, CM, etc. or None if failed
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save airfoil coordinates
        coord_file = tmpdir / "airfoil.dat"
        with open(coord_file, 'w') as f:
            f.write("Airfoil\n")
            for x, y in coords:
                f.write(f"{x:12.8f}  {y:12.8f}\n")
        
        # Create XFOIL input script
        script_file = tmpdir / "xfoil_input.txt"
        output_file = tmpdir / "results.txt"
        
        with open(script_file, 'w') as f:
            f.write(f"LOAD {coord_file}\n")
            f.write("\n")  # Accept airfoil name
            f.write("PANE\n")  # Repanel
            f.write("OPER\n")  # Enter OPER menu
            f.write(f"VISC {reynolds}\n")
            if mach > 0:
                f.write(f"MACH {mach}\n")
            f.write(f"ITER {n_iter}\n")
            f.write(f"ALFA {aoa}\n")
            f.write(f"DUMP {output_file}\n")
            f.write("\n")
            f.write("QUIT\n")
        
        # Run XFOIL
        try:
            result = subprocess.run(
                ['xfoil'],
                stdin=open(script_file),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                cwd=tmpdir
            )
            
            if verbose:
                print(result.stdout.decode())
            
            # Parse results
            if output_file.exists():
                return parse_xfoil_dump(output_file)
            else:
                return None
        
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if verbose:
                print(f"XFOIL analysis failed: {e}")
            return None


def parse_xfoil_dump(dump_file: Path) -> Dict:
    """
    Parse XFOIL dump file
    
    Returns:
    --------
    dict
        Results with CL, CD, CM, L/D, etc.
    """
    
    results = {}
    
    with open(dump_file, 'r') as f:
        for line in f:
            if 'CL =' in line:
                results['CL'] = float(line.split('=')[1].strip())
            elif 'CD =' in line:
                results['CD'] = float(line.split('=')[1].strip())
            elif 'CM =' in line:
                results['CM'] = float(line.split('=')[1].strip())
    
    # Calculate L/D
    if 'CL' in results and 'CD' in results and results['CD'] > 0:
        results['L/D'] = results['CL'] / results['CD']
    
    return results


def run_multi_point_analysis(coords: np.ndarray,
                             design_points: list,
                             verbose: bool = False) -> Optional[Dict]:
    """
    Run XFOIL analysis at multiple design points
    
    Parameters:
    -----------
    coords : np.ndarray
        Airfoil coordinates
    design_points : list
        List of dicts with 'reynolds', 'aoa', 'mach', 'weight'
    verbose : bool
        Print output
    
    Returns:
    --------
    dict or None
        Weighted average of results
    """
    
    results_list = []
    weights = []
    
    for dp in design_points:
        result = run_xfoil_analysis(
            coords,
            reynolds=dp['reynolds'],
            aoa=dp['aoa'],
            mach=dp.get('mach', 0.0),
            verbose=verbose
        )
        
        if result is not None:
            results_list.append(result)
            weights.append(dp['weight'])
    
    if not results_list:
        return None
    
    # Weighted average
    weights = np.array(weights) / np.sum(weights)
    
    combined = {}
    for key in results_list[0].keys():
        values = [r[key] for r in results_list]
        combined[key] = np.average(values, weights=weights)
    
    return combined
