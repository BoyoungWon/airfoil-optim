"""
XFoil Solver Interface

XFoil: Panel method with integral boundary layer for 2D airfoil analysis

Features:
- Viscous/inviscid analysis
- Boundary layer transition (e^N method)
- Pressure distribution
- CL, CD, CM computation

Limitations:
- Re < 1e6 recommended
- Mach < 0.5 (incompressible correction)
- May not converge for separated flows
"""

import subprocess
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class XFoilSolver:
    """XFoil solver interface"""
    
    def __init__(self, xfoil_path: str = None):
        # Auto-detect XFoil location
        if xfoil_path is None:
            xfoil_path = self._find_xfoil()
        self.xfoil_path = xfoil_path
        self._available = self._check_installation()
    
    def _find_xfoil(self) -> str:
        """Find XFoil binary"""
        # Check common locations
        locations = [
            "xfoil",  # System PATH
            "/usr/local/bin/xfoil",
            "/usr/bin/xfoil",
            str(Path(__file__).parent.parent / "xfoil" / "bin" / "xfoil"),
        ]
        
        for loc in locations:
            try:
                process = subprocess.run(
                    [loc],
                    input=b'quit\n',
                    capture_output=True,
                    timeout=5
                )
                return loc
            except:
                continue
        
        return "xfoil"  # Default, will fail later if not found
    
    def _check_installation(self) -> bool:
        """Check if XFoil is available"""
        try:
            process = subprocess.run(
                [self.xfoil_path],
                input=b'quit\n',
                capture_output=True,
                timeout=5
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    @staticmethod
    def is_available() -> bool:
        """Check if XFoil is installed"""
        locations = [
            "xfoil",
            "/usr/local/bin/xfoil",
            "/usr/bin/xfoil",
            str(Path(__file__).parent.parent / "xfoil" / "bin" / "xfoil"),
        ]
        
        for loc in locations:
            try:
                process = subprocess.run(
                    [loc],
                    input=b'quit\n',
                    capture_output=True,
                    timeout=5
                )
                return True
            except:
                continue
        return False
    
    def analyze(self, 
                airfoil_file: str,
                reynolds: float,
                alpha: float,
                mach: float = 0.0,
                ncrit: float = 9.0,
                iter_limit: int = 100,
                viscous: bool = True) -> Optional[Dict]:
        """
        Run XFoil analysis
        
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
        ncrit : float
            Critical amplification factor (default: 9.0)
        iter_limit : int
            Maximum iterations (default: 100)
        viscous : bool
            Enable viscous analysis (default: True)
            
        Returns:
        --------
        Dict with keys: CL, CD, CM, Top_Xtr, Bot_Xtr, converged
        """
        
        airfoil_path = Path(airfoil_file)
        if not airfoil_path.exists():
            raise FileNotFoundError(f"Airfoil file not found: {airfoil_file}")
        
        # Create XFoil command script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            script_file = f.name
            
            f.write(f"load {airfoil_path.absolute()}\n")
            f.write("pane\n")  # Repanel
            
            if viscous:
                f.write("oper\n")
                f.write("visc\n")
                f.write(f"{reynolds:.0f}\n")
                f.write(f"mach {mach}\n")
                f.write(f"vpar\n")
                f.write(f"n {ncrit}\n")
                f.write("\n")  # Exit vpar
                f.write(f"iter {iter_limit}\n")
                f.write(f"alfa {alpha}\n")
                f.write("\n")  # Back to oper
                f.write("\n")  # Exit oper
            else:
                f.write("oper\n")
                f.write(f"alfa {alpha}\n")
                f.write("\n")
                f.write("\n")
            
            f.write("quit\n")
        
        try:
            # Run XFoil
            with open(script_file, 'r') as script:
                process = subprocess.run(
                    [self.xfoil_path],
                    stdin=script,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            
            # Parse output
            result = self._parse_output(process.stdout, reynolds, alpha, mach)
            return result
            
        except subprocess.TimeoutExpired:
            return {'converged': False, 'error': 'Timeout'}
        except Exception as e:
            return {'converged': False, 'error': str(e)}
        finally:
            # Cleanup
            if os.path.exists(script_file):
                os.remove(script_file)
    
    def _parse_output(self, output: str, reynolds: float, 
                      alpha: float, mach: float) -> Dict:
        """Parse XFoil output"""
        
        result = {
            'reynolds': reynolds,
            'aoa': alpha,
            'mach': mach,
            'CL': None,
            'CD': None,
            'CM': None,
            'Top_Xtr': None,
            'Bot_Xtr': None,
            'converged': False,
            'solver': 'xfoil'
        }
        
        lines = output.split('\n')
        
        for line in lines:
            # Look for converged results
            if 'CL =' in line and 'CD =' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'CL':
                            result['CL'] = float(parts[i + 2])
                        elif part == 'CD':
                            result['CD'] = float(parts[i + 2])
                        elif part == 'CM':
                            result['CM'] = float(parts[i + 2])
                    result['converged'] = True
                except (ValueError, IndexError):
                    pass
            
            # Look for transition locations
            if 'Top Xtr' in line or 'Bot Xtr' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Top' in part and i + 2 < len(parts):
                            result['Top_Xtr'] = float(parts[i + 2])
                        elif 'Bot' in part and i + 2 < len(parts):
                            result['Bot_Xtr'] = float(parts[i + 2])
                except (ValueError, IndexError):
                    pass
        
        return result
    
    def analyze_sweep(self,
                      airfoil_file: str,
                      reynolds: float,
                      alpha_range: Tuple[float, float, float],
                      mach: float = 0.0,
                      ncrit: float = 9.0) -> List[Dict]:
        """
        Run AoA sweep analysis
        
        Parameters:
        -----------
        alpha_range : Tuple[float, float, float]
            (min, max, step) in degrees
            
        Returns:
        --------
        List of result dictionaries
        """
        
        alpha_min, alpha_max, alpha_step = alpha_range
        results = []
        
        alpha = alpha_min
        while alpha <= alpha_max:
            result = self.analyze(
                airfoil_file, reynolds, alpha, 
                mach=mach, ncrit=ncrit
            )
            if result:
                results.append(result)
            alpha += alpha_step
        
        return results


# Convenience function
def analyze(airfoil_file: str, reynolds: float, alpha: float, 
            mach: float = 0.0, **kwargs) -> Optional[Dict]:
    """Quick analysis using XFoil"""
    solver = XFoilSolver()
    return solver.analyze(airfoil_file, reynolds, alpha, mach, **kwargs)
