import subprocess
import os
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

class XfoilWrapper:
    """X-foil을 Python에서 제어하기 위한 래퍼 클래스"""
    
    def __init__(self, xfoil_path: str = "xfoil"):
        self.xfoil_path = xfoil_path
        self.logger = logging.getLogger(__name__)
        
    def create_airfoil_dat(self, coords: np.ndarray, filename: str) -> str:
        """
        에어포일 좌표를 X-foil 형식 .dat 파일로 저장
        
        Args:
            coords: nx2 numpy array (x, y 좌표)
            filename: 저장할 파일명
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("AIRFOIL\n")
            for x, y in coords:
                f.write(f"{x:.6f} {y:.6f}\n")
        
        return str(filepath)
    
    def run_analysis(self, 
                    airfoil_coords: np.ndarray,
                    alpha_range: List[float],
                    reynolds: float = 1e6,
                    max_iter: int = 200) -> Dict:
        """
        X-foil 해석 실행
        
        Args:
            airfoil_coords: 에어포일 좌표
            alpha_range: 받음각 범위 [deg]
            reynolds: 레이놀즈 수
            max_iter: 최대 반복 횟수
            
        Returns:
            Dict: 해석 결과 (Cl, Cd, Cm 등)
        """
        
        results = {
            'alpha': [],
            'cl': [],
            'cd': [],
            'cm': [],
            'converged': []
        }
        
        # 임시 파일 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            airfoil_file = os.path.join(temp_dir, "airfoil.dat")
            output_file = os.path.join(temp_dir, "output.txt")
            
            # 에어포일 파일 생성
            self.create_airfoil_dat(airfoil_coords, airfoil_file)
            
            # X-foil 명령어 생성
            commands = [
                f"LOAD {airfoil_file}",
                "",  # 에어포일 이름 입력
                "PANE",
                "OPER",
                f"VISC {reynolds}",
                f"ITER {max_iter}",
                f"PACC {output_file}",
                "",  # dump 파일 이름 (사용하지 않음)
            ]
            
            # 각 받음각에 대해 해석
            for alpha in alpha_range:
                commands.append(f"ALFA {alpha}")
            
            commands.extend([
                "PACC",  # 결과 저장 종료
                "",
                "QUIT"
            ])
            
            # X-foil 실행
            command_string = "\n".join(commands)
            
            try:
                process = subprocess.Popen(
                    [self.xfoil_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temp_dir
                )
                
                stdout, stderr = process.communicate(input=command_string, timeout=60)
                
                # 결과 파싱
                if os.path.exists(output_file):
                    results = self._parse_output_file(output_file)
                else:
                    self.logger.warning("X-foil output file not found")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error("X-foil analysis timeout")
            except Exception as e:
                self.logger.error(f"X-foil analysis error: {e}")
        
        return results
    
    def _parse_output_file(self, filepath: str) -> Dict:
        """X-foil 출력 파일 파싱"""
        results = {
            'alpha': [],
            'cl': [],
            'cd': [],
            'cm': [],
            'converged': []
        }
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # 헤더 스킵하고 데이터 파싱
            data_started = False
            for line in lines:
                if 'alpha' in line and 'CL' in line:
                    data_started = True
                    continue
                    
                if data_started and line.strip():
                    try:
                        values = line.split()
                        if len(values) >= 4:
                            alpha = float(values[0])
                            cl = float(values[1])
                            cd = float(values[2])
                            cm = float(values[4]) if len(values) > 4 else 0.0
                            
                            results['alpha'].append(alpha)
                            results['cl'].append(cl)
                            results['cd'].append(cd)
                            results['cm'].append(cm)
                            results['converged'].append(True)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error parsing output file: {e}")
        
        return results
    
    def calculate_objectives(self, results: Dict, alpha_range: List[float]) -> Dict[str, float]:
        """
        최적화 목적함수 계산
        
        Returns:
            Dict: max_cl, min_cd, min_dcl_dcm
        """
        if not results['cl']:
            return {'max_cl': -999, 'min_cd': 999, 'min_dcl_dcm': 999}
        
        cl_array = np.array(results['cl'])
        cd_array = np.array(results['cd'])
        cm_array = np.array(results['cm'])
        alpha_array = np.array(results['alpha'])
        
        # 목적함수 계산
        max_cl = np.max(cl_array)
        min_cd = np.min(cd_array)
        
        # dCl/dCm 계산 (수치 미분)
        dcl_dcm_values = []
        for i in range(1, len(cl_array)):
            dcl = cl_array[i] - cl_array[i-1]
            dcm = cm_array[i] - cm_array[i-1]
            if abs(dcm) > 1e-6:
                dcl_dcm_values.append(abs(dcl / dcm))
        
        min_dcl_dcm = np.min(dcl_dcm_values) if dcl_dcm_values else 999
        
        return {
            'max_cl': max_cl,
            'min_cd': min_cd,
            'min_dcl_dcm': min_dcl_dcm
        }
    