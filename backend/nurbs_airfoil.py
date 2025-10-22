import subprocess
import os
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

class XfoilWrapper:
    """X-foil을 Python에서 제어하기 위한 래퍼 클래스 (Ncrit 등 세부 파라미터 지원)"""
    
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
                    mach: float = 0.0,
                    ncrit: float = 9.0,
                    max_iter: int = 200,
                    viscous: bool = True) -> Dict:
        """
        X-foil 해석 실행 (세부 파라미터 지원)
        
        Args:
            airfoil_coords: 에어포일 좌표
            alpha_range: 받음각 범위 [deg]
            reynolds: 레이놀즈 수
            mach: 마하수
            ncrit: Critical amplification factor (transition 민감도)
            max_iter: 최대 반복 횟수
            viscous: 점성 해석 여부
            
        Returns:
            Dict: 해석 결과 (Cl, Cd, Cm 등)
        """
        
        results = {
            'alpha': [],
            'cl': [],
            'cd': [],
            'cm': [],
            'converged': [],
            'analysis_info': {
                'reynolds': reynolds,
                'mach': mach,
                'ncrit': ncrit,
                'max_iter': max_iter,
                'viscous': viscous
            }
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
                "PANE",  # 패널링
                "OPER",  # 운용 모드
            ]
            
            # 점성 해석 설정
            if viscous:
                commands.extend([
                    f"VISC {reynolds:.0f}",  # 레이놀즈 수 설정
                    f"MACH {mach:.3f}",     # 마하수 설정
                ])
                
                # Ncrit 설정 (BL 메뉴에서)
                commands.extend([
                    "BL",              # Boundary Layer 메뉴 진입
                    f"NCRIT {ncrit:.1f}",  # Critical amplification factor 설정
                    "",                # BL 메뉴 종료
                ])
            
            commands.extend([
                f"ITER {max_iter}",    # 최대 반복 횟수
                f"PACC {output_file}", # 결과 저장 시작
                "",                    # dump 파일 이름 (사용하지 않음)
            ])
            
            # 각 받음각에 대해 해석
            for alpha in alpha_range:
                commands.append(f"ALFA {alpha:.2f}")
            
            commands.extend([
                "PACC",  # 결과 저장 종료
                "",
                "QUIT"
            ])
            
            # X-foil 실행
            command_string = "\n".join(commands)
            
            try:
                self.logger.info(f"Running X-foil analysis: Re={reynolds:.0f}, M={mach:.3f}, Ncrit={ncrit:.1f}")
                
                process = subprocess.Popen(
                    [self.xfoil_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temp_dir
                )
                
                stdout, stderr = process.communicate(input=command_string, timeout=120)
                
                # 결과 파싱
                if os.path.exists(output_file):
                    results = self._parse_output_file(output_file)
                    results['analysis_info'] = {
                        'reynolds': reynolds,
                        'mach': mach,
                        'ncrit': ncrit,
                        'max_iter': max_iter,
                        'viscous': viscous
                    }
                    
                    self.logger.info(f"X-foil analysis completed: {len(results['alpha'])} points converged")
                else:
                    self.logger.warning("X-foil output file not found")
                
                # X-foil 출력 로깅 (디버깅용)
                if stderr:
                    self.logger.warning(f"X-foil stderr: {stderr[:500]}...")
                    
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
                if 'alpha' in line.lower() and 'cl' in line.lower():
                    data_started = True
                    continue
                    
                if data_started and line.strip():
                    try:
                        values = line.split()
                        if len(values) >= 7:  # X-foil 표준 출력 형식
                            alpha = float(values[0])
                            cl = float(values[1])
                            cd = float(values[2])
                            cdp = float(values[3])  # Pressure drag
                            cm = float(values[4])
                            top_xtr = float(values[5])  # Top transition
                            bot_xtr = float(values[6])  # Bottom transition
                            
                            results['alpha'].append(alpha)
                            results['cl'].append(cl)
                            results['cd'].append(cd)
                            results['cm'].append(cm)
                            results['converged'].append(True)
                            
                    except (ValueError, IndexError):
                        # 수렴하지 않은 포인트는 건너뛰기
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
        
        # 추가 성능 지표
        lift_drag_ratios = cl_array / np.maximum(cd_array, 1e-6)
        max_lift_drag = np.max(lift_drag_ratios)
        
        # Cl=1.0에서의 Cd (효율성 지표)
        cd_at_cl1 = None
        for i in range(len(cl_array) - 1):
            if (cl_array[i] <= 1.0 <= cl_array[i+1]) or (cl_array[i] >= 1.0 >= cl_array[i+1]):
                # 선형 보간
                t = (1.0 - cl_array[i]) / (cl_array[i+1] - cl_array[i])
                cd_at_cl1 = cd_array[i] + t * (cd_array[i+1] - cd_array[i])
                break
        
        return {
            'max_cl': max_cl,
            'min_cd': min_cd,
            'min_dcl_dcm': min_dcl_dcm,
            'max_lift_drag': max_lift_drag,
            'cd_at_cl1': cd_at_cl1 or min_cd,
            'convergence_rate': len(results['alpha']) / len(alpha_range) if alpha_range else 0
        }
    
    def validate_xfoil_installation(self) -> bool:
        """X-foil 설치 및 실행 가능 여부 확인"""
        try:
            process = subprocess.run(
                [self.xfoil_path],
                input="QUIT\n",
                text=True,
                capture_output=True,
                timeout=10
            )
            return True
        except Exception as e:
            self.logger.error(f"X-foil validation failed: {e}")
            return False
    
    def get_airfoil_info(self, airfoil_coords: np.ndarray) -> Dict:
        """에어포일 기하학적 정보 추출"""
        if len(airfoil_coords) == 0:
            return {}
        
        x_coords = airfoil_coords[:, 0]
        y_coords = airfoil_coords[:, 1]
        
        # 두께 분포 계산
        chord_stations = np.linspace(0, 1, 100)
        thicknesses = []
        
        for x in chord_stations:
            # 각 x 위치에서 상하면 y값 찾기
            y_upper = np.interp(x, x_coords, y_coords, left=0, right=0)
            y_lower = np.interp(x, x_coords, y_coords, left=0, right=0)
            thickness = abs(y_upper - y_lower)
            thicknesses.append(thickness)
        
        max_thickness = np.max(thicknesses)
        max_thickness_loc = chord_stations[np.argmax(thicknesses)]
        
        # 캠버 계산
        camber_line = [(y_coords[i] + y_coords[-(i+1)]) / 2 
                      for i in range(len(y_coords) // 2)]
        max_camber = np.max(np.abs(camber_line)) if camber_line else 0
        
        return {
            'max_thickness': max_thickness,
            'max_thickness_location': max_thickness_loc,
            'max_camber': max_camber,
            'leading_edge_radius': self._estimate_le_radius(airfoil_coords),
            'trailing_edge_thickness': abs(y_coords[0] - y_coords[-1]) if len(y_coords) > 1 else 0
        }
    
    def _estimate_le_radius(self, coords: np.ndarray) -> float:
        """앞전 반지름 추정 (간단한 곡률 계산)"""
        if len(coords) < 5:
            return 0
        
        # 앞전 근처 점들로 곡률 반지름 추정
        le_points = coords[:5]  # 첫 5개 점
        x_le = le_points[:, 0]
        y_le = le_points[:, 1]
        
        # 2차 다항식 피팅
        try:
            coeffs = np.polyfit(x_le, y_le, 2)
            # 곡률 = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
            # x=0에서의 곡률 반지름
            curvature = 2 * abs(coeffs[0])
            radius = 1 / curvature if curvature > 0 else 0
            return min(radius, 0.1)  # 최대값 제한
        except:
            return 0