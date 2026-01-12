"""
NACA Airfoil Database Module

NACA 익형 데이터베이스 스캔 및 초기 설계 선정
Phase 1: Database screening (30분)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class NACADatabase:
    """
    NACA 익형 데이터베이스
    
    다양한 NACA 익형을 스캔하여 목표 조건에 가장 적합한 초기 설계 선정
    """
    
    # 일반적인 NACA 4-digit 익형 목록
    COMMON_NACA = [
        # Symmetric (0% camber)
        "0006", "0008", "0009", "0010", "0012", "0015", "0018", "0021", "0024",
        # Low camber (1-2%)
        "1408", "1410", "1412",
        "2408", "2410", "2412", "2415", "2418", "2421",
        # Medium camber (3-4%)
        "3408", "3412", "3415",
        "4408", "4410", "4412", "4415", "4418", "4421",
        # High camber (5-6%)
        "5412", "5415",
        "6408", "6412", "6415",
    ]
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize database
        
        Parameters
        ----------
        cache_dir : str, optional
            캐시 디렉토리 (분석 결과 저장)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("output/database")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "naca_database.json"
        self.database = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """캐시된 데이터베이스 로드"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """데이터베이스 캐시 저장"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.database, f, indent=2)
    
    @staticmethod
    def parse_naca_code(naca_code: str) -> Dict:
        """
        NACA 4-digit 코드 파싱
        
        Parameters
        ----------
        naca_code : str
            NACA 코드 (예: "2412")
            
        Returns
        -------
        dict
            m (max camber), p (camber position), t (thickness)
        """
        if len(naca_code) != 4:
            raise ValueError(f"Invalid NACA 4-digit code: {naca_code}")
        
        m = int(naca_code[0]) / 100  # Max camber as fraction of chord
        p = int(naca_code[1]) / 10   # Position of max camber
        t = int(naca_code[2:]) / 100 # Thickness as fraction of chord
        
        return {'m': m, 'p': p, 't': t, 'naca': naca_code}
    
    @staticmethod
    def encode_naca_code(m: float, p: float, t: float) -> str:
        """
        NACA 파라미터를 코드로 변환
        
        Parameters
        ----------
        m : float
            Max camber (0-0.1)
        p : float
            Camber position (0-1)
        t : float
            Thickness (0.06-0.24)
            
        Returns
        -------
        str
            NACA 4-digit code
        """
        m_int = int(round(m * 100))
        p_int = int(round(p * 10))
        t_int = int(round(t * 100))
        
        return f"{m_int}{p_int}{t_int:02d}"
    
    def generate_naca_coords(self, m: float, p: float, t: float, 
                             n_points: int = 100) -> np.ndarray:
        """
        NACA 4-digit 익형 좌표 생성
        
        Parameters
        ----------
        m : float
            Max camber (0-0.1)
        p : float
            Camber position (0.1-0.9)
        t : float
            Thickness (0.06-0.24)
        n_points : int
            Number of points per surface
            
        Returns
        -------
        np.ndarray
            Airfoil coordinates (N, 2)
        """
        # Cosine spacing for better resolution at LE/TE
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        
        # Thickness distribution (NACA 4-digit equation)
        yt = 5 * t * (
            0.2969 * np.sqrt(x) -
            0.1260 * x -
            0.3516 * x**2 +
            0.2843 * x**3 -
            0.1015 * x**4  # Closed TE: -0.1036
        )
        
        # Camber line
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        
        if m > 0 and p > 0:
            # Avoid division by zero
            p_safe = max(p, 0.01)
            
            # Forward of max camber
            mask_forward = x < p_safe
            yc[mask_forward] = (m / p_safe**2) * (2*p_safe*x[mask_forward] - x[mask_forward]**2)
            dyc_dx[mask_forward] = (2*m / p_safe**2) * (p_safe - x[mask_forward])
            
            # Aft of max camber
            mask_aft = ~mask_forward
            yc[mask_aft] = (m / (1-p_safe)**2) * ((1 - 2*p_safe) + 2*p_safe*x[mask_aft] - x[mask_aft]**2)
            dyc_dx[mask_aft] = (2*m / (1-p_safe)**2) * (p_safe - x[mask_aft])
        
        # Angle
        theta = np.arctan(dyc_dx)
        
        # Upper and lower surfaces
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        
        # Combine (upper from TE to LE, lower from LE to TE)
        coords = np.vstack([
            np.column_stack([xu[::-1], yu[::-1]]),
            np.column_stack([xl[1:], yl[1:]])
        ])
        
        return coords
    
    def analyze_single(self, naca_code: str, reynolds: float, aoa: float,
                       mach: float = 0.0) -> Optional[Dict]:
        """
        단일 NACA 익형 분석
        
        Parameters
        ----------
        naca_code : str
            NACA 4-digit code
        reynolds : float
            Reynolds number
        aoa : float
            Angle of attack (degrees)
        mach : float
            Mach number
            
        Returns
        -------
        dict or None
            Analysis results
        """
        from ..optimize.xfoil_interface import run_xfoil_analysis
        
        params = self.parse_naca_code(naca_code)
        coords = self.generate_naca_coords(params['m'], params['p'], params['t'])
        
        result = run_xfoil_analysis(
            coords,
            reynolds=reynolds,
            aoa=aoa,
            mach=mach,
            n_iter=200
        )
        
        if result:
            result['naca'] = naca_code
            result['m'] = params['m']
            result['p'] = params['p']
            result['t'] = params['t']
            
        return result
    
    def scan_database(self, reynolds: float, aoa: float, mach: float = 0.0,
                      naca_list: Optional[List[str]] = None,
                      target_cl: Optional[float] = None,
                      n_workers: int = 4,
                      verbose: bool = True) -> List[Dict]:
        """
        NACA 데이터베이스 스캔
        
        Parameters
        ----------
        reynolds : float
            Reynolds number
        aoa : float
            Angle of attack (degrees)
        mach : float
            Mach number
        naca_list : list, optional
            스캔할 NACA 코드 목록 (None이면 기본 목록 사용)
        target_cl : float, optional
            목표 CL (이 값에 가까운 익형 우선)
        n_workers : int
            병렬 처리 워커 수
        verbose : bool
            상세 출력 여부
            
        Returns
        -------
        list
            L/D 기준 정렬된 분석 결과 목록
        """
        if naca_list is None:
            naca_list = self.COMMON_NACA
        
        if verbose:
            print(f"\n📊 NACA Database Scan")
            print(f"   Reynolds: {reynolds:.2e}")
            print(f"   AoA: {aoa}°")
            print(f"   Mach: {mach}")
            print(f"   Scanning {len(naca_list)} airfoils...")
        
        results = []
        cache_key = f"Re{reynolds:.0e}_a{aoa}_M{mach}"
        
        # Check cache first
        for naca in naca_list:
            key = f"{naca}_{cache_key}"
            if key in self.database:
                results.append(self.database[key])
        
        # Analyze missing airfoils
        missing = [n for n in naca_list if f"{n}_{cache_key}" not in self.database]
        
        if missing:
            if verbose:
                print(f"   Analyzing {len(missing)} new airfoils...")
            
            from ..optimize.xfoil_interface import run_xfoil_analysis
            
            for i, naca in enumerate(missing):
                if verbose and (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{len(missing)}")
                
                result = self.analyze_single(naca, reynolds, aoa, mach)
                
                if result and 'L/D' in result:
                    key = f"{naca}_{cache_key}"
                    self.database[key] = result
                    results.append(result)
            
            # Save cache
            self._save_cache()
        
        # Sort by L/D (descending)
        results.sort(key=lambda x: x.get('L/D', 0), reverse=True)
        
        # If target CL specified, filter and re-sort
        if target_cl is not None:
            # Filter airfoils with CL close to target
            results = [r for r in results if abs(r.get('CL', 0) - target_cl) < 0.3]
            # Re-sort by L/D
            results.sort(key=lambda x: x.get('L/D', 0), reverse=True)
        
        if verbose:
            print(f"\n✓ Database scan complete")
            if results:
                print(f"\n📋 Top 5 Airfoils by L/D:")
                print("-" * 60)
                for i, r in enumerate(results[:5]):
                    print(f"   {i+1}. NACA {r['naca']}: "
                          f"L/D={r.get('L/D', 0):.1f}, "
                          f"CL={r.get('CL', 0):.3f}, "
                          f"CD={r.get('CD', 0):.5f}")
        
        return results
    
    def recommend_initial(self, reynolds: float, aoa: float, mach: float = 0.0,
                          target_cl: Optional[float] = None,
                          min_thickness: float = 0.10) -> Dict:
        """
        초기 설계 추천
        
        Parameters
        ----------
        reynolds : float
            Reynolds number
        aoa : float
            Angle of attack (degrees)
        mach : float
            Mach number
        target_cl : float, optional
            목표 CL
        min_thickness : float
            최소 두께 비율 (구조 요구사항)
            
        Returns
        -------
        dict
            추천 익형 정보
        """
        results = self.scan_database(
            reynolds, aoa, mach, 
            target_cl=target_cl,
            verbose=False
        )
        
        # Filter by minimum thickness
        results = [r for r in results if r.get('t', 0) >= min_thickness]
        
        if not results:
            # Fallback to NACA 2412
            return {
                'naca': '2412',
                'm': 0.02,
                'p': 0.4,
                't': 0.12,
                'L/D': None,
                'reason': 'No suitable airfoil found, using default NACA 2412'
            }
        
        best = results[0]
        best['reason'] = f"Best L/D ({best.get('L/D', 0):.1f}) from database scan"
        
        return best


class ExtendedNACADatabase(NACADatabase):
    """
    확장된 NACA 데이터베이스
    
    더 많은 익형과 다양한 조건에서의 분석 지원
    """
    
    def generate_naca_range(self, 
                            m_range: Tuple[int, int] = (0, 6),
                            p_range: Tuple[int, int] = (2, 5),
                            t_range: Tuple[int, int] = (9, 18),
                            t_step: int = 3) -> List[str]:
        """
        NACA 코드 범위 생성
        
        Parameters
        ----------
        m_range : tuple
            Camber range (0-9)
        p_range : tuple
            Camber position range (1-9)
        t_range : tuple
            Thickness range (6-24)
        t_step : int
            Thickness step
            
        Returns
        -------
        list
            NACA code list
        """
        codes = []
        
        for m in range(m_range[0], m_range[1] + 1):
            for p in range(p_range[0], p_range[1] + 1):
                if m == 0 and p != 0:
                    continue  # Symmetric airfoils only have p=0
                if m > 0 and p == 0:
                    p = 4  # Default position for cambered
                    
                for t in range(t_range[0], t_range[1] + 1, t_step):
                    code = f"{m}{p}{t:02d}"
                    codes.append(code)
        
        return codes
    
    def polar_analysis(self, naca_code: str, reynolds: float,
                       aoa_range: Tuple[float, float] = (-4, 12),
                       aoa_step: float = 0.5,
                       mach: float = 0.0) -> Dict:
        """
        Polar 분석 (여러 받음각에서 분석)
        
        Parameters
        ----------
        naca_code : str
            NACA code
        reynolds : float
            Reynolds number
        aoa_range : tuple
            AoA range (min, max) in degrees
        aoa_step : float
            AoA step in degrees
        mach : float
            Mach number
            
        Returns
        -------
        dict
            Polar data with arrays of AoA, CL, CD, CM
        """
        from ..optimize.xfoil_interface import run_polar_analysis
        
        params = self.parse_naca_code(naca_code)
        coords = self.generate_naca_coords(params['m'], params['p'], params['t'])
        
        aoa_list = np.arange(aoa_range[0], aoa_range[1] + aoa_step, aoa_step)
        
        return run_polar_analysis(coords, reynolds, aoa_list, mach)
