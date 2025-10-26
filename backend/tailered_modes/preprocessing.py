"""
UIUC Airfoil Database Preprocessing
에어포일 데이터를 정규화하고 학습 가능한 형태로 변환
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from scipy import interpolate

logger = logging.getLogger(__name__)


class AirfoilPreprocessor:
    """UIUC airfoil 데이터베이스 전처리"""
    
    def __init__(self, n_points: int = 251):
        """
        Args:
            n_points: 각 에어포일의 표면점 개수 (논문: 251)
        """
        self.n_points = n_points
        self.x_template = self.create_x_template()
        
    def create_x_template(self) -> np.ndarray:
        """
        코사인 분포를 사용한 x 좌표 템플릿 생성
        앞전과 뒷전에 더 많은 점을 배치
        """
        beta = np.linspace(0, np.pi, self.n_points)
        x = 0.5 * (1 - np.cos(beta))  # [0, 1] 범위
        return x
    
    def load_airfoil_from_dat(self, filepath: Path) -> Optional[np.ndarray]:
        """
        .dat 파일에서 에어포일 좌표 로드
        
        Returns:
            np.ndarray: shape (n, 2) [x, y] 좌표
        """
        try:
            # 파일 읽기
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # 헤더 제거 (첫 줄은 보통 이름)
            data_lines = [l.strip() for l in lines[1:] if l.strip()]
            
            # 좌표 파싱
            coords = []
            for line in data_lines:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        coords.append([x, y])
                except ValueError:
                    continue
            
            if len(coords) < 10:
                logger.warning(f"Too few points in {filepath.name}")
                return None
                
            return np.array(coords)
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def split_upper_lower(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        에어포일을 상면(upper)과 하면(lower)으로 분리
        
        Args:
            coords: shape (n, 2) 에어포일 좌표
            
        Returns:
            upper, lower: 각각 상면/하면 좌표
        """
        # 앞전 찾기 (x가 최소인 점)
        le_idx = np.argmin(coords[:, 0])
        
        # 데이터 형식 확인 (시계방향 vs 반시계방향)
        if le_idx == 0:
            # 앞전이 시작점: 0 -> TE_upper -> LE -> TE_lower
            # 중간 지점 찾기 (y가 최대 또는 x가 최대)
            mid_idx = len(coords) // 2
            upper = coords[:mid_idx+1]
            lower = coords[mid_idx:]
        else:
            # 일반적인 경우: TE_upper -> LE -> TE_lower
            upper = coords[:le_idx+1]
            lower = coords[le_idx:]
        
        # x 좌표 기준 정렬
        upper = upper[np.argsort(upper[:, 0])]
        lower = lower[np.argsort(lower[:, 0])]
        
        return upper, lower
    
    def interpolate_surface(self, surface: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """
        주어진 x 템플릿에 맞춰 y 좌표를 보간
        
        Args:
            surface: shape (n, 2) [x, y]
            x_new: 새로운 x 좌표 템플릿
            
        Returns:
            y_new: 보간된 y 좌표
        """
        # 중복 x 제거
        unique_indices = np.unique(surface[:, 0], return_index=True)[1]
        surface_unique = surface[unique_indices]
        
        # 1D 보간
        f = interpolate.interp1d(
            surface_unique[:, 0], 
            surface_unique[:, 1],
            kind='cubic',
            fill_value='extrapolate'
        )
        
        return f(x_new)
    
    def normalize_airfoil(self, coords: np.ndarray) -> np.ndarray:
        """
        에어포일을 정규화:
        1. Chord length = 1
        2. 앞전을 (0, 0)으로
        3. x 템플릿에 맞춰 재샘플링
        
        Args:
            coords: shape (n, 2) 원본 좌표
            
        Returns:
            normalized: shape (n_points, 2) 정규화된 좌표
        """
        # Chord length 정규화
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        chord = x_max - x_min
        
        coords_norm = coords.copy()
        coords_norm[:, 0] = (coords[:, 0] - x_min) / chord
        coords_norm[:, 1] = coords[:, 1] / chord
        
        # 상/하면 분리
        upper, lower = self.split_upper_lower(coords_norm)
        
        # x 템플릿에 맞춰 보간
        y_upper = self.interpolate_surface(upper, self.x_template)
        y_lower = self.interpolate_surface(lower, self.x_template)
        
        # 결합: upper (reversed) + lower
        # 논문에서는 TE(upper) -> LE -> TE(lower) 순서
        y_upper_rev = y_upper[::-1]
        y_combined = np.concatenate([y_upper_rev, y_lower])
        x_combined = np.concatenate([self.x_template[::-1], self.x_template])
        
        normalized = np.column_stack([x_combined, y_combined])
        
        return normalized
    
    def validate_airfoil(self, coords: np.ndarray) -> bool:
        """
        기본적인 에어포일 유효성 검증
        
        Args:
            coords: shape (n, 2) 좌표
            
        Returns:
            유효하면 True
        """
        if coords is None or len(coords) < 10:
            return False
        
        # Chord length 확인
        x_range = coords[:, 0].max() - coords[:, 0].min()
        if x_range < 0.5 or x_range > 2.0:
            return False
        
        # 두께 확인 (너무 얇거나 두꺼운 경우)
        y_range = coords[:, 1].max() - coords[:, 1].min()
        thickness_ratio = y_range / x_range
        if thickness_ratio < 0.01 or thickness_ratio > 0.5:
            return False
        
        # Self-intersection 간단 체크
        # (완벽하지 않지만 기본적인 검증)
        x = coords[:, 0]
        if len(np.unique(x)) < len(x) * 0.8:
            return False
        
        return True
    
    def normalize_to_tanh_range(self, coords: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] for GAN training"""
        coords_normalized = 2 * (coords - coords.min()) / (coords.max() - coords.min()) - 1
        return coords_normalized
    
    def process_database(self, 
                        data_dir: Path,
                        output_path: Optional[Path] = None) -> np.ndarray:
        """
        전체 UIUC database 처리
        
        Args:
            data_dir: UIUC .dat 파일들이 있는 디렉토리
            output_path: 처리된 데이터를 저장할 경로 (optional)
            
        Returns:
            processed_airfoils: shape (m, n_points*2) 
                              m개 에어포일, 각각 x,y 좌표 flatten
        """
        logger.info(f"Processing UIUC database from {data_dir}")
        
        # .dat 파일 찾기
        dat_files = list(data_dir.glob("*.dat"))
        logger.info(f"Found {len(dat_files)} .dat files")
        
        processed = []
        failed = []
        
        for dat_file in dat_files:
            # 로드
            coords = self.load_airfoil_from_dat(dat_file)
            
            if coords is None:
                failed.append(dat_file.name)
                continue
            
            # 유효성 검증
            if not self.validate_airfoil(coords):
                failed.append(dat_file.name)
                continue
            
            try:
                # 정규화
                normalized = self.normalize_airfoil(coords)
                
                # Flatten (x, y를 일렬로)
                # shape: (n_points*2,)
                flattened = normalized.flatten()
                
                processed.append(flattened)
                
            except Exception as e:
                logger.warning(f"Failed to process {dat_file.name}: {e}")
                failed.append(dat_file.name)
        
        logger.info(f"Successfully processed: {len(processed)}/{len(dat_files)}")
        logger.info(f"Failed: {len(failed)}")
        
        if len(processed) == 0:
            raise ValueError("No airfoils were successfully processed!")
        
        # numpy array로 변환
        processed_array = np.array(processed)
        
        # 저장
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, processed_array)
            logger.info(f"Saved processed data to {output_path}")
        
        return processed_array
    
    @staticmethod
    def unflatten_airfoil(flattened: np.ndarray, n_points: int = 251) -> np.ndarray:
        """
        Flatten된 데이터를 (n, 2) 형태로 복원
        
        Args:
            flattened: shape (n_points*2,)
            
        Returns:
            coords: shape (n_points*2, 2) [x, y]
        """
        return flattened.reshape(-1, 2)


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = AirfoilPreprocessor(n_points=251)
    
    # 예제: 단일 파일 처리
    # test_file = Path("data/uiuc_airfoils/naca0012.dat")
    # coords = preprocessor.load_airfoil_from_dat(test_file)
    # normalized = preprocessor.normalize_airfoil(coords)
    # print(f"Normalized shape: {normalized.shape}")
    
    print("Preprocessor module loaded successfully")
    print(f"X template shape: {preprocessor.x_template.shape}")
    