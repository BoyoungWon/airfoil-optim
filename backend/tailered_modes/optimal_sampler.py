"""
Optimal Airfoil Sampler
Optimizes GAN-generated airfoils to satisfy geometric constraints
while maintaining validity
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import logging
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeometricConstraints:
    """에어포일 기하학적 제약조건"""
    max_thickness: Optional[float] = None  # 최대 두께 (chord 기준)
    min_thickness: Optional[float] = None
    thickness_at_positions: Optional[Dict[float, Tuple[float, float]]] = None  # {x_pos: (min, max)}
    min_area: Optional[float] = None  # 최소 면적
    max_area: Optional[float] = None


class OptimalSampler:
    """
    GAN 생성 에어포일을 제약조건을 만족하도록 최적화
    논문: minimize displacement while satisfying constraints + validity
    """
    
    def __init__(self,
                 geometric_validator: Optional[object] = None,
                 n_points: int = 251):
        """
        Args:
            geometric_validator: GeometricValidator 인스턴스
            n_points: 에어포일 표면점 개수
        """
        self.validator = geometric_validator
        self.n_points = n_points
        self.airfoil_dim = n_points * 4  # upper + lower surfaces, x and y
    
    def objective_function(self,
                          airfoil: np.ndarray,
                          target: np.ndarray) -> float:
        """
        목적함수: GAN 생성 에어포일과의 거리 최소화
        
        Args:
            airfoil: 최적화 중인 에어포일
            target: 원본 GAN 생성 에어포일
            
        Returns:
            distance: L2 distance
        """
        return np.linalg.norm(airfoil - target)
    
    def compute_thickness(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        에어포일 두께 분포 계산
        
        Args:
            coords: shape (n_points*2, 2) [x, y] 좌표
            
        Returns:
            max_thickness: 최대 두께
            thickness_dist: 두께 분포
        """
        n = len(coords) // 2
        upper = coords[:n]
        lower = coords[n:]
        
        # x 좌표 정렬
        upper_sorted = upper[np.argsort(upper[:, 0])]
        lower_sorted = lower[np.argsort(lower[:, 0])]
        
        # 같은 x 위치에서의 두께
        x_common = np.linspace(0, 1, 100)
        y_upper = np.interp(x_common, upper_sorted[:, 0], upper_sorted[:, 1])
        y_lower = np.interp(x_common, lower_sorted[:, 0], lower_sorted[:, 1])
        
        thickness_dist = np.abs(y_upper - y_lower)
        max_thickness = np.max(thickness_dist)
        
        return max_thickness, thickness_dist
    
    def compute_area(self, coords: np.ndarray) -> float:
        """
        에어포일 단면적 계산 (Shoelace formula)
        
        Args:
            coords: shape (n_points*2, 2)
            
        Returns:
            area: 단면적
        """
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Shoelace formula
        area = 0.5 * np.abs(
            np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]) +
            x[-1] * y[0] - x[0] * y[-1]
        )
        
        return area
    
    def thickness_at_position(self,
                            coords: np.ndarray,
                            x_pos: float) -> float:
        """
        특정 x 위치에서의 두께
        
        Args:
            coords: shape (n_points*2, 2)
            x_pos: x 위치 (0~1)
            
        Returns:
            thickness: 해당 위치 두께
        """
        n = len(coords) // 2
        upper = coords[:n]
        lower = coords[n:]
        
        # x 기준 정렬
        upper_sorted = upper[np.argsort(upper[:, 0])]
        lower_sorted = lower[np.argsort(lower[:, 0])]
        
        # 보간
        y_upper = np.interp(x_pos, upper_sorted[:, 0], upper_sorted[:, 1])
        y_lower = np.interp(x_pos, lower_sorted[:, 0], lower_sorted[:, 1])
        
        thickness = abs(y_upper - y_lower)
        
        return thickness
    
    def create_constraint_functions(self,
                                   constraints: GeometricConstraints
                                   ) -> List[NonlinearConstraint]:
        """
        제약조건 함수 생성
        
        Args:
            constraints: GeometricConstraints 객체
            
        Returns:
            constraint_list: scipy.optimize용 제약조건 리스트
        """
        constraint_list = []
        
        def reshape_coords(x):
            return x.reshape(-1, 2)
        
        # 최대 두께 제약
        if constraints.max_thickness is not None:
            def max_thickness_constraint(x):
                coords = reshape_coords(x)
                max_t, _ = self.compute_thickness(coords)
                return constraints.max_thickness - max_t  # >= 0
            
            constraint_list.append(
                NonlinearConstraint(max_thickness_constraint, 0, np.inf)
            )
        
        # 최소 두께 제약
        if constraints.min_thickness is not None:
            def min_thickness_constraint(x):
                coords = reshape_coords(x)
                max_t, _ = self.compute_thickness(coords)
                return max_t - constraints.min_thickness  # >= 0
            
            constraint_list.append(
                NonlinearConstraint(min_thickness_constraint, 0, np.inf)
            )
        
        # 특정 위치 두께 제약
        if constraints.thickness_at_positions:
            for x_pos, (t_min, t_max) in constraints.thickness_at_positions.items():
                def pos_thickness_constraint(x, pos=x_pos, tmin=t_min, tmax=t_max):
                    coords = reshape_coords(x)
                    t = self.thickness_at_position(coords, pos)
                    # tmin <= t <= tmax
                    return [t - tmin, tmax - t]  # both >= 0
                
                constraint_list.append(
                    NonlinearConstraint(pos_thickness_constraint, 0, np.inf)
                )
        
        # 면적 제약
        if constraints.min_area is not None:
            def min_area_constraint(x):
                coords = reshape_coords(x)
                area = self.compute_area(coords)
                return area - constraints.min_area  # >= 0
            
            constraint_list.append(
                NonlinearConstraint(min_area_constraint, 0, np.inf)
            )
        
        if constraints.max_area is not None:
            def max_area_constraint(x):
                coords = reshape_coords(x)
                area = self.compute_area(coords)
                return constraints.max_area - area  # >= 0
            
            constraint_list.append(
                NonlinearConstraint(max_area_constraint, 0, np.inf)
            )
        
        # Validity 제약 (if validator provided)
        if self.validator is not None:
            def validity_constraint(x):
                score = self.validator.predict_validity(x.reshape(1, -1))[0]
                return score - 0.75  # >= 0 (threshold 0.75)
            
            constraint_list.append(
                NonlinearConstraint(validity_constraint, 0, np.inf)
            )
        
        logger.info(f"Created {len(constraint_list)} constraints")
        
        return constraint_list
    
    def optimize_sample(self,
                       gan_airfoil: np.ndarray,
                       constraints: GeometricConstraints,
                       max_iter: int = 200) -> Tuple[np.ndarray, bool]:
        """
        GAN 생성 에어포일을 제약조건 만족하도록 최적화
        
        Args:
            gan_airfoil: shape (airfoil_dim,) GAN 생성 에어포일
            constraints: 기하학적 제약조건
            max_iter: 최대 반복 횟수
            
        Returns:
            optimized_airfoil: 최적화된 에어포일
            success: 최적화 성공 여부
        """
        # 제약조건 생성
        constraint_list = self.create_constraint_functions(constraints)
        
        # 초기값
        x0 = gan_airfoil.copy()
        
        # 목적함수
        def objective(x):
            return self.objective_function(x, gan_airfoil)
        
        # 최적화 (SLSQP)
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraint_list,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        
        if result.success:
            logger.debug(f"Optimization successful: {result.message}")
        else:
            logger.warning(f"Optimization failed: {result.message}")
        
        return result.x, result.success
    
    def generate_samples(self,
                        gan_generator: Callable,
                        constraints: GeometricConstraints,
                        n_samples: int = 500,
                        max_attempts: int = 1000) -> np.ndarray:
        """
        제약조건을 만족하는 샘플 생성
        
        Args:
            gan_generator: GAN generator function (n_samples) -> airfoils
            constraints: 기하학적 제약조건
            n_samples: 목표 샘플 수
            max_attempts: 최대 시도 횟수
            
        Returns:
            samples: shape (n_samples, airfoil_dim) 최적화된 샘플들
        """
        logger.info(f"Generating {n_samples} constrained samples")
        
        samples = []
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            # GAN으로 에어포일 생성
            gan_airfoils = gan_generator(n_samples=10)
            
            # 각각 최적화
            for gan_airfoil in gan_airfoils:
                if len(samples) >= n_samples:
                    break
                
                optimized, success = self.optimize_sample(
                    gan_airfoil, constraints
                )
                
                if success:
                    samples.append(optimized)
                    
                    if len(samples) % 50 == 0:
                        logger.info(f"Generated {len(samples)}/{n_samples} samples")
                
                attempts += 1
        
        if len(samples) < n_samples:
            logger.warning(
                f"Only generated {len(samples)}/{n_samples} samples "
                f"after {attempts} attempts"
            )
        
        return np.array(samples)


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # Sampler 생성
    sampler = OptimalSampler(n_points=251)
    
    # 더미 에어포일
    dummy_airfoil = np.random.randn(502)
    
    # 제약조건
    constraints = GeometricConstraints(
        max_thickness=0.15,
        min_thickness=0.08,
        thickness_at_positions={0.75: (0.07, 0.20)},
        min_area=0.09
    )
    
    print("Optimal Sampler created successfully")
    print(f"Constraints: {constraints}")
    
    # 단일 샘플 최적화 테스트
    optimized, success = sampler.optimize_sample(dummy_airfoil, constraints, max_iter=10)
    print(f"\nOptimization success: {success}")
    print(f"Optimized shape: {optimized.shape}")