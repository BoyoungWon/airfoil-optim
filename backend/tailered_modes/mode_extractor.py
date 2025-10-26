"""
SVD-based Mode Shape Extraction
Extracts modal basis from airfoil samples using Singular Value Decomposition
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModeExtractor:
    """
    SVD를 사용하여 에어포일 데이터베이스로부터 mode shapes 추출
    """
    
    def __init__(self, n_points: int = 251):
        """
        Args:
            n_points: 각 에어포일의 표면점 개수
        """
        self.n_points = n_points
        self.airfoil_dim = n_points * 2  # full airfoil: upper + lower, x + y
        
        # Mode shapes (will be computed)
        self.modes = None  # U matrix from SVD
        self.singular_values = None  # Singular values
        self.mean_airfoil = None  # Mean airfoil shape
        self.mode_coefficients_bounds = None  # Bounds for optimization
    
    def extract_modes(self,
                     airfoils: np.ndarray,
                     n_modes: Optional[int] = None) -> np.ndarray:
        """
        SVD를 사용하여 mode shapes 추출
        
        Args:
            airfoils: shape (m, airfoil_dim) m개 에어포일
            n_modes: 추출할 mode 개수 (None이면 전체)
            
        Returns:
            modes: shape (airfoil_dim, n_modes) mode shapes
        """
        logger.info(f"Extracting modes from {len(airfoils)} airfoils")
        
        # Mean airfoil 계산
        self.mean_airfoil = np.mean(airfoils, axis=0)
        
        # Mean-centered data matrix
        # A[i, j] = y_j^(i) - mean_y_j
        A = airfoils - self.mean_airfoil
        
        # SVD: A = U * Σ * V^T
        # U의 columns이 mode shapes
        U, S, Vt = np.linalg.svd(A.T, full_matrices=False)
        
        self.singular_values = S
        
        # Mode shapes (U matrix의 columns)
        if n_modes is None:
            n_modes = len(S)
        
        self.modes = U[:, :n_modes]
        
        logger.info(f"Extracted {n_modes} modes")
        logger.info(f"Explained variance: {self._explained_variance(n_modes):.2%}")
        
        # Mode coefficient bounds 계산
        self._compute_coefficient_bounds(airfoils, n_modes)
        
        return self.modes
    
    def _explained_variance(self, n_modes: int) -> float:
        """
        선택한 modes가 설명하는 variance 비율
        """
        if self.singular_values is None:
            return 0.0
        
        total_variance = np.sum(self.singular_values ** 2)
        explained_variance = np.sum(self.singular_values[:n_modes] ** 2)
        
        return explained_variance / total_variance
    
    def _compute_coefficient_bounds(self,
                                   airfoils: np.ndarray,
                                   n_modes: int):
        """
        Mode coefficients의 bounds 계산
        논문: percentiles of 0.1% and 99.9%
        """
        # Project airfoils onto modes
        A = airfoils - self.mean_airfoil
        coefficients = A @ self.modes
        
        # Percentile bounds (0.1%, 99.9%)
        lower_bounds = np.percentile(coefficients, 0.1, axis=0)
        upper_bounds = np.percentile(coefficients, 99.9, axis=0)
        
        self.mode_coefficients_bounds = {
            'lower': lower_bounds,
            'upper': upper_bounds
        }
        
        logger.info(f"Coefficient bounds computed for {n_modes} modes")
    
    def reconstruct_airfoil(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Mode coefficients로부터 에어포일 재구성
        
        y = mean + Σ(c_i * mode_i)
        
        Args:
            coefficients: shape (n_modes,) mode coefficients
            
        Returns:
            airfoil: shape (airfoil_dim,) 재구성된 에어포일
        """
        if self.modes is None or self.mean_airfoil is None:
            raise ValueError("Modes not extracted yet!")
        
        # y = mean + U * c
        airfoil = self.mean_airfoil + self.modes @ coefficients
        
        return airfoil
    
    def project_airfoil(self, airfoil: np.ndarray) -> np.ndarray:
        """
        에어포일을 mode space로 projection
        
        c = U^T * (y - mean)
        
        Args:
            airfoil: shape (airfoil_dim,)
            
        Returns:
            coefficients: shape (n_modes,)
        """
        if self.modes is None or self.mean_airfoil is None:
            raise ValueError("Modes not extracted yet!")
        
        # c = U^T * (y - mean)
        centered = airfoil - self.mean_airfoil
        coefficients = self.modes.T @ centered
        
        return coefficients
    
    def get_coefficient_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mode coefficients의 bounds 반환
        
        Returns:
            lower_bounds, upper_bounds: shape (n_modes,)
        """
        if self.mode_coefficients_bounds is None:
            raise ValueError("Bounds not computed yet!")
        
        return (
            self.mode_coefficients_bounds['lower'],
            self.mode_coefficients_bounds['upper']
        )
    
    def visualize_modes(self,
                       n_modes_to_show: int = 6,
                       save_path: Optional[Path] = None):
        """
        Mode shapes 시각화
        
        Args:
            n_modes_to_show: 표시할 mode 개수
            save_path: 저장 경로
        """
        if self.modes is None:
            raise ValueError("Modes not extracted yet!")
        
        n_modes = min(n_modes_to_show, self.modes.shape[1])
        
        fig, axes = plt.subplots(
            (n_modes + 2) // 3, 3,
            figsize=(15, 5 * ((n_modes + 2) // 3))
        )
        axes = axes.flatten() if n_modes > 1 else [axes]
        
        # Mean airfoil
        mean_coords = self.mean_airfoil.reshape(-1, 2)
        
        for i in range(n_modes):
            ax = axes[i]
            
            # Mode shape 효과 시각화
            # mean + alpha * mode (alpha = ±std)
            mode_effect = self.modes[:, i]
            std = np.std(mode_effect)
            
            airfoil_plus = self.mean_airfoil + 3 * std * mode_effect
            airfoil_minus = self.mean_airfoil - 3 * std * mode_effect
            
            coords_plus = airfoil_plus.reshape(-1, 2)
            coords_minus = airfoil_minus.reshape(-1, 2)
            
            # Plot
            ax.plot(mean_coords[:, 0], mean_coords[:, 1],
                   'k-', linewidth=2, label='Mean', alpha=0.5)
            ax.plot(coords_plus[:, 0], coords_plus[:, 1],
                   'b-', linewidth=1.5, label=f'+3σ')
            ax.plot(coords_minus[:, 0], coords_minus[:, 1],
                   'r-', linewidth=1.5, label=f'-3σ')
            
            ax.set_title(f'Mode {i+1}')
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_modes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Mode visualization saved to {save_path}")
        
        plt.show()
    
    def save_modes(self, save_path: Path):
        """
        Mode shapes와 관련 데이터 저장
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'modes': self.modes,
            'singular_values': self.singular_values,
            'mean_airfoil': self.mean_airfoil,
            'coefficient_bounds': self.mode_coefficients_bounds,
            'n_points': self.n_points
        }
        
        np.savez(save_path, **data)
        logger.info(f"Modes saved to {save_path}")
    
    def load_modes(self, load_path: Path):
        """
        저장된 mode shapes 로드
        """
        data = np.load(load_path, allow_pickle=True)
        
        self.modes = data['modes']
        self.singular_values = data['singular_values']
        self.mean_airfoil = data['mean_airfoil']
        self.mode_coefficients_bounds = data['coefficient_bounds'].item()
        self.n_points = int(data['n_points'])
        
        logger.info(f"Modes loaded from {load_path}")
        logger.info(f"Number of modes: {self.modes.shape[1]}")


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 더미 데이터
    n_samples = 500
    airfoil_dim = 502
    dummy_airfoils = np.random.randn(n_samples, airfoil_dim)
    
    # Mode extractor
    extractor = ModeExtractor(n_points=251)
    
    # Modes 추출
    modes = extractor.extract_modes(dummy_airfoils, n_modes=15)
    
    print(f"Extracted modes shape: {modes.shape}")
    print(f"Explained variance: {extractor._explained_variance(15):.2%}")
    
    # Coefficient bounds
    lower, upper = extractor.get_coefficient_bounds()
    print(f"\nCoefficient bounds:")
    print(f"  Lower: {lower[:5]}")
    print(f"  Upper: {upper[:5]}")
    
    # Reconstruction test
    test_coeffs = np.random.randn(15) * 0.1
    reconstructed = extractor.reconstruct_airfoil(test_coeffs)
    print(f"\nReconstructed airfoil shape: {reconstructed.shape}")