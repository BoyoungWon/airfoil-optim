"""
Shape Parametrization Module

다양한 airfoil 형상 매개변수화 방법:
- NACA: 4-digit NACA series (4 parameters)
- CST: Class/Shape Transformation (8-30 parameters)
- FFD: Free Form Deformation (15-100+ parameters)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from pathlib import Path


class BaseParametrization(ABC):
    """Base class for airfoil parametrization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_params = None
        self.bounds = None
    
    @abstractmethod
    def generate_airfoil(self, params: np.ndarray) -> np.ndarray:
        """Generate airfoil coordinates from parameters"""
        pass
    
    @abstractmethod
    def get_bounds(self) -> list:
        """Get parameter bounds for optimization"""
        pass
    
    @abstractmethod
    def get_initial(self) -> np.ndarray:
        """Get initial parameters"""
        pass
    
    def scale_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Scale unit hypercube samples to parameter bounds
        
        Parameters:
        -----------
        samples : np.ndarray
            Samples in [0, 1]^n
        
        Returns:
        --------
        np.ndarray
            Scaled samples in parameter space
        """
        bounds = self.get_bounds()
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        return lower + samples * (upper - lower)


class NACAParametrization(BaseParametrization):
    """
    NACA 4-digit airfoil parametrization
    
    Parameters:
    - m: Maximum camber (0-0.1)
    - p: Location of maximum camber (0.1-0.9)
    - t: Thickness (0.06-0.21)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.n_params = 3
    
    def generate_airfoil(self, params: np.ndarray) -> np.ndarray:
        """
        Generate NACA 4-digit airfoil
        
        Parameters:
        -----------
        params : np.ndarray
            [m, p, t]
        
        Returns:
        --------
        np.ndarray
            Airfoil coordinates (N, 2)
        """
        m, p, t = params
        
        # Cosine spacing for better resolution at LE/TE
        beta = np.linspace(0, np.pi, 100)
        x = 0.5 * (1 - np.cos(beta))
        
        # Thickness distribution
        yt = 5 * t * (
            0.2969 * np.sqrt(x) -
            0.1260 * x -
            0.3516 * x**2 +
            0.2843 * x**3 -
            0.1015 * x**4
        )
        
        # Camber line
        yc = np.zeros_like(x)
        if m > 0:
            yc = np.where(
                x < p,
                m * x / p**2 * (2*p - x),
                m * (1-x) / (1-p)**2 * (1 + x - 2*p)
            )
        
        # Camber line slope
        dyc_dx = np.zeros_like(x)
        if m > 0:
            dyc_dx = np.where(
                x < p,
                2*m / p**2 * (p - x),
                2*m / (1-p)**2 * (p - x)
            )
        
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
    
    def get_bounds(self) -> list:
        """Parameter bounds: [(m_min, m_max), (p_min, p_max), (t_min, t_max)]"""
        return [
            (0.0, 0.1),   # m: camber
            (0.1, 0.9),   # p: location of max camber
            (0.06, 0.21)  # t: thickness
        ]
    
    def get_initial(self) -> np.ndarray:
        """Initial guess (NACA 2412)"""
        return np.array([0.02, 0.4, 0.12])


class CSTParametrization(BaseParametrization):
    """
    CST (Class/Shape Transformation) parametrization
    
    More flexible than NACA, typically uses 8-30 parameters
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.n_upper = config.get('n_upper', 6)
        self.n_lower = config.get('n_lower', 6)
        self.n_params = self.n_upper + self.n_lower + 1  # +1 for TE thickness
    
    def class_function(self, x: np.ndarray, N1=0.5, N2=1.0) -> np.ndarray:
        """CST class function"""
        return x**N1 * (1 - x)**N2
    
    def shape_function(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Bernstein polynomial shape function"""
        n = len(weights) - 1
        S = np.zeros_like(x)
        
        for i, w in enumerate(weights):
            # Binomial coefficient
            binom = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
            S += w * binom * x**i * (1 - x)**(n - i)
        
        return S
    
    def generate_airfoil(self, params: np.ndarray) -> np.ndarray:
        """
        Generate CST airfoil
        
        Parameters:
        -----------
        params : np.ndarray
            [w_upper_1, ..., w_upper_n, w_lower_1, ..., w_lower_n, dz_te]
        
        Returns:
        --------
        np.ndarray
            Airfoil coordinates
        """
        w_upper = params[:self.n_upper]
        w_lower = params[self.n_upper:self.n_upper + self.n_lower]
        dz_te = params[-1]
        
        # Cosine spacing
        beta = np.linspace(0, np.pi, 100)
        x = 0.5 * (1 - np.cos(beta))
        
        # Class function
        C = self.class_function(x)
        
        # Shape functions
        S_upper = self.shape_function(x, w_upper)
        S_lower = self.shape_function(x, w_lower)
        
        # CST coordinates
        zeta_upper = C * S_upper + x * dz_te
        zeta_lower = C * S_lower - x * dz_te
        
        # Combine
        coords = np.vstack([
            np.column_stack([x[::-1], zeta_upper[::-1]]),
            np.column_stack([x[1:], zeta_lower[1:]])
        ])
        
        return coords
    
    def get_bounds(self) -> list:
        """Parameter bounds"""
        bounds = []
        
        # Upper surface weights
        for _ in range(self.n_upper):
            bounds.append((-0.2, 0.5))
        
        # Lower surface weights
        for _ in range(self.n_lower):
            bounds.append((-0.5, 0.2))
        
        # TE thickness
        bounds.append((0.0, 0.01))
        
        return bounds
    
    def get_initial(self) -> np.ndarray:
        """Initial guess"""
        w_upper = np.array([0.2, 0.15, 0.1, 0.05, 0.02, 0.01][:self.n_upper])
        w_lower = np.array([-0.2, -0.15, -0.1, -0.05, -0.02, -0.01][:self.n_lower])
        dz_te = 0.001
        
        return np.concatenate([w_upper, w_lower, [dz_te]])


class FFDParametrization(BaseParametrization):
    """
    FFD (Free Form Deformation) parametrization
    
    Most flexible, uses Bernstein polynomials
    Can have 15-100+ parameters
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_naca = config.get('base_naca', '2412')
        self.nx = config.get('nx', 5)
        self.ny = config.get('ny', 3)
        self.n_params = self.nx * self.ny * 2  # x and y deformations
        
        # Generate base airfoil
        self.base_coords = self.generate_naca(self.base_naca)
    
    def generate_naca(self, naca: str) -> np.ndarray:
        """Generate base NACA airfoil"""
        # Simple NACA 4-digit generator
        m = int(naca[0]) / 100
        p = int(naca[1]) / 10
        t = int(naca[2:]) / 100
        
        naca_gen = NACAParametrization({})
        return naca_gen.generate_airfoil(np.array([m, p, t]))
    
    def bernstein(self, i: int, n: int, u: float) -> float:
        """Bernstein polynomial"""
        from math import factorial
        binom = factorial(n) / (factorial(i) * factorial(n - i))
        return binom * u**i * (1 - u)**(n - i)
    
    def ffd_deform(self, coords: np.ndarray, control_points: np.ndarray) -> np.ndarray:
        """
        Apply FFD deformation
        
        Parameters:
        -----------
        coords : np.ndarray
            Original coordinates
        control_points : np.ndarray
            Control point displacements (nx, ny, 2)
        
        Returns:
        --------
        np.ndarray
            Deformed coordinates
        """
        # Normalize to [0, 1]
        x_min, x_max = 0.0, 1.0
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()
        
        u = coords[:, 0]
        v = (coords[:, 1] - y_min) / (y_max - y_min)
        
        # Apply FFD
        deformed = coords.copy()
        
        for i in range(self.nx):
            for j in range(self.ny):
                Bi = np.array([self.bernstein(i, self.nx - 1, ui) for ui in u])
                Bj = np.array([self.bernstein(j, self.ny - 1, vj) for vj in v])
                
                weight = Bi * Bj
                
                deformed[:, 0] += weight * control_points[i, j, 0]
                deformed[:, 1] += weight * control_points[i, j, 1]
        
        return deformed
    
    def generate_airfoil(self, params: np.ndarray) -> np.ndarray:
        """
        Generate FFD airfoil
        
        Parameters:
        -----------
        params : np.ndarray
            Flat array of control point displacements
        
        Returns:
        --------
        np.ndarray
            Airfoil coordinates
        """
        # Reshape to control points
        control_points = params.reshape(self.nx, self.ny, 2)
        
        # Apply FFD
        deformed_coords = self.ffd_deform(self.base_coords, control_points)
        
        return deformed_coords
    
    def get_bounds(self) -> list:
        """Parameter bounds"""
        bounds = []
        
        for _ in range(self.n_params // 2):
            bounds.append((-0.02, 0.02))  # x displacement
        
        for _ in range(self.n_params // 2):
            bounds.append((-0.05, 0.05))  # y displacement
        
        return bounds
    
    def get_initial(self) -> np.ndarray:
        """Initial guess (no deformation)"""
        return np.zeros(self.n_params)
