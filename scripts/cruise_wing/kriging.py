"""
Kriging Surrogate Model for Cruise Wing Optimization

Gaussian Process Regression with Matérn 5/2 kernel
- Optimized for NACA parametrization (3 variables)
- Uncertainty quantification support
- Gradient information for SLSQP
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pickle

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, ConstantKernel, Matern, WhiteKernel
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from scipy.stats import qmc


class CruiseWingKriging:
    """
    Kriging Surrogate Model for Cruise Wing Optimization
    
    Optimized settings:
    - Kernel: Matérn 5/2 (smooth, differentiable)
    - Training samples: 50-100 for 3 variables
    - Sampling: Latin Hypercube
    """
    
    def __init__(self, 
                 kernel: str = 'matern',
                 length_scale: float = 1.0,
                 length_scale_bounds: Tuple[float, float] = (1e-3, 100.0),
                 noise_level: float = 1e-5,
                 n_restarts: int = 10,
                 normalize: bool = True):
        """
        Initialize Kriging model
        
        Parameters
        ----------
        kernel : str
            Kernel type ('matern', 'rbf')
        length_scale : float
            Initial length scale
        length_scale_bounds : tuple
            Bounds for length scale optimization
        noise_level : float
            White noise level for numerical stability
        n_restarts : int
            Number of optimizer restarts
        normalize : bool
            Whether to normalize inputs and outputs
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for Kriging")
        
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.noise_level = noise_level
        self.n_restarts = n_restarts
        self.normalize = normalize
        
        # Models for each output metric
        self.models: Dict[str, GaussianProcessRegressor] = {}
        self.scalers_X: Dict[str, StandardScaler] = {}
        self.scalers_y: Dict[str, StandardScaler] = {}
        
        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[Dict] = None
        
        self.is_trained = False
        self.metrics: List[str] = []
    
    def _create_kernel(self, n_dims: int):
        """Create GP kernel"""
        if self.kernel_type == 'matern':
            kernel = (
                ConstantKernel(1.0, constant_value_bounds=(1e-3, 100.0)) *
                Matern(
                    length_scale=[self.length_scale] * n_dims,
                    length_scale_bounds=self.length_scale_bounds,
                    nu=2.5  # Matérn 5/2
                ) +
                WhiteKernel(
                    noise_level=self.noise_level,
                    noise_level_bounds=(1e-10, 1e-1)
                )
            )
        elif self.kernel_type == 'rbf':
            kernel = (
                ConstantKernel(1.0, constant_value_bounds=(1e-3, 100.0)) *
                RBF(
                    length_scale=[self.length_scale] * n_dims,
                    length_scale_bounds=self.length_scale_bounds
                ) +
                WhiteKernel(
                    noise_level=self.noise_level,
                    noise_level_bounds=(1e-10, 1e-1)
                )
            )
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
        
        return kernel
    
    def train(self, X: np.ndarray, y: Union[np.ndarray, List[Dict]],
              metrics: Optional[List[str]] = None,
              verbose: bool = True) -> Dict:
        """
        Train Kriging model
        
        Parameters
        ----------
        X : np.ndarray
            Training inputs (n_samples, n_features)
        y : np.ndarray or list of dict
            Training outputs
        metrics : list, optional
            Output metric names (if y is array)
        verbose : bool
            Print training progress
            
        Returns
        -------
        dict
            Training statistics
        """
        self.X_train = X.copy()
        n_samples, n_dims = X.shape
        
        if verbose:
            print(f"\n🧠 Training Kriging Surrogate")
            print(f"   Samples: {n_samples}")
            print(f"   Dimensions: {n_dims}")
            print(f"   Kernel: {self.kernel_type}")
        
        # Handle dict outputs
        if isinstance(y[0], dict):
            self.metrics = list(y[0].keys())
            self.y_train = {m: np.array([yi[m] for yi in y]) for m in self.metrics}
        else:
            if metrics:
                self.metrics = metrics
            else:
                self.metrics = ['output']
            self.y_train = {self.metrics[0]: y}
        
        stats = {}
        
        for metric in self.metrics:
            y_metric = self.y_train[metric]
            
            # Normalize
            if self.normalize:
                self.scalers_X[metric] = StandardScaler()
                self.scalers_y[metric] = StandardScaler()
                
                X_scaled = self.scalers_X[metric].fit_transform(X)
                y_scaled = self.scalers_y[metric].fit_transform(y_metric.reshape(-1, 1)).ravel()
            else:
                X_scaled = X
                y_scaled = y_metric
            
            # Create and train GP
            kernel = self._create_kernel(n_dims)
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                normalize_y=False,  # We handle normalization
                alpha=1e-10
            )
            
            gp.fit(X_scaled, y_scaled)
            self.models[metric] = gp
            
            # Calculate training score
            y_pred = gp.predict(X_scaled)
            if self.normalize:
                y_pred = self.scalers_y[metric].inverse_transform(y_pred.reshape(-1, 1)).ravel()
            
            r2 = 1 - np.sum((y_metric - y_pred)**2) / np.sum((y_metric - np.mean(y_metric))**2)
            rmse = np.sqrt(np.mean((y_metric - y_pred)**2))
            
            stats[metric] = {
                'R2': r2,
                'RMSE': rmse,
                'kernel': str(gp.kernel_)
            }
            
            if verbose:
                print(f"   ✓ {metric}: R²={r2:.4f}, RMSE={rmse:.6f}")
        
        self.is_trained = True
        return stats
    
    def predict(self, X: np.ndarray, 
                return_std: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """
        Predict using trained model
        
        Parameters
        ----------
        X : np.ndarray
            Input points (n_points, n_features)
        return_std : bool
            Return prediction uncertainty
            
        Returns
        -------
        dict or tuple
            Predictions (and uncertainties if return_std=True)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Ensure 2D
        X = np.atleast_2d(X)
        
        predictions = {}
        uncertainties = {}
        
        for metric in self.metrics:
            # Normalize input
            if self.normalize:
                X_scaled = self.scalers_X[metric].transform(X)
            else:
                X_scaled = X
            
            # Predict
            if return_std:
                y_pred, y_std = self.models[metric].predict(X_scaled, return_std=True)
            else:
                y_pred = self.models[metric].predict(X_scaled)
            
            # Denormalize
            if self.normalize:
                y_pred = self.scalers_y[metric].inverse_transform(
                    y_pred.reshape(-1, 1)
                ).ravel()
                if return_std:
                    y_std = y_std * self.scalers_y[metric].scale_[0]
            
            predictions[metric] = y_pred
            if return_std:
                uncertainties[metric] = y_std
        
        if return_std:
            return predictions, uncertainties
        return predictions
    
    def predict_single(self, x: np.ndarray, 
                       return_std: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """
        Predict for a single point
        
        Parameters
        ----------
        x : np.ndarray
            Single input point (n_features,)
        return_std : bool
            Return uncertainty
            
        Returns
        -------
        dict
            Single prediction values
        """
        result = self.predict(x.reshape(1, -1), return_std=return_std)
        
        if return_std:
            pred, std = result
            return (
                {k: v[0] for k, v in pred.items()},
                {k: v[0] for k, v in std.items()}
            )
        else:
            return {k: v[0] for k, v in result.items()}
    
    def cross_validate(self, n_folds: int = 5) -> Dict:
        """
        Cross-validation for model assessment
        
        Parameters
        ----------
        n_folds : int
            Number of CV folds
            
        Returns
        -------
        dict
            CV scores for each metric
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("No training data available")
        
        cv_scores = {}
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for metric in self.metrics:
            scores = []
            
            for train_idx, val_idx in kf.split(self.X_train):
                X_train_fold = self.X_train[train_idx]
                X_val_fold = self.X_train[val_idx]
                y_train_fold = self.y_train[metric][train_idx]
                y_val_fold = self.y_train[metric][val_idx]
                
                # Train on fold
                kernel = self._create_kernel(X_train_fold.shape[1])
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=3,
                    normalize_y=True
                )
                gp.fit(X_train_fold, y_train_fold)
                
                # Predict
                y_pred = gp.predict(X_val_fold)
                
                # R² score
                r2 = 1 - np.sum((y_val_fold - y_pred)**2) / np.sum((y_val_fold - np.mean(y_val_fold))**2)
                scores.append(r2)
            
            cv_scores[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        return cv_scores
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers_X': self.scalers_X,
                'scalers_y': self.scalers_y,
                'metrics': self.metrics,
                'config': {
                    'kernel_type': self.kernel_type,
                    'length_scale': self.length_scale,
                    'normalize': self.normalize
                }
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CruiseWingKriging':
        """Load model from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            kernel=data['config']['kernel_type'],
            length_scale=data['config']['length_scale'],
            normalize=data['config']['normalize']
        )
        model.models = data['models']
        model.scalers_X = data['scalers_X']
        model.scalers_y = data['scalers_y']
        model.metrics = data['metrics']
        model.is_trained = True
        
        return model


class LHSSampler:
    """
    Latin Hypercube Sampling for training data generation
    """
    
    def __init__(self, bounds: List[Tuple[float, float]], seed: int = 42):
        """
        Initialize sampler
        
        Parameters
        ----------
        bounds : list
            List of (min, max) tuples for each dimension
        seed : int
            Random seed
        """
        self.bounds = bounds
        self.n_dims = len(bounds)
        self.seed = seed
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate Latin Hypercube samples
        
        Parameters
        ----------
        n_samples : int
            Number of samples
            
        Returns
        -------
        np.ndarray
            Samples in parameter space (n_samples, n_dims)
        """
        sampler = qmc.LatinHypercube(d=self.n_dims, seed=self.seed)
        samples_unit = sampler.random(n=n_samples)
        
        # Scale to bounds
        samples = self.lower + samples_unit * (self.upper - self.lower)
        
        return samples
    
    def sample_centered(self, n_samples: int, 
                        center: np.ndarray,
                        radius_fraction: float = 0.5) -> np.ndarray:
        """
        Generate samples centered around a point
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        center : np.ndarray
            Center point
        radius_fraction : float
            Fraction of full range for local sampling
            
        Returns
        -------
        np.ndarray
            Samples around center
        """
        range_full = self.upper - self.lower
        local_lower = np.maximum(self.lower, center - radius_fraction * range_full / 2)
        local_upper = np.minimum(self.upper, center + radius_fraction * range_full / 2)
        
        sampler = qmc.LatinHypercube(d=self.n_dims, seed=self.seed)
        samples_unit = sampler.random(n=n_samples)
        
        samples = local_lower + samples_unit * (local_upper - local_lower)
        
        return samples


def generate_training_data(bounds: List[Tuple[float, float]],
                           n_samples: int,
                           evaluate_func,
                           verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate training data for surrogate model
    
    Parameters
    ----------
    bounds : list
        Parameter bounds
    n_samples : int
        Number of samples
    evaluate_func : callable
        Function that takes parameters and returns results dict
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple
        (X_train, y_train) arrays
    """
    sampler = LHSSampler(bounds)
    samples = sampler.sample(n_samples)
    
    X_train = []
    y_train = []
    
    if verbose:
        print(f"\n📊 Generating {n_samples} training samples...")
    
    for i, params in enumerate(samples):
        if verbose and (i + 1) % 10 == 0:
            print(f"   Progress: {i + 1}/{n_samples}")
        
        try:
            result = evaluate_func(params)
            if result is not None:
                X_train.append(params)
                y_train.append(result)
        except Exception as e:
            if verbose:
                print(f"   Warning: Sample {i} failed - {e}")
            continue
    
    if verbose:
        print(f"   ✓ Generated {len(X_train)} valid samples")
    
    return np.array(X_train), y_train
