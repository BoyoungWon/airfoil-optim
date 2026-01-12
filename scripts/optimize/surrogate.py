"""
Surrogate Model Module

다양한 surrogate modeling 기법:
- Kriging/GPR: Gaussian Process Regression (3-30 parameters)
- Neural Network: Deep learning (50+ parameters)
- Polynomial: Response Surface Method (3-10 parameters)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BaseSurrogate(ABC):
    """Base class for surrogate models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train surrogate model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using surrogate model"""
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score"""
        y_pred = self.predict(X)
        
        # Handle dict output
        if isinstance(y, list) and isinstance(y[0], dict):
            # Use first metric
            metric = list(y[0].keys())[0]
            y_true = np.array([yi[metric] for yi in y])
            y_pred_val = np.array([yi[metric] for yi in y_pred])
        else:
            y_true = y
            y_pred_val = y_pred
        
        ss_res = np.sum((y_true - y_pred_val) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        return 1 - ss_res / ss_tot


class KrigingSurrogate(BaseSurrogate):
    """
    Kriging/Gaussian Process Regression
    
    Best for: 3-30 parameters
    Pros: Uncertainty quantification, works well with small data
    Cons: Slow for large datasets
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for Kriging")
        
        # Setup kernel
        kernel_type = config.get('kernel', 'matern')
        length_scale = config.get('length_scale', 1.0)
        
        if kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
        else:
            raise ValueError(f"Unknown kernel: {kernel_type}")
        
        self.models = {}  # One model per output metric
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train Kriging model
        
        Parameters:
        -----------
        X : np.ndarray
            Training inputs (n_samples, n_features)
        y : list of dict or np.ndarray
            Training outputs
        """
        
        # Handle dict outputs
        if isinstance(y[0], dict):
            metrics = list(y[0].keys())
            
            for metric in metrics:
                y_metric = np.array([yi[metric] for yi in y])
                
                kernel_type = self.config.get('kernel', 'matern')
                length_scale = self.config.get('length_scale', 1.0)
                
                if kernel_type == 'rbf':
                    kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
                else:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
                
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=10,
                    normalize_y=True
                )
                
                model.fit(X, y_metric)
                self.models[metric] = model
                
                print(f"  ✓ Trained Kriging for {metric}")
        else:
            # Single output
            model = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=10,
                normalize_y=True
            )
            model.fit(X, y)
            self.models['output'] = model
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Kriging
        
        Returns:
        --------
        dict or np.ndarray
            Predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if len(self.models) > 1:
            # Multiple outputs
            predictions = []
            for i in range(X.shape[0]):
                pred = {}
                for metric, model in self.models.items():
                    pred[metric] = model.predict(X[i:i+1])[0]
                predictions.append(pred)
            return predictions
        else:
            # Single output
            model = list(self.models.values())[0]
            return model.predict(X)


class NeuralNetworkSurrogate(BaseSurrogate):
    """
    Neural Network surrogate
    
    Best for: 50+ parameters
    Pros: Handles high-dimensional spaces, non-linear relationships
    Cons: Requires more training data, black-box
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required for Neural Network")
        
        self.hidden_layers = config.get('hidden_layers', [64, 32, 16])
        self.activation = config.get('activation', 'relu')
        self.epochs = config.get('epochs', 1000)
        self.lr = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_X = None
        self.scaler_y = None
        self.output_metrics = None
    
    def build_network(self, input_dim: int, output_dim: int):
        """Build neural network"""
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'tanh':
                layers.append(nn.Tanh())
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers).to(self.device)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train neural network"""
        
        # Normalize inputs
        self.scaler_X = {'mean': X.mean(axis=0), 'std': X.std(axis=0) + 1e-8}
        X_norm = (X - self.scaler_X['mean']) / self.scaler_X['std']
        
        # Handle dict outputs
        if isinstance(y[0], dict):
            self.output_metrics = list(y[0].keys())
            y_array = np.array([[yi[m] for m in self.output_metrics] for yi in y])
        else:
            y_array = y
            self.output_metrics = ['output']
        
        # Normalize outputs
        self.scaler_y = {'mean': y_array.mean(axis=0), 'std': y_array.std(axis=0) + 1e-8}
        y_norm = (y_array - self.scaler_y['mean']) / self.scaler_y['std']
        
        # Build network
        input_dim = X.shape[1]
        output_dim = y_array.shape[1] if len(y_array.shape) > 1 else 1
        
        self.model = self.build_network(input_dim, output_dim)
        
        # Training
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_norm).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # Training loop
        n_samples = X_tensor.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for epoch in range(self.epochs):
            # Shuffle
            indices = torch.randperm(n_samples)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0.0
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / n_batches:.6f}")
        
        self.is_trained = True
        print(f"✓ Trained Neural Network")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using neural network"""
        
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Normalize
        X_norm = (X - self.scaler_X['mean']) / self.scaler_X['std']
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(self.device)
            y_pred_norm = self.model(X_tensor).cpu().numpy()
        
        # Denormalize
        y_pred = y_pred_norm * self.scaler_y['std'] + self.scaler_y['mean']
        
        # Convert to dict if needed
        if len(self.output_metrics) > 1:
            predictions = []
            for i in range(y_pred.shape[0]):
                pred = {metric: y_pred[i, j] for j, metric in enumerate(self.output_metrics)}
                predictions.append(pred)
            return predictions
        else:
            return y_pred


class PolynomialSurrogate(BaseSurrogate):
    """
    Polynomial Response Surface
    
    Best for: 3-10 parameters
    Pros: Simple, interpretable, fast
    Cons: Limited flexibility for complex relationships
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.degree = config.get('degree', 2)
        self.coefficients = None
        self.output_metrics = None
    
    def polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Generate polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=self.degree)
        return poly.fit_transform(X)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train polynomial model"""
        
        # Generate polynomial features
        X_poly = self.polynomial_features(X)
        
        # Handle dict outputs
        if isinstance(y[0], dict):
            self.output_metrics = list(y[0].keys())
            self.coefficients = {}
            
            for metric in self.output_metrics:
                y_metric = np.array([yi[metric] for yi in y])
                
                # Least squares fit
                coef = np.linalg.lstsq(X_poly, y_metric, rcond=None)[0]
                self.coefficients[metric] = coef
                
                print(f"  ✓ Trained Polynomial for {metric}")
        else:
            y_array = y
            self.coefficients = {'output': np.linalg.lstsq(X_poly, y_array, rcond=None)[0]}
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using polynomial model"""
        
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        X_poly = self.polynomial_features(X)
        
        if len(self.coefficients) > 1:
            # Multiple outputs
            predictions = []
            for i in range(X.shape[0]):
                pred = {}
                for metric, coef in self.coefficients.items():
                    pred[metric] = np.dot(X_poly[i], coef)
                predictions.append(pred)
            return predictions
        else:
            coef = list(self.coefficients.values())[0]
            return X_poly @ coef
