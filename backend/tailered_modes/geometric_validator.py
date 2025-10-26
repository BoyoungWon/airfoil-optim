"""
Geometric Validity Discriminator
CNN-based model to detect geometric abnormalities in airfoil shapes
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class GeometricValidator:
    """
    CNN-based discriminative model for airfoil validity detection
    논문: 4 convolutional layers, filter size=5, 64 filters, MSE loss
    """
    
    def __init__(self,
                 airfoil_dim: int = 502,
                 n_filters: int = 64,
                 filter_size: int = 5):
        """
        Args:
            airfoil_dim: 에어포일 좌표 차원 (flatten)
            n_filters: 각 conv layer의 필터 수
            filter_size: 필터 크기
        """
        self.airfoil_dim = airfoil_dim
        self.n_points = airfoil_dim // 2  # 251
        self.n_filters = n_filters
        self.filter_size = filter_size
        
        # 모델 생성
        self.model = self.build_model()
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error as in paper
            metrics=['mae', 'accuracy']
        )
    
    def build_model(self) -> keras.Model:
        """
        CNN-based validity discriminator
        논문: 4 conv layers, filter size 5, 64 filters each
        """
        model = keras.Sequential([
            # Input: reshape to (n_points, 2) for Conv1D
            layers.Input(shape=(self.airfoil_dim,)),
            layers.Reshape((self.n_points, 2)),
            
            # Conv Layer 1
            layers.Conv1D(
                filters=self.n_filters,
                kernel_size=self.filter_size,
                padding='same',
                activation='relu'
            ),
            layers.MaxPooling1D(pool_size=2),
            
            # Conv Layer 2
            layers.Conv1D(
                filters=self.n_filters,
                kernel_size=self.filter_size,
                padding='same',
                activation='relu'
            ),
            layers.MaxPooling1D(pool_size=2),
            
            # Conv Layer 3
            layers.Conv1D(
                filters=self.n_filters,
                kernel_size=self.filter_size,
                padding='same',
                activation='relu'
            ),
            layers.MaxPooling1D(pool_size=2),
            
            # Conv Layer 4
            layers.Conv1D(
                filters=self.n_filters,
                kernel_size=self.filter_size,
                padding='same',
                activation='relu'
            ),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output: validity score [0, 1]
            layers.Dense(1, activation='sigmoid')
        ], name='geometric_validator')
        
        logger.info("Geometric Validator model created")
        return model
    
    def create_training_data(self,
                            realistic_airfoils: np.ndarray,
                            n_abnormal: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터 생성: realistic + abnormal airfoils
        
        Args:
            realistic_airfoils: shape (n, airfoil_dim) realistic 에어포일
            n_abnormal: 생성할 abnormal 샘플 수 (None이면 realistic과 동일)
            
        Returns:
            X, y: 학습 데이터와 레이블
        """
        n_realistic = len(realistic_airfoils)
        if n_abnormal is None:
            n_abnormal = n_realistic
        
        logger.info(f"Creating training data: {n_realistic} realistic + {n_abnormal} abnormal")
        
        # Abnormal airfoils 생성 (random deformation)
        abnormal_airfoils = self.generate_abnormal_airfoils(
            realistic_airfoils, n_abnormal
        )
        
        # 데이터 결합
        X = np.vstack([realistic_airfoils, abnormal_airfoils])
        
        # 레이블: realistic=1, abnormal=0
        y = np.concatenate([
            np.ones(n_realistic),
            np.zeros(n_abnormal)
        ])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def generate_abnormal_airfoils(self,
                                  realistic_airfoils: np.ndarray,
                                  n_samples: int) -> np.ndarray:
        """
        Abnormal airfoils 생성 by random deformation
        
        Strategy:
        1. 극단적인 두께 변화
        2. 불규칙한 표면
        3. Self-intersection
        4. 비현실적인 camber
        """
        n_realistic = len(realistic_airfoils)
        abnormal = []
        
        for _ in range(n_samples):
            # 랜덤하게 realistic airfoil 선택
            base = realistic_airfoils[np.random.randint(n_realistic)].copy()
            base_coords = base.reshape(self.n_points * 2, 2)
            
            # Random deformation 타입 선택
            deform_type = np.random.choice(['thickness', 'noise', 'discontinuity', 'extreme'])
            
            if deform_type == 'thickness':
                # 극단적 두께 변화
                scale = np.random.uniform(0.1, 3.0)
                base_coords[:, 1] *= scale
            
            elif deform_type == 'noise':
                # 불규칙한 표면
                noise = np.random.normal(0, 0.05, base_coords.shape)
                base_coords += noise
            
            elif deform_type == 'discontinuity':
                # 급격한 불연속
                split_idx = np.random.randint(50, self.n_points * 2 - 50)
                offset = np.random.uniform(-0.2, 0.2, 2)
                base_coords[split_idx:] += offset
            
            elif deform_type == 'extreme':
                # 극단적 변형
                base_coords[:, 1] = np.abs(base_coords[:, 1]) * np.random.uniform(2, 5)
                base_coords += np.random.normal(0, 0.1, base_coords.shape)
            
            abnormal.append(base_coords.flatten())
        
        return np.array(abnormal)
    
    def train(self,
             realistic_airfoils: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32,
             validation_split: float = 0.2,
             save_path: Optional[Path] = None):
        """
        모델 학습
        
        Args:
            realistic_airfoils: shape (n, airfoil_dim) realistic 에어포일
            epochs: 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            save_path: 모델 저장 경로
        """
        # 학습 데이터 생성
        X, y = self.create_training_data(realistic_airfoils)
        
        logger.info(f"Training data: {X.shape}, Labels: {y.shape}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # 학습
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        # 모델 저장
        if save_path:
            self.save_model(save_path)
        
        return history
    
    def predict_validity(self, airfoils: np.ndarray) -> np.ndarray:
        """
        에어포일의 유효성 점수 예측
        
        Args:
            airfoils: shape (n, airfoil_dim)
            
        Returns:
            scores: shape (n,) validity scores [0, 1]
                   1 = realistic, 0 = abnormal
        """
        scores = self.model.predict(airfoils, verbose=0)
        return scores.flatten()
    
    def filter_valid_airfoils(self,
                             airfoils: np.ndarray,
                             threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        """
        유효성 threshold 기준으로 에어포일 필터링
        
        Args:
            airfoils: shape (n, airfoil_dim)
            threshold: 최소 유효성 점수 (논문: 0.75)
            
        Returns:
            valid_airfoils: 유효한 에어포일들
            validity_scores: 각 에어포일의 유효성 점수
        """
        scores = self.predict_validity(airfoils)
        valid_mask = scores >= threshold
        
        valid_airfoils = airfoils[valid_mask]
        
        logger.info(
            f"Filtered: {np.sum(valid_mask)}/{len(airfoils)} "
            f"airfoils passed threshold {threshold}"
        )
        
        return valid_airfoils, scores
    
    def compute_validity_gradient(self, airfoil: np.ndarray) -> np.ndarray:
        """
        유효성 점수에 대한 그래디언트 계산
        (최적화에서 constraint로 사용)
        
        Args:
            airfoil: shape (airfoil_dim,)
            
        Returns:
            gradient: shape (airfoil_dim,)
        """
        airfoil_tensor = tf.constant(airfoil.reshape(1, -1), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(airfoil_tensor)
            score = self.model(airfoil_tensor, training=False)
        
        gradient = tape.gradient(score, airfoil_tensor)
        
        return gradient.numpy().flatten()
    
    def save_model(self, save_path: Path):
        """모델 저장"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: Path):
        """모델 로드"""
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # Validator 생성
    validator = GeometricValidator(airfoil_dim=502)
    
    print("Geometric Validator created successfully")
    print("\nModel summary:")
    validator.model.summary()
    
    # 더미 데이터로 테스트
    dummy_realistic = np.random.randn(50, 502).astype(np.float32)
    dummy_test = np.random.randn(10, 502).astype(np.float32)
    
    # 유효성 예측
    scores = validator.predict_validity(dummy_test)
    print(f"\nValidity scores: {scores}")