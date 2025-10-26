"""
GAN-based Airfoil Generator
Wasserstein GAN with CNN architecture for generating realistic airfoil shapes
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class AirfoilGAN:
    """
    Wasserstein GAN for airfoil generation
    논문 참조: CNN-based generator and discriminator
    """
    
    def __init__(self,
                 latent_dim: int = 100,
                 airfoil_dim: int = 502,  # 251 points * 2 (x, y)
                 n_critic: int = 5,
                 learning_rate: float = 0.0001):
        """
        Args:
            latent_dim: Noise vector 차원
            airfoil_dim: 에어포일 좌표 차원 (flatten)
            n_critic: Generator 1회당 Critic 학습 횟수
            learning_rate: 학습률
        """
        self.latent_dim = latent_dim
        self.airfoil_dim = airfoil_dim
        self.n_critic = n_critic
        self.learning_rate = learning_rate
        
        # 모델 생성
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        
        # Optimizer
        self.g_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.9
        )
        self.c_optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.9
        )
        
    def build_generator(self) -> keras.Model:
        """
        CNN-based Generator
        논문: convolutional layers for smooth airfoils
        """
        model = keras.Sequential([
            # Input: latent vector
            layers.Input(shape=(self.latent_dim,)),
            
            # Dense layer
            layers.Dense(128 * 32, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((32, 128)),
            
            # Conv1D Transpose layers
            # 32 -> 64
            layers.Conv1DTranspose(128, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 64 -> 128
            layers.Conv1DTranspose(64, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 128 -> 256
            layers.Conv1DTranspose(32, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 256 -> 502 (or nearest then adjust)
            layers.Conv1DTranspose(16, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Final layer
            layers.Conv1D(2, 5, padding='same', activation='tanh'),
            
            # Flatten to (502,)
            layers.Flatten(),
        ], name='generator')
        
        # 출력 차원 조정
        # 실제 출력이 airfoil_dim과 다를 수 있으므로 Dense로 조정
        output_layer = keras.Sequential([
            model,
            layers.Dense(self.airfoil_dim, activation='tanh')
        ])
        
        logger.info(f"Generator output shape: {output_layer.output_shape}")
        return output_layer
    
    def build_critic(self) -> keras.Model:
        """
        CNN-based Critic (Discriminator in WGAN)
        논문: 4 convolutional layers, filter size=5, 64 filters
        """
        # Reshape for Conv1D: (batch, 251, 2)
        n_points = self.airfoil_dim // 2
        
        model = keras.Sequential([
            layers.Input(shape=(self.airfoil_dim,)),
            layers.Reshape((n_points, 2)),
            
            # Conv layer 1
            layers.Conv1D(64, 5, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Conv layer 2
            layers.Conv1D(64, 5, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Conv layer 3
            layers.Conv1D(64, 5, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Conv layer 4
            layers.Conv1D(64, 5, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Flatten and output
            layers.Flatten(),
            layers.Dense(1)  # No activation for WGAN
        ], name='critic')
        
        logger.info(f"Critic input shape: {model.input_shape}")
        return model
    
    @tf.function
    def train_step(self, real_airfoils: tf.Tensor) -> Tuple[float, float]:
        """
        한 번의 학습 스텝 (WGAN)
        
        Args:
            real_airfoils: shape (batch_size, airfoil_dim)
            
        Returns:
            c_loss, g_loss: Critic과 Generator loss
        """
        batch_size = tf.shape(real_airfoils)[0]
        
        # Train Critic
        for _ in range(self.n_critic):
            noise = tf.random.normal([batch_size, self.latent_dim])
            
            with tf.GradientTape() as tape:
                fake_airfoils = self.generator(noise, training=True)
                
                real_output = self.critic(real_airfoils, training=True)
                fake_output = self.critic(fake_airfoils, training=True)
                
                # Wasserstein loss
                c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                
                # Gradient penalty (optional but recommended)
                gp = self.gradient_penalty(real_airfoils, fake_airfoils)
                c_loss += 10.0 * gp
            
            c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(c_gradients, self.critic.trainable_variables)
            )
        
        # Train Generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            fake_airfoils = self.generator(noise, training=True)
            fake_output = self.critic(fake_airfoils, training=True)
            
            # Generator wants to maximize critic output (minimize negative)
            g_loss = -tf.reduce_mean(fake_output)
        
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        
        return c_loss, g_loss
    
    def gradient_penalty(self, real: tf.Tensor, fake: tf.Tensor) -> tf.Tensor:
        """
        Gradient Penalty for WGAN-GP
        """
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        
        interpolated = alpha * real + (1 - alpha) * fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp
    
    def train(self,
              airfoils: np.ndarray,
              epochs: int = 1000,
              batch_size: int = 32,
              save_interval: int = 100,
              save_dir: Optional[Path] = None):
        """
        GAN 학습
        
        Args:
            airfoils: shape (n_samples, airfoil_dim) 학습 데이터
            epochs: 에폭 수
            batch_size: 배치 크기
            save_interval: 모델 저장 간격
            save_dir: 모델 저장 디렉토리
        """
        logger.info(f"Training GAN for {epochs} epochs")
        
        # Dataset 생성
        dataset = tf.data.Dataset.from_tensor_slices(airfoils)
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # 학습 루프
        for epoch in range(epochs):
            c_losses = []
            g_losses = []
            
            for real_batch in dataset:
                c_loss, g_loss = self.train_step(real_batch)
                c_losses.append(c_loss.numpy())
                g_losses.append(g_loss.numpy())
            
            # 로깅
            avg_c_loss = np.mean(c_losses)
            avg_g_loss = np.mean(g_losses)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"C Loss: {avg_c_loss:.4f}, G Loss: {avg_g_loss:.4f}"
                )
            
            # 모델 저장
            if save_dir and (epoch + 1) % save_interval == 0:
                self.save_models(save_dir, epoch + 1)
        
        logger.info("Training completed!")
    
    def generate(self, n_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        새로운 에어포일 생성
        
        Args:
            n_samples: 생성할 샘플 수
            seed: Random seed
            
        Returns:
            generated: shape (n_samples, airfoil_dim)
        """
        if seed is not None:
            tf.random.set_seed(seed)
        
        noise = tf.random.normal([n_samples, self.latent_dim])
        generated = self.generator(noise, training=False)
        
        return generated.numpy()
    
    def save_models(self, save_dir: Path, epoch: Optional[int] = None):
        """모델 저장"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        suffix = f"_epoch{epoch}" if epoch else ""
        
        generator_path = save_dir / f"generator{suffix}.h5"
        critic_path = save_dir / f"critic{suffix}.h5"
        
        self.generator.save(generator_path)
        self.critic.save(critic_path)
        
        logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: Path, epoch: Optional[int] = None):
        """모델 로드"""
        save_dir = Path(save_dir)
        
        suffix = f"_epoch{epoch}" if epoch else ""
        
        generator_path = save_dir / f"generator{suffix}.h5"
        critic_path = save_dir / f"critic{suffix}.h5"
        
        if generator_path.exists():
            self.generator = keras.models.load_model(generator_path)
            logger.info(f"Generator loaded from {generator_path}")
        
        if critic_path.exists():
            self.critic = keras.models.load_model(critic_path)
            logger.info(f"Critic loaded from {critic_path}")


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # GAN 생성
    gan = AirfoilGAN(latent_dim=100, airfoil_dim=502)
    
    # 더미 데이터로 테스트
    dummy_data = np.random.randn(100, 502).astype(np.float32)
    
    print("GAN model created successfully")
    print(f"Generator summary:")
    gan.generator.summary()
    
    # 생성 테스트
    samples = gan.generate(n_samples=5)
    print(f"\nGenerated samples shape: {samples.shape}")
    