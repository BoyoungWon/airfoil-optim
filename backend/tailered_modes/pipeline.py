"""
Tailored Modal Parameterization Pipeline
Complete workflow for generating tailored airfoil modes
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # Data paths
    uiuc_data_dir: Path
    output_dir: Path
    
    # Preprocessing
    n_points: int = 251
    
    # GAN training
    gan_latent_dim: int = 100
    gan_epochs: int = 1000
    gan_batch_size: int = 32
    
    # Validator training
    validator_epochs: int = 100
    validator_batch_size: int = 32
    validity_threshold: float = 0.75
    
    # Optimal sampling
    n_samples: int = 500
    sample_max_attempts: int = 1000
    
    # Geometric constraints
    max_thickness: Optional[float] = 0.15
    min_thickness: Optional[float] = 0.08
    thickness_at_075c: Optional[tuple] = (0.07, 0.20)
    min_area_ratio: Optional[float] = 0.9  # relative to NACA0015
    
    # Mode extraction
    n_modes: int = 15
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            'uiuc_data_dir': str(self.uiuc_data_dir),
            'output_dir': str(self.output_dir),
            'n_points': self.n_points,
            'gan_latent_dim': self.gan_latent_dim,
            'gan_epochs': self.gan_epochs,
            'gan_batch_size': self.gan_batch_size,
            'validator_epochs': self.validator_epochs,
            'validator_batch_size': self.validator_batch_size,
            'validity_threshold': self.validity_threshold,
            'n_samples': self.n_samples,
            'sample_max_attempts': self.sample_max_attempts,
            'max_thickness': self.max_thickness,
            'min_thickness': self.min_thickness,
            'thickness_at_075c': self.thickness_at_075c,
            'min_area_ratio': self.min_area_ratio,
            'n_modes': self.n_modes
        }


class TailoredModesPipeline:
    """
    Tailored Modal Parameterization 전체 파이프라인
    
    Steps:
    1. UIUC airfoil database 전처리
    2. GAN 학습 (realistic airfoil generation)
    3. Geometric validator 학습 (CNN-based)
    4. Optimal sampling (constraint satisfaction)
    5. Mode extraction (SVD)
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: PipelineConfig 객체
        """
        self.config = config
        
        # 출력 디렉토리 생성
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Components (will be initialized)
        self.preprocessor = None
        self.gan = None
        self.validator = None
        self.sampler = None
        self.extractor = None
        
        # Data
        self.uiuc_airfoils = None
        self.tailored_samples = None
        self.tailored_modes = None
    
    def run_full_pipeline(self,
                         skip_training: bool = False,
                         load_existing: bool = False):
        """
        전체 파이프라인 실행
        
        Args:
            skip_training: True면 학습 스킵 (저장된 모델 사용)
            load_existing: True면 기존 결과 로드
        """
        logger.info("="*60)
        logger.info("Starting Tailored Modal Parameterization Pipeline")
        logger.info("="*60)
        
        # Step 1: Preprocessing
        self.step1_preprocess_data(load_existing)
        
        if not skip_training:
            # Step 2: Train GAN
            self.step2_train_gan()
            
            # Step 3: Train Validator
            self.step3_train_validator()
        else:
            # Load pre-trained models
            self.load_pretrained_models()
        
        # Step 4: Generate Optimal Samples
        self.step4_generate_samples(load_existing)
        
        # Step 5: Extract Modes
        self.step5_extract_modes(load_existing)
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Tailored modes saved to: {self.config.output_dir}")
        logger.info("="*60)
        
        return self.tailored_modes, self.extractor
    
    def step1_preprocess_data(self, load_existing: bool = False):
        """Step 1: UIUC 데이터 전처리"""
        logger.info("\n[Step 1] Preprocessing UIUC database...")
        
        processed_path = self.config.output_dir / "uiuc_processed.npy"
        
        if load_existing and processed_path.exists():
            logger.info(f"Loading existing processed data from {processed_path}")
            self.uiuc_airfoils = np.load(processed_path)
        else:
            # Import here to avoid circular dependency
            from preprocessing import AirfoilPreprocessor
            
            self.preprocessor = AirfoilPreprocessor(n_points=self.config.n_points)
            self.uiuc_airfoils = self.preprocessor.process_database(
                self.config.uiuc_data_dir,
                output_path=processed_path
            )
        
        logger.info(f"Processed {len(self.uiuc_airfoils)} airfoils")
        logger.info(f"Airfoil dimension: {self.uiuc_airfoils.shape[1]}")
    
    def step2_train_gan(self):
        """Step 2: GAN 학습"""
        logger.info("\n[Step 2] Training GAN...")
        
        from gan_generator import AirfoilGAN
        
        self.gan = AirfoilGAN(
            latent_dim=self.config.gan_latent_dim,
            airfoil_dim=self.uiuc_airfoils.shape[1]
        )
        
        save_dir = self.config.output_dir / "gan_models"
        
        self.gan.train(
            airfoils=self.uiuc_airfoils.astype(np.float32),
            epochs=self.config.gan_epochs,
            batch_size=self.config.gan_batch_size,
            save_interval=100,
            save_dir=save_dir
        )
        
        logger.info(f"GAN models saved to {save_dir}")
    
    def step3_train_validator(self):
        """Step 3: Geometric Validator 학습"""
        logger.info("\n[Step 3] Training Geometric Validator...")
        
        from geometric_validator import GeometricValidator
        
        self.validator = GeometricValidator(
            airfoil_dim=self.uiuc_airfoils.shape[1]
        )
        
        save_path = self.config.output_dir / "validator_model.h5"
        
        self.validator.train(
            realistic_airfoils=self.uiuc_airfoils.astype(np.float32),
            epochs=self.config.validator_epochs,
            batch_size=self.config.validator_batch_size,
            save_path=save_path
        )
        
        logger.info(f"Validator model saved to {save_path}")
    
    def step4_generate_samples(self, load_existing: bool = False):
        """Step 4: Optimal Sampling"""
        logger.info("\n[Step 4] Generating Optimal Samples...")
        
        samples_path = self.config.output_dir / "tailored_samples.npy"
        
        if load_existing and samples_path.exists():
            logger.info(f"Loading existing samples from {samples_path}")
            self.tailored_samples = np.load(samples_path)
        else:
            from optimal_sampler import OptimalSampler, GeometricConstraints
            
            # Setup sampler
            self.sampler = OptimalSampler(
                geometric_validator=self.validator,
                n_points=self.config.n_points
            )
            
            # Define constraints
            constraints = GeometricConstraints(
                max_thickness=self.config.max_thickness,
                min_thickness=self.config.min_thickness,
                thickness_at_positions={
                    0.75: self.config.thickness_at_075c
                } if self.config.thickness_at_075c else None,
                min_area=self.config.min_area_ratio * 0.0597 if self.config.min_area_ratio else None
                # NACA0015 area ≈ 0.0597
            )
            
            # GAN generator function
            def gan_generator(n_samples):
                return self.gan.generate(n_samples=n_samples)
            
            # Generate samples
            self.tailored_samples = self.sampler.generate_samples(
                gan_generator=gan_generator,
                constraints=constraints,
                n_samples=self.config.n_samples,
                max_attempts=self.config.sample_max_attempts
            )
            
            # Save
            np.save(samples_path, self.tailored_samples)
            logger.info(f"Samples saved to {samples_path}")
        
        logger.info(f"Generated {len(self.tailored_samples)} tailored samples")
    
    def step5_extract_modes(self, load_existing: bool = False):
        """Step 5: Mode Extraction via SVD"""
        logger.info("\n[Step 5] Extracting Mode Shapes...")
        
        modes_path = self.config.output_dir / "tailored_modes.npz"
        
        if load_existing and modes_path.exists():
            logger.info(f"Loading existing modes from {modes_path}")
            from mode_extractor import ModeExtractor
            self.extractor = ModeExtractor(n_points=self.config.n_points)
            self.extractor.load_modes(modes_path)
            self.tailored_modes = self.extractor.modes
        else:
            from mode_extractor import ModeExtractor
            
            self.extractor = ModeExtractor(n_points=self.config.n_points)
            
            self.tailored_modes = self.extractor.extract_modes(
                airfoils=self.tailored_samples,
                n_modes=self.config.n_modes
            )
            
            # Save
            self.extractor.save_modes(modes_path)
            logger.info(f"Modes saved to {modes_path}")
            
            # Visualize
            viz_path = self.config.output_dir / "mode_shapes.png"
            try:
                self.extractor.visualize_modes(
                    n_modes_to_show=min(12, self.config.n_modes),
                    save_path=viz_path
                )
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        
        logger.info(f"Extracted {self.tailored_modes.shape[1]} mode shapes")
    
    def load_pretrained_models(self):
        """저장된 모델 로드"""
        logger.info("\nLoading pre-trained models...")
        
        from gan_generator import AirfoilGAN
        from geometric_validator import GeometricValidator
        
        # Load GAN
        self.gan = AirfoilGAN(
            latent_dim=self.config.gan_latent_dim,
            airfoil_dim=self.uiuc_airfoils.shape[1]
        )
        gan_dir = self.config.output_dir / "gan_models"
        self.gan.load_models(gan_dir)
        
        # Load Validator
        self.validator = GeometricValidator(
            airfoil_dim=self.uiuc_airfoils.shape[1]
        )
        validator_path = self.config.output_dir / "validator_model.h5"
        self.validator.load_model(validator_path)
        
        logger.info("Models loaded successfully")
    
    def save_config(self):
        """설정 저장"""
        config_path = self.config.output_dir / "pipeline_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    # 예제 실행
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 설정
    config = PipelineConfig(
        uiuc_data_dir=Path("data/uiuc_airfoils"),
        output_dir=Path("output/tailored_modes"),
        n_points=251,
        gan_epochs=100,  # 테스트용 (실제로는 1000+)
        validator_epochs=50,  # 테스트용
        n_samples=100,  # 테스트용 (실제로는 500)
        n_modes=15
    )
    
    # 파이프라인 생성
    pipeline = TailoredModesPipeline(config)
    
    print("Pipeline configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # 실행 예시 (실제 데이터 없이는 실행 안됨)
    # modes, extractor = pipeline.run_full_pipeline()
    
    print("\nPipeline module loaded successfully!")