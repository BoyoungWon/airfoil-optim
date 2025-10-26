# Tailored Airfoil Optimization Platform

Deep learning-based airfoil design optimization using tailored modal parameterization and XFOIL integration.

## Overview

This platform implements a two-phase approach for low Reynolds number airfoil optimization:

- **Phase 1**: Tailored Modal Parameterization using GANs and SVD
- **Phase 2**: XFOIL-based aerodynamic optimization with NURBS representation

## Project Structure

```
project/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI backend server
│   ├── tailored_modes/          # Phase 1: Modal parameterization
│   │   ├── preprocessing.py     # UIUC database preprocessing
│   │   ├── gan_generator.py     # WGAN-GP airfoil generator
│   │   ├── geometric_validator.py # CNN-based validity checker
│   │   ├── optimal_sampler.py   # Constrained sampling
│   │   ├── mode_extractor.py    # SVD mode extraction
│   │   └── pipeline.py          # Full pipeline orchestration
│   ├── xfoil_wrapper.py         # Phase 2: XFOIL interface
│   └── nurbs_airfoil.py         # NURBS airfoil generation
├── debug/
│   ├── index.html               # Debug/test interface
│   ├── style.css                # Interface styling
│   └── debug.js                 # WebSocket & API client
├── data/
│   ├── uiuc_airfoils/          # Input: UIUC airfoil database
│   └── outputs/                 # Output: Generated modes and models
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data directory
mkdir -p data/uiuc_airfoils data/outputs
```

### 2. Run Backend

```bash
cd backend/api
python main.py
```

Backend will start on `http://localhost:8000`

### 3. Open Debug Interface

Open `debug/index.html` in a web browser or serve it:

```bash
cd debug
python -m http.server 8080
```

Then open `http://localhost:8080`

## Usage

### Debug Interface

The debug interface provides step-by-step execution and monitoring:

1. **Step 1: Preprocessing** - Load and normalize UIUC airfoils
2. **Step 2: GAN Training** - Train WGAN for airfoil generation
3. **Step 3: Validator Training** - Train CNN-based geometric validator
4. **Step 4: Sample Generation** - Generate constrained airfoil samples
5. **Step 5: Mode Extraction** - Extract mode shapes via SVD

Each step can be run individually or use "Run Full Phase 1" for sequential execution.

### API Endpoints

#### Status & Health
- `GET /` - API information
- `GET /health` - Health check
- `GET /api/status` - Current pipeline status
- `WS /ws` - WebSocket for real-time updates

#### Phase 1 Operations
- `POST /api/phase1/preprocess` - Run preprocessing
- `POST /api/phase1/train_gan` - Train GAN
  - Parameters: `epochs`, `batch_size`, `latent_dim`
- `POST /api/phase1/train_validator` - Train validator
  - Parameters: `epochs`, `batch_size`
- `POST /api/phase1/generate_samples` - Generate samples
  - Parameters: `n_samples`, `max_thickness`, `min_thickness`
- `POST /api/phase1/extract_modes` - Extract modes
  - Parameters: `n_modes`
- `GET /api/phase1/results` - Get Phase 1 results

#### Utilities
- `GET /api/download/{filename}` - Download generated files
- `POST /api/reset` - Reset pipeline state

## Configuration

### Phase 1 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_points` | 251 | Surface points per airfoil |
| `gan_epochs` | 1000 | GAN training epochs |
| `gan_batch_size` | 32 | GAN batch size |
| `gan_latent_dim` | 100 | Latent vector dimension |
| `validator_epochs` | 100 | Validator training epochs |
| `n_samples` | 500 | Samples to generate |
| `n_modes` | 15 | Number of modes to extract |

### Geometric Constraints

| Constraint | Default | Description |
|------------|---------|-------------|
| `max_thickness` | 0.15 | Maximum thickness ratio (t/c) |
| `min_thickness` | 0.08 | Minimum thickness ratio |
| `min_area` | 0.9 × NACA0015 | Minimum cross-sectional area |

## Data Requirements

### Input
- **UIUC Airfoil Database**: `.dat` files in `data/uiuc_airfoils/`
  - Format: Space-separated x, y coordinates
  - First line: Airfoil name (ignored)

### Output
- `uiuc_processed.npy` - Preprocessed airfoils
- `gan_models/` - Trained GAN models
- `validator_model.h5` - Trained validator model
- `tailored_samples.npy` - Generated samples
- `tailored_modes.npz` - Extracted mode shapes

## Technical Details

### Phase 1: Tailored Modal Parameterization

1. **Preprocessing**: Normalize airfoils to unit chord, resample to 251 points using cosine distribution
2. **GAN**: Wasserstein GAN with gradient penalty (WGAN-GP) for realistic airfoil generation
3. **Validator**: CNN-based discriminator (4 conv layers, 64 filters) for geometric validity
4. **Sampling**: SLSQP optimization to satisfy constraints while maintaining validity
5. **Mode Extraction**: SVD on constraint-satisfied samples to extract modal basis

### Architecture

- **GAN Generator**: CNN with Conv1DTranspose layers, tanh activation
- **GAN Critic**: 4 Conv1D layers, no activation (WGAN)
- **Validator**: 4 Conv1D + Dense layers, sigmoid output

## Development

### Adding New Features

1. Implement in respective module (`backend/tailored_modes/`)
2. Add API endpoint in `backend/api/main.py`
3. Update debug interface if needed

### Testing

```bash
# Run unit tests (if implemented)
pytest tests/

# Manual testing via debug interface
# or direct API calls
curl -X POST http://localhost:8000/api/phase1/preprocess
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- NumPy, SciPy, Matplotlib
- FastAPI, Uvicorn
- XFOIL (for Phase 2)

See `requirements.txt` for complete list.

## References

Based on the paper:
> "Low Reynolds number airfoil design optimization using deep learning-based tailored airfoil modes"
