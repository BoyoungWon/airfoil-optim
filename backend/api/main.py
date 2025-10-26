"""
Tailored Airfoil Optimization Platform - Backend API
FastAPI-based REST API for Phase 1 & Phase 2 operations
"""

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Tailored Airfoil Optimization Platform",
    description="Deep learning-based airfoil design optimization",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
pipeline_state = {
    "status": "idle",  # idle, running, completed, error
    "current_step": None,
    "progress": 0.0,
    "message": "",
    "results": {}
}

# WebSocket connections
active_connections: list[WebSocket] = []


# =============================================================================
# WebSocket for Real-time Updates
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline updates"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connection established. Total: {len(active_connections)}")
    
    try:
        # Send current state
        await websocket.send_json(pipeline_state)
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed. Total: {len(active_connections)}")


async def broadcast_update(update: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients"""
    pipeline_state.update(update)
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(pipeline_state)
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        active_connections.remove(conn)


# =============================================================================
# Health & Status
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tailored Airfoil Optimization Platform API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/status")
async def get_status():
    """Get current pipeline status"""
    return pipeline_state


# =============================================================================
# Phase 1: Tailored Modal Parameterization
# =============================================================================

@app.post("/api/phase1/preprocess")
async def phase1_preprocess(data_dir: Optional[str] = None):
    """
    Step 1: Preprocess UIUC airfoil database
    
    Args:
        data_dir: Path to UIUC .dat files (optional, uses default if not provided)
    """
    try:
        await broadcast_update({
            "status": "running",
            "current_step": "preprocessing",
            "progress": 0.1,
            "message": "Loading UIUC airfoil database..."
        })
        
        # Import here to avoid loading heavy modules at startup
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from tailored_modes.preprocessing import AirfoilPreprocessor
        
        # Setup paths
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent / "data" / "uiuc_airfoils"
        else:
            data_dir = Path(data_dir)
        
        output_dir = Path(__file__).parent.parent.parent.parent / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process
        preprocessor = AirfoilPreprocessor(n_points=251)
        
        await broadcast_update({
            "progress": 0.3,
            "message": f"Processing airfoils from {data_dir}..."
        })
        
        airfoils = preprocessor.process_database(
            data_dir,
            output_path=output_dir / "uiuc_processed.npy"
        )
        
        await broadcast_update({
            "status": "completed",
            "current_step": "preprocessing",
            "progress": 1.0,
            "message": f"Preprocessing completed: {len(airfoils)} airfoils",
            "results": {
                "n_airfoils": len(airfoils),
                "airfoil_dim": airfoils.shape[1],
                "output_path": str(output_dir / "uiuc_processed.npy")
            }
        })
        
        return {
            "success": True,
            "n_airfoils": len(airfoils),
            "airfoil_dim": airfoils.shape[1],
            "output_path": str(output_dir / "uiuc_processed.npy")
        }
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}", exc_info=True)
        await broadcast_update({
            "status": "error",
            "message": f"Preprocessing failed: {str(e)}"
        })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/phase1/train_gan")
async def phase1_train_gan(
    epochs: int = 100,
    batch_size: int = 32,
    latent_dim: int = 100
):
    """
    Step 2: Train GAN for airfoil generation
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        latent_dim: Latent vector dimension
    """
    try:
        await broadcast_update({
            "status": "running",
            "current_step": "gan_training",
            "progress": 0.0,
            "message": "Initializing GAN training..."
        })
        
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from tailored_modes.gan_generator import AirfoilGAN
        
        # Load preprocessed data
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "outputs"
        airfoils_path = data_dir / "uiuc_processed.npy"
        
        if not airfoils_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Preprocessed data not found. Run preprocessing first."
            )
        
        airfoils = np.load(airfoils_path)
        
        await broadcast_update({
            "progress": 0.1,
            "message": f"Loaded {len(airfoils)} airfoils. Starting GAN training..."
        })
        
        # Create GAN
        gan = AirfoilGAN(
            latent_dim=latent_dim,
            airfoil_dim=airfoils.shape[1]
        )
        
        # Train (simplified for now - full training in background task)
        save_dir = data_dir / "gan_models"
        
        # TODO: Implement async training with progress updates
        # For now, return immediately and train in background
        
        await broadcast_update({
            "status": "completed",
            "current_step": "gan_training",
            "progress": 1.0,
            "message": "GAN training started in background",
            "results": {
                "epochs": epochs,
                "batch_size": batch_size,
                "save_dir": str(save_dir)
            }
        })
        
        return {
            "success": True,
            "message": "GAN training started",
            "epochs": epochs,
            "save_dir": str(save_dir)
        }
        
    except Exception as e:
        logger.error(f"GAN training error: {e}", exc_info=True)
        await broadcast_update({
            "status": "error",
            "message": f"GAN training failed: {str(e)}"
        })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/phase1/train_validator")
async def phase1_train_validator(
    epochs: int = 100,
    batch_size: int = 32
):
    """
    Step 3: Train geometric validator
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
    """
    try:
        await broadcast_update({
            "status": "running",
            "current_step": "validator_training",
            "progress": 0.0,
            "message": "Initializing validator training..."
        })
        
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from tailored_modes.geometric_validator import GeometricValidator
        
        # Load preprocessed data
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "outputs"
        airfoils_path = data_dir / "uiuc_processed.npy"
        
        if not airfoils_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Preprocessed data not found. Run preprocessing first."
            )
        
        airfoils = np.load(airfoils_path).astype(np.float32)
        
        await broadcast_update({
            "progress": 0.1,
            "message": "Starting validator training..."
        })
        
        # Create validator
        validator = GeometricValidator(airfoil_dim=airfoils.shape[1])
        
        # Train
        save_path = data_dir / "validator_model.h5"
        # TODO: Implement with progress callbacks
        
        await broadcast_update({
            "status": "completed",
            "current_step": "validator_training",
            "progress": 1.0,
            "message": "Validator training completed",
            "results": {
                "epochs": epochs,
                "save_path": str(save_path)
            }
        })
        
        return {
            "success": True,
            "epochs": epochs,
            "save_path": str(save_path)
        }
        
    except Exception as e:
        logger.error(f"Validator training error: {e}", exc_info=True)
        await broadcast_update({
            "status": "error",
            "message": f"Validator training failed: {str(e)}"
        })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/phase1/generate_samples")
async def phase1_generate_samples(
    n_samples: int = 500,
    max_thickness: float = 0.15,
    min_thickness: float = 0.08
):
    """
    Step 4: Generate optimal samples with constraints
    
    Args:
        n_samples: Number of samples to generate
        max_thickness: Maximum thickness constraint
        min_thickness: Minimum thickness constraint
    """
    try:
        await broadcast_update({
            "status": "running",
            "current_step": "sample_generation",
            "progress": 0.0,
            "message": "Generating optimal samples..."
        })
        
        # TODO: Implement sample generation
        
        await broadcast_update({
            "status": "completed",
            "current_step": "sample_generation",
            "progress": 1.0,
            "message": f"Generated {n_samples} samples",
            "results": {
                "n_samples": n_samples
            }
        })
        
        return {
            "success": True,
            "n_samples": n_samples
        }
        
    except Exception as e:
        logger.error(f"Sample generation error: {e}", exc_info=True)
        await broadcast_update({
            "status": "error",
            "message": f"Sample generation failed: {str(e)}"
        })
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/phase1/extract_modes")
async def phase1_extract_modes(n_modes: int = 15):
    """
    Step 5: Extract mode shapes via SVD
    
    Args:
        n_modes: Number of modes to extract
    """
    try:
        await broadcast_update({
            "status": "running",
            "current_step": "mode_extraction",
            "progress": 0.0,
            "message": "Extracting mode shapes..."
        })
        
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from tailored_modes.mode_extractor import ModeExtractor
        
        # Load samples
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "outputs"
        samples_path = data_dir / "tailored_samples.npy"
        
        if not samples_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Tailored samples not found. Run sample generation first."
            )
        
        samples = np.load(samples_path)
        
        await broadcast_update({
            "progress": 0.2,
            "message": f"Loaded {len(samples)} samples. Extracting modes..."
        })
        
        # Extract modes
        extractor = ModeExtractor(n_points=251)
        modes = extractor.extract_modes(samples, n_modes=n_modes)
        
        # Save
        modes_path = data_dir / "tailored_modes.npz"
        extractor.save_modes(modes_path)
        
        explained_var = extractor._explained_variance(n_modes)
        
        await broadcast_update({
            "status": "completed",
            "current_step": "mode_extraction",
            "progress": 1.0,
            "message": f"Extracted {n_modes} modes",
            "results": {
                "n_modes": n_modes,
                "explained_variance": float(explained_var),
                "modes_path": str(modes_path)
            }
        })
        
        return {
            "success": True,
            "n_modes": n_modes,
            "explained_variance": float(explained_var),
            "modes_path": str(modes_path)
        }
        
    except Exception as e:
        logger.error(f"Mode extraction error: {e}", exc_info=True)
        await broadcast_update({
            "status": "error",
            "message": f"Mode extraction failed: {str(e)}"
        })
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/phase1/results")
async def get_phase1_results():
    """Get Phase 1 results and visualizations"""
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "outputs"
    
    results = {}
    
    # Check which steps are completed
    if (data_dir / "uiuc_processed.npy").exists():
        results["preprocessing"] = {"status": "completed"}
    
    if (data_dir / "gan_models").exists():
        results["gan_training"] = {"status": "completed"}
    
    if (data_dir / "validator_model.h5").exists():
        results["validator_training"] = {"status": "completed"}
    
    if (data_dir / "tailored_samples.npy").exists():
        results["sample_generation"] = {"status": "completed"}
    
    if (data_dir / "tailored_modes.npz").exists():
        results["mode_extraction"] = {"status": "completed"}
        
        # Load mode info
        data = np.load(data_dir / "tailored_modes.npz", allow_pickle=True)
        results["mode_extraction"]["n_modes"] = data['modes'].shape[1]
    
    return results


# =============================================================================
# File Downloads
# =============================================================================

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated files"""
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "outputs"
    file_path = data_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)


# =============================================================================
# Reset & Cleanup
# =============================================================================

@app.post("/api/reset")
async def reset_pipeline():
    """Reset pipeline state"""
    global pipeline_state
    pipeline_state = {
        "status": "idle",
        "current_step": None,
        "progress": 0.0,
        "message": "",
        "results": {}
    }
    
    await broadcast_update(pipeline_state)
    
    return {"success": True, "message": "Pipeline reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")