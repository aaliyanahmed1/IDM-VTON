"""
Production-ready FastAPI server for IDM-VTON mobile integration
"""
import os
import sys
import logging
import base64
import io
from typing import Optional
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Also add current directory for relative imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import from api package (works when running from root) or directly (when running from api/)
try:
    # Try absolute import first (when running from repository root)
    from api.service_optimized import OptimizedTryOnService as TryOnService
    from api.models import TryOnRequest, TryOnResponse, ErrorResponse
except ImportError:
    try:
        # Fallback to relative import (when running from api/ directory)
        from service_optimized import OptimizedTryOnService as TryOnService
        from models import TryOnRequest, TryOnResponse, ErrorResponse
    except ImportError:
        # Last resort: direct import with path manipulation
        import importlib.util
        spec = importlib.util.spec_from_file_location("service_optimized", Path(__file__).parent / "service_optimized.py")
        service_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service_module)
        TryOnService = service_module.OptimizedTryOnService
        
        spec = importlib.util.spec_from_file_location("models", Path(__file__).parent / "models.py")
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        TryOnRequest = models_module.TryOnRequest
        TryOnResponse = models_module.TryOnResponse
        ErrorResponse = models_module.ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IDM-VTON API",
    description="Production-ready Virtual Try-On API for Mobile Applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service (lazy loading)
tryon_service: Optional[TryOnService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global tryon_service
    try:
        logger.info("Initializing IDM-VTON service...")
        tryon_service = TryOnService()
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global tryon_service
    if tryon_service:
        tryon_service.cleanup()
    logger.info("Service shutdown complete")


@app.get("/", response_model=dict)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "IDM-VTON API",
        "version": "1.0.0"
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Detailed health check"""
    global tryon_service
    if tryon_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "service_ready": tryon_service is not None
    }


@app.post("/api/v1/tryon", response_model=TryOnResponse)
async def virtual_tryon(
    person_image: UploadFile = File(..., description="Person image file"),
    garment_image: UploadFile = File(..., description="Garment image file"),
    garment_description: Optional[str] = Field(None, description="Optional garment description"),
    denoise_steps: int = Field(30, ge=10, le=50, description="Number of denoising steps"),
    seed: Optional[int] = Field(None, description="Random seed for reproducibility"),
    auto_mask: bool = Field(True, description="Automatically generate mask using pose estimation"),
    crop_image: bool = Field(False, description="Crop image to optimal aspect ratio")
):
    """
    Generate virtual try-on image
    
    - **person_image**: Image of the person (JPEG/PNG)
    - **garment_image**: Image of the garment (JPEG/PNG)
    - **garment_description**: Optional text description of the garment
    - **denoise_steps**: Number of diffusion steps (10-50, default: 30)
    - **seed**: Random seed for reproducibility
    - **auto_mask**: Use automatic pose-based masking (default: True)
    - **crop_image**: Crop to optimal aspect ratio (default: False)
    
    Returns base64-encoded result image and mask
    """
    global tryon_service
    
    if tryon_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Validate file types
        if not person_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="person_image must be an image file")
        if not garment_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="garment_image must be an image file")
        
        # Read images
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        person_img = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        garment_img = Image.open(io.BytesIO(garment_bytes)).convert("RGB")
        
        logger.info(f"Processing try-on request: person_size={person_img.size}, garment_size={garment_img.size}")
        
        # Process try-on
        result_image, mask_image = await tryon_service.process_tryon(
            person_image=person_img,
            garment_image=garment_img,
            garment_description=garment_description or "clothing",
            denoise_steps=denoise_steps,
            seed=seed,
            auto_mask=auto_mask,
            crop_image=crop_image
        )
        
        # Convert to base64
        result_buffer = io.BytesIO()
        result_image.save(result_buffer, format="PNG")
        result_base64 = base64.b64encode(result_buffer.getvalue()).decode('utf-8')
        
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        
        return TryOnResponse(
            success=True,
            result_image=result_base64,
            mask_image=mask_base64,
            message="Try-on completed successfully"
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory")
        raise HTTPException(status_code=507, detail="GPU out of memory. Please try with smaller images.")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/tryon/async", response_model=dict)
async def virtual_tryon_async(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_description: Optional[str] = None,
    denoise_steps: int = 30,
    seed: Optional[int] = None,
    auto_mask: bool = True,
    crop_image: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Async virtual try-on endpoint (for long-running requests)
    Returns job ID immediately, result available via /api/v1/jobs/{job_id}
    """
    # TODO: Implement job queue system (Redis/Celery)
    raise HTTPException(status_code=501, detail="Async endpoint not yet implemented")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Single worker for GPU models
    )

