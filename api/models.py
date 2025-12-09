"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional


class TryOnRequest(BaseModel):
    """Request model for try-on endpoint"""
    garment_description: Optional[str] = Field(None, description="Garment description")
    denoise_steps: int = Field(30, ge=10, le=50, description="Number of denoising steps")
    seed: Optional[int] = Field(None, description="Random seed")
    auto_mask: bool = Field(True, description="Auto-generate mask")
    crop_image: bool = Field(False, description="Crop image")


class TryOnResponse(BaseModel):
    """Response model for try-on endpoint"""
    success: bool = Field(..., description="Whether the operation succeeded")
    result_image: str = Field(..., description="Base64-encoded result image (PNG)")
    mask_image: str = Field(..., description="Base64-encoded mask image (PNG)")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message")

