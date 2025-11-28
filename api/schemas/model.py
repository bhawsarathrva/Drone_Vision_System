"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class DetectionRequest(BaseModel):
    """Request model for detection"""
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class BoundingBox(BaseModel):
    """Bounding box model"""
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    """Detection model"""
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int
    class_name: str
    size: Optional[str] = None

class DetectionResponse(BaseModel):
    """Response model for detection"""
    detections: List[Detection]
    frame_count: int
    processing_time: float
    timestamp: datetime

class AlertRequest(BaseModel):
    """Request model for alert"""
    detections: List[Detection]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    timestamp: Optional[datetime] = None

class AlertResponse(BaseModel):
    """Response model for alert"""
    alert_id: int
    status: str
    message: str
    timestamp: datetime

class TelemetryData(BaseModel):
    """Telemetry data model"""
    latitude: float
    longitude: float
    altitude: float
    heading: float
    ground_speed: float
    battery_percent: float
    armed: bool
    mode: str
    satellite_count: int
    timestamp: datetime

class MapRequest(BaseModel):
    """Request model for map generation"""
    fire_detections: List[Detection]
    drone_position: Optional[Dict[str, float]] = None
    center_latitude: Optional[float] = None
    center_longitude: Optional[float] = None

class MapResponse(BaseModel):
    """Response model for map generation"""
    map_url: str
    fire_count: int
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime: float

