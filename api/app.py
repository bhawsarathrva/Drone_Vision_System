"""
FastAPI application for fire detection system
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from datetime import datetime
from typing import List
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas.model import (
    DetectionRequest, DetectionResponse, Detection, BoundingBox,
    AlertRequest, AlertResponse,
    TelemetryData,
    MapRequest, MapResponse,
    HealthResponse
)
from src.detection.fire_detector import FireDetector
from src.detection.confidence_filter import ConfidenceFilter
from src.detection.size_classifier import SizeClassifier
from src.utils.alert_system import AlertSystem
from src.geolocation.map_api_integration import MapAPI
from src.utils.logger import setup_logger
from loguru import logger

# Setup logger
setup_logger(log_file="outputs/logs/api.log")

# Initialize FastAPI app
app = FastAPI(
    title="Fire Detection API",
    description="API for real-time fire detection using drones",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = None
confidence_filter = None
size_classifier = None
alert_system = None
map_api = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global detector, confidence_filter, size_classifier, alert_system, map_api
    
    try:
        # Initialize detector
        model_path = os.getenv("MODEL_PATH", "Model/best.pt")
        detector = FireDetector(model_path=model_path, confidence_threshold=0.5)
        logger.info("Fire detector initialized")
        
        # Initialize filters
        confidence_filter = ConfidenceFilter(min_confidence=0.5)
        size_classifier = SizeClassifier()
        
        # Initialize alert system
        alert_system = AlertSystem(alert_threshold=1, cooldown_period=60)
        
        # Initialize map API
        map_api = MapAPI()
        
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("API shutdown")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=0.0
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=0.0
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_fire(request: DetectionRequest):
    """Detect fire in image"""
    try:
        import time
        start_time = time.time()
        
        # Load image
        if request.image_path:
            image = cv2.imread(request.image_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to load image")
        elif request.image_url:
            import requests
            response = requests.get(request.image_url)
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Failed to load image from URL")
        else:
            raise HTTPException(status_code=400, detail="Either image_path or image_url must be provided")
        
        # Update confidence threshold
        if request.confidence_threshold:
            detector.update_confidence_threshold(request.confidence_threshold)
            confidence_filter.update_threshold(request.confidence_threshold)
        
        # Detect fire
        detections = detector.detect(image)
        
        # Filter by confidence
        filtered_detections = confidence_filter.filter(detections)
        
        # Classify sizes
        detection_models = []
        for det in filtered_detections:
            size = size_classifier.classify(det, image.shape[1], image.shape[0])
            det['size'] = size.value
            
            detection_models.append(Detection(
                bbox=BoundingBox(
                    x1=det['bbox'][0],
                    y1=det['bbox'][1],
                    x2=det['bbox'][2],
                    y2=det['bbox'][3]
                ),
                confidence=det['confidence'],
                class_id=det['class'],
                class_name=det['class_name'],
                size=det['size']
            ))
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            detections=detection_models,
            frame_count=detector.total_frames,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/upload", response_model=DetectionResponse)
async def detect_fire_upload(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    """Detect fire in uploaded image"""
    try:
        import time
        start_time = time.time()
        
        # Read image from upload
        contents = await file.read()
        image_array = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Update confidence threshold
        detector.update_confidence_threshold(confidence_threshold)
        confidence_filter.update_threshold(confidence_threshold)
        
        # Detect fire
        detections = detector.detect(image)
        filtered_detections = confidence_filter.filter(detections)
        
        # Classify sizes
        detection_models = []
        for det in filtered_detections:
            size = size_classifier.classify(det, image.shape[1], image.shape[0])
            det['size'] = size.value
            
            detection_models.append(Detection(
                bbox=BoundingBox(
                    x1=det['bbox'][0],
                    y1=det['bbox'][1],
                    x2=det['bbox'][2],
                    y2=det['bbox'][3]
                ),
                confidence=det['confidence'],
                class_id=det['class'],
                class_name=det['class_name'],
                size=det['size']
            ))
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            detections=detection_models,
            frame_count=detector.total_frames,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts", response_model=AlertResponse)
async def create_alert(request: AlertRequest):
    """Create fire alert"""
    try:
        # Convert detections to dict format
        detections_dict = []
        for det in request.detections:
            detections_dict.append({
                'bbox': [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                'confidence': det.confidence,
                'class': det.class_id,
                'class_name': det.class_name,
                'size': det.size
            })
        
        # Create metadata
        metadata = {}
        if request.latitude and request.longitude:
            metadata['latitude'] = request.latitude
            metadata['longitude'] = request.longitude
        if request.altitude:
            metadata['altitude'] = request.altitude
        if request.timestamp:
            metadata['timestamp'] = request.timestamp.isoformat()
        
        # Trigger alert
        alert_triggered = alert_system.trigger_alert(detections_dict, metadata)
        
        if alert_triggered:
            alerts = alert_system.get_alerts(limit=1)
            alert = alerts[0] if alerts else None
            
            return AlertResponse(
                alert_id=alert['id'] if alert else 0,
                status="success",
                message="Alert created successfully",
                timestamp=datetime.now()
            )
        else:
            return AlertResponse(
                alert_id=0,
                status="failed",
                message="Failed to create alert",
                timestamp=datetime.now()
            )
            
    except Exception as e:
        logger.error(f"Alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts", response_model=List[dict])
async def get_alerts(limit: int = 10):
    """Get alerts"""
    try:
        alerts = alert_system.get_alerts(limit=limit)
        return alerts
    except Exception as e:
        logger.error(f"Get alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/map", response_model=MapResponse)
async def create_map(request: MapRequest):
    """Create map with fire detections"""
    try:
        # Convert detections to dict format
        fire_detections = []
        for det in request.fire_detections:
            fire_detections.append({
                'latitude': request.drone_position.get('latitude', 0) if request.drone_position else 0,
                'longitude': request.drone_position.get('longitude', 0) if request.drone_position else 0,
                'confidence': det.confidence,
                'size': det.size,
                'timestamp': datetime.now().isoformat()
            })
        
        # Get drone position
        drone_position = None
        if request.drone_position:
            drone_position = (
                request.drone_position.get('latitude', 0),
                request.drone_position.get('longitude', 0)
            )
        elif request.center_latitude and request.center_longitude:
            drone_position = (request.center_latitude, request.center_longitude)
        
        # Create map
        output_path = f"outputs/fire_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        map_api.create_fire_detection_map(
            fire_detections=fire_detections,
            drone_position=drone_position,
            output_path=output_path
        )
        
        return MapResponse(
            map_url=f"/static/{Path(output_path).name}",
            fire_count=len(fire_detections),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Map creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get detection statistics"""
    try:
        stats = detector.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run API server"""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()

