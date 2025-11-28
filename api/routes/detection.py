"""
Detection routes
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from api.schemas.model import DetectionRequest, DetectionResponse
import cv2
import numpy as np
from datetime import datetime
import time

router = APIRouter(prefix="/detection", tags=["detection"])

# This will be initialized from main app
detector = None
confidence_filter = None
size_classifier = None

def init_detection_routes(det, conf_filter, size_class):
    """Initialize detection routes with dependencies"""
    global detector, confidence_filter, size_classifier
    detector = det
    confidence_filter = conf_filter
    size_classifier = size_class

@router.post("/detect", response_model=DetectionResponse)
async def detect_fire(request: DetectionRequest):
    """Detect fire in image"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
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
        filtered_detections = confidence_filter.filter(detections)
        
        # Classify sizes and convert to response models
        from api.schemas.model import Detection, BoundingBox
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
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=DetectionResponse)
async def detect_fire_upload(file: UploadFile = File(...), confidence_threshold: float = 0.5):
    """Detect fire in uploaded image"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
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
        from api.schemas.model import Detection, BoundingBox
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
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_statistics():
    """Get detection statistics"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        stats = detector.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

