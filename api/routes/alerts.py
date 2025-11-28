"""
Alert routes
"""
from fastapi import APIRouter, HTTPException
from api.schemas.model import AlertRequest, AlertResponse
from typing import List
from datetime import datetime

router = APIRouter(prefix="/alerts", tags=["alerts"])

# This will be initialized from main app
alert_system = None

def init_alert_routes(alert_sys):
    """Initialize alert routes with dependencies"""
    global alert_system
    alert_system = alert_sys

@router.post("", response_model=AlertResponse)
async def create_alert(request: AlertRequest):
    """Create fire alert"""
    if alert_system is None:
        raise HTTPException(status_code=500, detail="Alert system not initialized")
    
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
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=List[dict])
async def get_alerts(limit: int = 10):
    """Get alerts"""
    if alert_system is None:
        raise HTTPException(status_code=500, detail="Alert system not initialized")
    
    try:
        alerts = alert_system.get_alerts(limit=limit)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_alert_statistics():
    """Get alert statistics"""
    if alert_system is None:
        raise HTTPException(status_code=500, detail="Alert system not initialized")
    
    try:
        stats = alert_system.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

