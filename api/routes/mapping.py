"""
Mapping routes
"""
from fastapi import APIRouter, HTTPException
from api.schemas.model import MapRequest, MapResponse
from datetime import datetime
from pathlib import Path

router = APIRouter(prefix="/mapping", tags=["mapping"])

# This will be initialized from main app
map_api = None

def init_mapping_routes(map_api_instance):
    """Initialize mapping routes with dependencies"""
    global map_api
    map_api = map_api_instance

@router.post("/map", response_model=MapResponse)
async def create_map(request: MapRequest):
    """Create map with fire detections"""
    if map_api is None:
        raise HTTPException(status_code=500, detail="Map API not initialized")
    
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
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/map/{map_id}")
async def get_map(map_id: str):
    """Get map by ID"""
    # Implement map retrieval logic
    raise HTTPException(status_code=501, detail="Not implemented")

