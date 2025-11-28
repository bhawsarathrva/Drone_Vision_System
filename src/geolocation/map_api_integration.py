"""
Map API integration for fire detection visualization
"""
import folium
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger

class MapAPI:
    def __init__(self, default_location: Tuple[float, float] = (22.7196, 75.8577)):
        """
        Initialize map API
        
        Args:
            default_location: Default map center (latitude, longitude)
        """
        self.default_location = default_location
    
    def create_map(
        self,
        center: Optional[Tuple[float, float]] = None,
        zoom_start: int = 10
    ) -> folium.Map:
        """
        Create Folium map
        
        Args:
            center: Map center (lat, lon)
            zoom_start: Initial zoom level
            
        Returns:
            Folium map object
        """
        center = center or self.default_location
        
        map_obj = folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        return map_obj
    
    def add_fire_marker(
        self,
        map_obj: folium.Map,
        latitude: float,
        longitude: float,
        popup_text: str = "Fire Detection",
        color: str = "red",
        icon: str = "fire"
    ):
        """
        Add fire detection marker to map
        
        Args:
            map_obj: Folium map object
            latitude: Fire latitude
            longitude: Fire longitude
            popup_text: Popup text
            color: Marker color
            icon: Marker icon
        """
        folium.Marker(
            location=[latitude, longitude],
            popup=popup_text,
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(map_obj)
    
    def add_drone_marker(
        self,
        map_obj: folium.Map,
        latitude: float,
        longitude: float,
        heading: float = 0.0,
        popup_text: str = "Drone"
    ):
        """
        Add drone marker to map
        
        Args:
            map_obj: Folium map object
            latitude: Drone latitude
            longitude: Drone longitude
            heading: Drone heading
            popup_text: Popup text
        """
        # Create custom icon with heading
        folium.Marker(
            location=[latitude, longitude],
            popup=popup_text,
            icon=folium.Icon(color='blue', icon='plane', prefix='fa'),
            rotation_angle=heading
        ).add_to(map_obj)
    
    def add_fire_polygon(
        self,
        map_obj: folium.Map,
        coordinates: List[Tuple[float, float]],
        popup_text: str = "Fire Area",
        color: str = "red",
        fill_color: str = "red",
        fill_opacity: float = 0.3
    ):
        """
        Add fire area polygon to map
        
        Args:
            map_obj: Folium map object
            coordinates: List of (lat, lon) coordinates
            popup_text: Popup text
            color: Border color
            fill_color: Fill color
            fill_opacity: Fill opacity
        """
        folium.Polygon(
            locations=coordinates,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=fill_opacity
        ).add_to(map_obj)
    
    def add_path(
        self,
        map_obj: folium.Map,
        coordinates: List[Tuple[float, float]],
        color: str = "blue",
        weight: int = 3
    ):
        """
        Add path to map
        
        Args:
            map_obj: Folium map object
            coordinates: List of (lat, lon) coordinates
            color: Line color
            weight: Line weight
        """
        folium.PolyLine(
            locations=coordinates,
            color=color,
            weight=weight
        ).add_to(map_obj)
    
    def save_map(self, map_obj: folium.Map, output_path: str):
        """Save map to HTML file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        map_obj.save(output_path)
        logger.info(f"Map saved to {output_path}")
    
    def create_fire_detection_map(
        self,
        fire_detections: List[Dict],
        drone_position: Optional[Tuple[float, float]] = None,
        output_path: str = "outputs/fire_map.html"
    ) -> folium.Map:
        """
        Create map with fire detections
        
        Args:
            fire_detections: List of fire detection dictionaries with GPS coordinates
            drone_position: Drone position (lat, lon)
            output_path: Output HTML file path
            
        Returns:
            Folium map object
        """
        # Determine map center
        if fire_detections:
            center = (
                sum(det.get('latitude', 0) for det in fire_detections) / len(fire_detections),
                sum(det.get('longitude', 0) for det in fire_detections) / len(fire_detections)
            )
        elif drone_position:
            center = drone_position
        else:
            center = self.default_location
        
        # Create map
        map_obj = self.create_map(center=center, zoom_start=15)
        
        # Add fire detections
        for i, det in enumerate(fire_detections):
            lat = det.get('latitude')
            lon = det.get('longitude')
            
            if lat and lon:
                popup_text = f"Fire Detection #{i+1}<br>"
                popup_text += f"Confidence: {det.get('confidence', 0):.2f}<br>"
                popup_text += f"Size: {det.get('size', 'unknown')}<br>"
                popup_text += f"Time: {det.get('timestamp', 'unknown')}"
                
                self.add_fire_marker(
                    map_obj,
                    lat,
                    lon,
                    popup_text=popup_text,
                    color="red"
                )
        
        # Add drone position
        if drone_position:
            self.add_drone_marker(
                map_obj,
                drone_position[0],
                drone_position[1],
                popup_text="Drone Position"
            )
        
        # Save map
        self.save_map(map_obj, output_path)
        
        return map_obj

