import argparse
import sys
import os
import yaml
import threading
import time
from pathlib import Path
from typing import Optional

# Setup logger first (before checking PyTorch)
try:
    from loguru import logger
    from src.utils.logger import setup_logger
    setup_logger(log_level="INFO", log_file="outputs/logs/fire_detection.log")
except:
    # Fallback if logger not available
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Check for PyTorch before importing other modules
def check_pytorch():
    """Check if PyTorch is properly installed"""
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        return True
    except OSError as e:
        if "DLL" in str(e) or "c10.dll" in str(e):
            print("\n" + "="*70)
            print("ERROR: PyTorch DLL Loading Failed")
            print("="*70)
            print("\nThis is a common Windows issue. Try the following solutions:\n")
            print("1. Reinstall PyTorch (CPU version):")
            print("   pip uninstall torch torchvision")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            print("\n2. Install Visual C++ Redistributables:")
            print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("\n3. If using CUDA, reinstall CUDA version:")
            print("   pip uninstall torch torchvision")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("\n4. Check antivirus - it may be blocking DLL files")
            print("\n5. Try running as Administrator")
            print("="*70 + "\n")
            return False
        else:
            print(f"\nERROR: PyTorch import failed: {e}\n")
            print("Please install PyTorch: pip install torch torchvision\n")
            return False
    except ImportError as e:
        print(f"\nERROR: PyTorch not installed: {e}\n")
        print("Please install PyTorch: pip install torch torchvision\n")
        return False
    except Exception as e:
        print(f"\nERROR: Unexpected error importing PyTorch: {e}\n")
        return False

if not check_pytorch():
    sys.exit(1)

# Now safe to import other modules
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.realtime.realtime_detector import RealtimeDetector
    from src.utils.alert_system import AlertSystem
    from src.geolocation.coordinate_mapper import CoordinateMapper
    from src.geolocation.gps_handler import GPSHandler
    from src.geolocation.map_api_integration import MapAPI
    from src.drone.drone_controller import DroneController, DroneType
    from src.utils.visualization import draw_detections
    from src.utils.metrics import DetectionMetrics
    import uvicorn
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class FireDetectionSystem:
    """Main fire detection system that integrates all components"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize fire detection system
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logger
        log_file = self.config.get('logging', {}).get('file', 'outputs/logs/fire_detection.log')
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        setup_logger(log_level=log_level, log_file=log_file)
        
        # Initialize components
        self.realtime_detector = None
        self.alert_system = None
        self.gps_handler = None
        self.coordinate_mapper = None
        self.map_api = None
        self.metrics = DetectionMetrics()
        
        # System state
        self.running = False
        self.api_thread = None
        self.realtime_thread = None
        self._display_available = None
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return config
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Return default configuration"""
        return {
            'model': {'path': 'Model/best.pt', 'confidence_threshold': 0.5},
            'camera': {'source': '0', 'fps': 30},
            'drone': {'type': 'simulator', 'connection_string': 'udp:127.0.0.1:14550'},
            'alerts': {'threshold': 1, 'cooldown_period': 60},
            'api': {'host': '0.0.0.0', 'port': 8000},
            'logging': {'level': 'INFO', 'file': 'outputs/logs/fire_detection.log'}
        }
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Fire Detection System...")
            
            # Get configuration
            model_config = self.config.get('model', {})
            camera_config = self.config.get('camera', {})
            drone_config = self.config.get('drone', {})
            alerts_config = self.config.get('alerts', {})
            
            # Initialize real-time detector
            model_path = model_config.get('path', 'Model/best.pt')
            confidence_threshold = model_config.get('confidence_threshold', 0.5)
            camera_source = camera_config.get('source', '0')
            fps = camera_config.get('fps', 30)
            
            self.realtime_detector = RealtimeDetector(
                model_path=model_path,
                camera_source=camera_source,
                confidence_threshold=confidence_threshold,
                fps=fps,
                callback=self._on_detection
            )
            
            # Initialize alert system
            alert_threshold = alerts_config.get('threshold', 1)
            cooldown_period = alerts_config.get('cooldown_period', 60)
            self.alert_system = AlertSystem(
                alert_threshold=alert_threshold,
                cooldown_period=cooldown_period
            )
            
            # Initialize GPS handler and coordinate mapper
            self.gps_handler = GPSHandler()
            self.coordinate_mapper = CoordinateMapper()
            
            # Initialize map API
            geolocation_config = self.config.get('geolocation', {})
            default_location = geolocation_config.get('default_location', {})
            default_lat = default_location.get('latitude', 22.7196)
            default_lon = default_location.get('longitude', 75.8577)
            self.map_api = MapAPI(default_location=(default_lat, default_lon))
            
            # Connect drone if configured
            drone_type_str = drone_config.get('type', 'simulator')
            drone_type = self._get_drone_type(drone_type_str)
            connection_string = drone_config.get('connection_string', 'udp:127.0.0.1:14550')
            
            if drone_type:
                self.realtime_detector.connect_drone(
                    drone_type=drone_type,
                    connection_string=connection_string
                )
                self.gps_handler.drone_controller = self.realtime_detector.drone
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def _get_drone_type(self, drone_type_str: str) -> Optional[DroneType]:
        """Convert string to DroneType enum"""
        drone_type_map = {
            'mavlink': DroneType.MAVLINK,
            'dji': DroneType.DJI,
            'custom': DroneType.CUSTOM,
            'simulator': DroneType.SIMULATOR
        }
        return drone_type_map.get(drone_type_str.lower())
    
    def _on_detection(self, detections: list, frame, telemetry: Optional[dict]):
        """Callback when fire is detected (called only when detections are found)"""
        try:
            if not detections:
                return
            
            # Update metrics
            fps = self.realtime_detector.get_statistics().get('fps', 0) if self.realtime_detector else 0
            self.metrics.update(detections, fps=fps)
            
            # Get GPS data if available
            gps_data = None
            if telemetry and self.gps_handler:
                self.gps_handler.update_gps(
                    latitude=telemetry.get('latitude', 0),
                    longitude=telemetry.get('longitude', 0),
                    altitude=telemetry.get('altitude', 0),
                    heading=telemetry.get('heading', 0),
                    speed=telemetry.get('ground_speed', 0),
                    satellite_count=telemetry.get('satellite_count', 0),
                    fix_type=telemetry.get('gps_fix_type', 0)
                )
                gps_data = self.gps_handler.get_current_gps()
            
            # Convert detections to GPS coordinates if GPS is available
            if gps_data and detections and frame is not None:
                frame_height, frame_width = frame.shape[:2]
                for det in detections:
                    bbox = det['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Map pixel to GPS
                    geolocation_config = self.config.get('geolocation', {})
                    fov_h = geolocation_config.get('camera_fov_horizontal', 60.0)
                    fov_v = geolocation_config.get('camera_fov_vertical', 45.0)
                    
                    try:
                        lat, lon = self.coordinate_mapper.pixel_to_gps(
                            pixel_x=center_x,
                            pixel_y=center_y,
                            image_width=frame_width,
                            image_height=frame_height,
                            drone_lat=gps_data.latitude,
                            drone_lon=gps_data.longitude,
                            drone_altitude=gps_data.altitude,
                            drone_heading=gps_data.heading,
                            camera_fov_horizontal=fov_h,
                            camera_fov_vertical=fov_v
                        )
                        
                        det['latitude'] = lat
                        det['longitude'] = lon
                        det['timestamp'] = time.time()
                    except Exception as e:
                        logger.warning(f"Failed to map pixel to GPS: {e}")
            
            # Check for alerts
            metadata = {}
            if gps_data:
                metadata['latitude'] = gps_data.latitude
                metadata['longitude'] = gps_data.longitude
                metadata['altitude'] = gps_data.altitude
                metadata['heading'] = gps_data.heading
            
            alert_triggered = self.alert_system.check_detections(detections, metadata)
            
            if alert_triggered:
                logger.warning(f"🔥 FIRE ALERT: {len(detections)} fire(s) detected!")
                for det in detections:
                    logger.warning(f"  - Confidence: {det['confidence']:.2f}, "
                                 f"Size: {det.get('size', 'unknown')}, "
                                 f"Location: [{det.get('latitude', 'N/A')}, {det.get('longitude', 'N/A')}]")
        
        except Exception as e:
            logger.error(f"Error in detection callback: {e}")
    
    def _check_display_available(self) -> bool:
        """Check if display is available (for GUI)"""
        if self._display_available is not None:
            return self._display_available
        
        try:
            # Try to create a test window (headless check)
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('test', test_img)
            cv2.waitKey(1)
            cv2.destroyWindow('test')
            self._display_available = True
            logger.info("Display available - GUI mode enabled")
        except cv2.error:
            self._display_available = False
            logger.info("No display available - running in headless mode")
        except Exception:
            self._display_available = False
            logger.info("Display check failed - running in headless mode")
        
        return self._display_available
    
    def _realtime_detection_loop(self, show_video: bool = True):
        """Real-time detection loop (runs in separate thread)"""
        try:
            logger.info("Real-time detection loop started")
            
            # Check if display is available
            display_available = self._check_display_available() if show_video else False
            if show_video and not display_available:
                logger.warning("Display not available, running in headless mode")
                show_video = False
            
            while self.running:
                try:
                    # Get current frame and detections
                    frame = self.realtime_detector.get_frame()
                    detections = self.realtime_detector.get_detections()
                    
                    if frame is not None:
                        # Draw detections on frame
                        if detections:
                            frame = draw_detections(frame, detections)
                        
                        # Draw statistics
                        stats = self.realtime_detector.get_statistics()
                        stats_text = f"FPS: {stats.get('fps', 0):.1f} | "
                        stats_text += f"Detections: {stats.get('detection_count', 0)} | "
                        stats_text += f"Frames: {stats.get('frame_count', 0)}"
                        
                        cv2.putText(frame, stats_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show video if requested and display is available
                        if show_video and display_available:
                            try:
                                cv2.imshow('Fire Detection System', frame)
                                # Check for quit
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    logger.info("Quit requested by user")
                                    self.running = False
                                    break
                            except cv2.error as e:
                                logger.warning(f"Display error: {e}, switching to headless mode")
                                display_available = False
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error in real-time detection loop: {e}")
                    time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Fatal error in real-time detection loop: {e}")
        finally:
            if display_available:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            logger.info("Real-time detection loop stopped")
    
    def start_realtime_detection(self, show_video: bool = True, run_in_thread: bool = False):
        """Start real-time fire detection
        
        Args:
            show_video: Whether to show video window
            run_in_thread: If True, run detection loop in separate thread (for simultaneous execution)
        """
        if not self.realtime_detector:
            logger.error("Components not initialized. Call initialize_components() first.")
            return False
        
        logger.info("Starting real-time fire detection...")
        
        if not self.realtime_detector.start():
            logger.error("Failed to start real-time detector")
            return False
        
        self.running = True
        
        if run_in_thread:
            # Run in separate thread for simultaneous execution
            self.realtime_thread = threading.Thread(
                target=self._realtime_detection_loop,
                args=(show_video,),
                daemon=False
            )
            self.realtime_thread.start()
            logger.info("Real-time detection running in background thread")
            return True
        else:
            # Run in current thread (blocking)
            try:
                self._realtime_detection_loop(show_video)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                self.stop()
            return True
    
    def start_api_server(self, host: str = None, port: int = None):
        """Start API server in a separate thread"""
        api_config = self.config.get('api', {})
        host = host or api_config.get('host', '0.0.0.0')
        port = port or api_config.get('port', 8000)
        
        def run_api():
            try:
                uvicorn.run(
                    "api.app:app",
                    host=host,
                    port=port,
                    log_level="info"
                )
            except Exception as e:
                logger.error(f"API server error: {e}")
        
        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.api_thread.start()
        logger.info(f"API server started on http://{host}:{port}")
    
    def stop(self):
        """Stop the fire detection system"""
        logger.info("Stopping Fire Detection System...")
        self.running = False
        
        # Wait for real-time detection thread to finish
        if self.realtime_thread and self.realtime_thread.is_alive():
            logger.info("Waiting for real-time detection thread to finish...")
            self.realtime_thread.join(timeout=5.0)
            if self.realtime_thread.is_alive():
                logger.warning("Real-time detection thread did not stop gracefully")
        
        # Stop detector
        if self.realtime_detector:
            self.realtime_detector.stop()
        
        # Clean up display
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("Fire Detection System stopped")
    
    def get_statistics(self) -> dict:
        """Get system statistics"""
        stats = {}
        
        if self.realtime_detector:
            stats['detection'] = self.realtime_detector.get_statistics()
        
        if self.metrics:
            stats['metrics'] = self.metrics.get_statistics()
        
        if self.alert_system:
            stats['alerts'] = self.alert_system.get_statistics()
        
        return stats

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fire Detection System')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['realtime', 'api', 'both'],
                       help='Operation mode: realtime, api, or both')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model weights (overrides config)')
    parser.add_argument('--camera', type=str, default=None,
                       help='Camera source (overrides config)')
    parser.add_argument('--no-video', action='store_true',
                       help='Don\'t show video window')
    parser.add_argument('--api-host', type=str, default=None,
                       help='API server host (overrides config)')
    parser.add_argument('--api-port', type=int, default=None,
                       help='API server port (overrides config)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = FireDetectionSystem(config_path=args.config)
    
    # Override config with command line arguments
    if args.model:
        system.config['model']['path'] = args.model
    if args.camera:
        system.config['camera']['source'] = args.camera
    
    # Initialize components
    if not system.initialize_components():
        logger.error("Failed to initialize system")
        sys.exit(1)
    
    try:
        # Start API server if requested
        if args.mode in ['api', 'both']:
            system.start_api_server(host=args.api_host, port=args.api_port)
            # Give API server time to start
            time.sleep(2)
            
            if args.mode == 'api':
                # Just run API server
                logger.info("="*70)
                logger.info("Fire Detection API Server Running")
                logger.info("="*70)
                logger.info(f"API available at: http://{args.api_host or '0.0.0.0'}:{args.api_port or 8000}")
                logger.info("API Documentation: http://{}/docs".format(args.api_host or '0.0.0.0' if args.api_host else 'localhost'))
                logger.info("Press Ctrl+C to stop.")
                logger.info("="*70)
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping API server...")
                return
        
        # Start real-time detection if requested
        if args.mode in ['realtime', 'both']:
            show_video = not args.no_video
            # Run in thread if both modes are active (for simultaneous execution)
            run_in_thread = (args.mode == 'both')
            system.start_realtime_detection(show_video=show_video, run_in_thread=run_in_thread)
        
        # Main execution loop - keep all services running
        if args.mode == 'both':
            logger.info("="*70)
            logger.info("Fire Detection System Running - All Services Active")
            logger.info("="*70)
            logger.info(f"API Server: http://{args.api_host or '0.0.0.0'}:{args.api_port or 8000}")
            logger.info("Real-time Detection: Active")
            logger.info("Press Ctrl+C to stop all services.")
            logger.info("="*70)
            
            try:
                # Keep main thread alive and monitor services
                while system.running:
                    # Check if threads are still alive
                    if system.api_thread and not system.api_thread.is_alive():
                        logger.error("API server thread died unexpectedly!")
                        system.running = False
                        break
                    
                    if system.realtime_thread and not system.realtime_thread.is_alive():
                        logger.error("Real-time detection thread died unexpectedly!")
                        system.running = False
                        break
                    
                    # Log periodic status
                    if system.realtime_detector:
                        stats = system.realtime_detector.get_statistics()
                        logger.debug(f"Status - FPS: {stats.get('fps', 0):.1f}, "
                                   f"Frames: {stats.get('frame_count', 0)}, "
                                   f"Detections: {stats.get('detection_count', 0)}")
                    
                    time.sleep(5)  # Check every 5 seconds
                    
            except KeyboardInterrupt:
                logger.info("Stopping all services...")
                system.stop()
        elif args.mode == 'realtime':
            # Real-time only mode - already handled in start_realtime_detection
            pass
    
    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        system.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()

