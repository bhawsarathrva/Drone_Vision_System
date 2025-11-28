"""
Drone controller interface for various drone types
Supports MAVLink (ArduPilot, PX4), DJI SDK, and custom protocols
"""
import time
import threading
import math
from typing import Dict, Optional, Tuple
from enum import Enum
from loguru import logger

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    logger.warning("pymavlink not installed. MAVLink support disabled.")

class DroneType(Enum):
    MAVLINK = "mavlink"  # ArduPilot, PX4
    DJI = "dji"
    CUSTOM = "custom"
    SIMULATOR = "simulator"

class DroneController:
    def __init__(
        self,
        drone_type: DroneType = DroneType.SIMULATOR,
        connection_string: str = "udp:127.0.0.1:14550",
        baudrate: int = 57600
    ):
        """
        Initialize drone controller
        
        Args:
            drone_type: Type of drone
            connection_string: Connection string (serial port, UDP, TCP)
            baudrate: Serial baudrate
        """
        self.drone_type = drone_type
        self.connection_string = connection_string
        self.baudrate = baudrate
        
        # Telemetry data
        self.telemetry = {
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0,
            'heading': 0.0,
            'ground_speed': 0.0,
            'vertical_speed': 0.0,
            'battery_voltage': 0.0,
            'battery_percent': 100.0,
            'armed': False,
            'mode': 'UNKNOWN',
            'satellite_count': 0,
            'gps_fix_type': 0
        }
        
        # Connection
        self.connection = None
        self.connected = False
        self.running = False
        
        # Threading
        self.telemetry_thread = None
        
    def connect(self) -> bool:
        """
        Connect to drone
        
        Returns:
            True if connected successfully
        """
        try:
            if self.drone_type == DroneType.MAVLINK:
                return self._connect_mavlink()
            elif self.drone_type == DroneType.SIMULATOR:
                return self._connect_simulator()
            elif self.drone_type == DroneType.CUSTOM:
                return self._connect_custom()
            else:
                logger.error(f"Drone type {self.drone_type} not implemented")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def _connect_mavlink(self) -> bool:
        """Connect to MAVLink drone"""
        if not MAVLINK_AVAILABLE:
            logger.error("MAVLink not available. Install pymavlink.")
            return False
        
        logger.info(f"Connecting to MAVLink drone at {self.connection_string}...")
        
        try:
            self.connection = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baudrate
            )
            
            # Wait for heartbeat
            logger.info("Waiting for heartbeat...")
            self.connection.wait_heartbeat(timeout=10)
            logger.info(f"Heartbeat received from system {self.connection.target_system}")
            
            self.connected = True
            self.running = True
            
            # Start telemetry thread
            self.telemetry_thread = threading.Thread(
                target=self._mavlink_telemetry_loop,
                daemon=True
            )
            self.telemetry_thread.start()
            
            logger.info("MAVLink drone connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"MAVLink connection failed: {e}")
            return False
    
    def _connect_simulator(self) -> bool:
        """Connect to simulator (generates fake data)"""
        logger.info("Connecting to simulator...")
        self.connected = True
        self.running = True
        
        # Start simulator thread
        self.telemetry_thread = threading.Thread(
            target=self._simulator_telemetry_loop,
            daemon=True
        )
        self.telemetry_thread.start()
        
        logger.info("Simulator connected successfully")
        return True
    
    def _connect_custom(self) -> bool:
        """Connect to custom drone (implement your protocol here)"""
        logger.warning("Custom drone connection not implemented")
        return False
    
    def _mavlink_telemetry_loop(self):
        """MAVLink telemetry update loop"""
        while self.running:
            try:
                # Request data streams
                self.connection.mav.request_data_stream_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    4,  # Hz
                    1   # Start
                )
                
                # Receive messages
                msg = self.connection.recv_match(blocking=True, timeout=1.0)
                
                if msg:
                    msg_type = msg.get_type()
                    
                    # GPS position
                    if msg_type == 'GLOBAL_POSITION_INT':
                        self.telemetry['latitude'] = msg.lat / 1e7
                        self.telemetry['longitude'] = msg.lon / 1e7
                        self.telemetry['altitude'] = msg.relative_alt / 1000.0
                        self.telemetry['heading'] = msg.hdg / 100.0
                        
                    # GPS raw
                    elif msg_type == 'GPS_RAW_INT':
                        self.telemetry['satellite_count'] = msg.satellites_visible
                        self.telemetry['gps_fix_type'] = msg.fix_type
                    
                    # Attitude
                    elif msg_type == 'ATTITUDE':
                        pass  # Can add roll, pitch, yaw if needed
                    
                    # Battery
                    elif msg_type == 'SYS_STATUS':
                        self.telemetry['battery_voltage'] = msg.voltage_battery / 1000.0
                        self.telemetry['battery_percent'] = msg.battery_remaining
                    
                    # Heartbeat
                    elif msg_type == 'HEARTBEAT':
                        self.telemetry['armed'] = bool(
                            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                        )
                        # Convert mode to string
                        mode_mapping = {
                            0: 'MANUAL',
                            4: 'GUIDED',
                            10: 'AUTO',
                            11: 'RTL',
                            9: 'LAND'
                        }
                        self.telemetry['mode'] = mode_mapping.get(msg.custom_mode, 'UNKNOWN')
                        
            except Exception as e:
                logger.error(f"MAVLink telemetry error: {e}")
                time.sleep(0.1)
    
    def _simulator_telemetry_loop(self):
        """Simulator telemetry (generates realistic fake data)"""
        # Starting position (Indore, India)
        base_lat = 22.7196
        base_lon = 75.8577
        base_alt = 50.0
        
        t = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Simulate circular flight pattern
                radius = 0.001  # ~111 meters
                elapsed = time.time() - start_time
                
                self.telemetry['latitude'] = base_lat + radius * math.sin(elapsed * 0.1)
                self.telemetry['longitude'] = base_lon + radius * math.cos(elapsed * 0.1)
                self.telemetry['altitude'] = base_alt + 5 * math.sin(elapsed * 0.05)
                self.telemetry['heading'] = (elapsed * 10) % 360
                self.telemetry['ground_speed'] = 5.0
                self.telemetry['vertical_speed'] = 0.5 * math.cos(elapsed * 0.05)
                self.telemetry['battery_voltage'] = max(10.0, 12.6 - (elapsed * 0.0001))
                self.telemetry['battery_percent'] = max(0, 100 - (elapsed * 0.01))
                self.telemetry['armed'] = True
                self.telemetry['mode'] = 'AUTO'
                self.telemetry['satellite_count'] = 12
                self.telemetry['gps_fix_type'] = 3
                
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Simulator telemetry error: {e}")
                time.sleep(0.1)
    
    def get_telemetry(self) -> Dict:
        """
        Get current telemetry data
        
        Returns:
            Telemetry dictionary
        """
        return self.telemetry.copy()
    
    def get_position(self) -> Tuple[float, float, float, float]:
        """
        Get current position
        
        Returns:
            (latitude, longitude, altitude, heading)
        """
        return (
            self.telemetry['latitude'],
            self.telemetry['longitude'],
            self.telemetry['altitude'],
            self.telemetry['heading']
        )
    
    def disconnect(self):
        """Disconnect from drone"""
        self.running = False
        
        if self.telemetry_thread:
            self.telemetry_thread.join(timeout=2.0)
        
        if self.connection and self.drone_type == DroneType.MAVLINK:
            try:
                self.connection.close()
            except Exception as e:
                logger.error(f"Error closing MAVLink connection: {e}")
        
        self.connected = False
        logger.info("Disconnected from drone")
    
    def send_command(self, command: str, params: Dict = None):
        """
        Send command to drone
        
        Args:
            command: Command name
            params: Command parameters
        """
        if not self.connected:
            logger.warning("Not connected to drone")
            return False
        
        # Implement specific commands based on drone type
        if self.drone_type == DroneType.MAVLINK:
            return self._send_mavlink_command(command, params)
        elif self.drone_type == DroneType.SIMULATOR:
            logger.info(f"Simulator: Command '{command}' with params {params}")
            return True
        else:
            logger.warning(f"Command not supported for drone type: {self.drone_type}")
            return False
    
    def _send_mavlink_command(self, command: str, params: Dict = None):
        """Send MAVLink command"""
        if not MAVLINK_AVAILABLE or self.connection is None:
            logger.error("MAVLink not available or not connected")
            return False
        
        if params is None:
            params = {}
        
        try:
            # Implement common MAVLink commands
            if command == "arm":
                # Arm the drone
                self.connection.arducopter_arm()
                logger.info("MAVLink command: ARM")
                return True
            elif command == "disarm":
                # Disarm the drone
                self.connection.arducopter_disarm()
                logger.info("MAVLink command: DISARM")
                return True
            elif command == "takeoff":
                # Takeoff command
                altitude = params.get('altitude', 10.0)
                self.connection.mav.command_long_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0, 0, 0, 0, 0, 0, 0, altitude
                )
                logger.info(f"MAVLink command: TAKEOFF to {altitude}m")
                return True
            elif command == "land":
                # Land command
                self.connection.mav.command_long_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_LAND,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                logger.info("MAVLink command: LAND")
                return True
            elif command == "rtl":
                # Return to launch
                self.connection.mav.command_long_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                    0, 0, 0, 0, 0, 0, 0, 0
                )
                logger.info("MAVLink command: RTL")
                return True
            else:
                logger.warning(f"Unknown MAVLink command: {command}")
                return False
        except Exception as e:
            logger.error(f"Error sending MAVLink command: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to drone"""
        return self.connected
    
    def get_status(self) -> str:
        """Get drone status string"""
        if not self.connected:
            return "DISCONNECTED"
        
        status = f"Mode: {self.telemetry['mode']} | "
        status += f"Armed: {self.telemetry['armed']} | "
        status += f"Battery: {self.telemetry['battery_percent']:.1f}% | "
        status += f"GPS: {self.telemetry['satellite_count']} sats"
        
        return status

def main():
    """Test drone controller"""
    # Create controller (simulator mode for testing)
    controller = DroneController(
        drone_type=DroneType.SIMULATOR,
        connection_string="udp:127.0.0.1:14550"
    )
    
    # Connect
    if controller.connect():
        logger.info("Connected to drone successfully")
        logger.info("Monitoring telemetry...")
        
        try:
            for i in range(20):
                telemetry = controller.get_telemetry()
                lat, lon, alt, hdg = controller.get_position()
                
                logger.info(f"Telemetry Update {i+1}:")
                logger.info(f"  Position: [{lat:.6f}, {lon:.6f}]")
                logger.info(f"  Altitude: {alt:.1f}m")
                logger.info(f"  Heading: {hdg:.1f}°")
                logger.info(f"  Status: {controller.get_status()}")
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Disconnect
            controller.disconnect()
            logger.info("Test completed")
    else:
        logger.error("Failed to connect to drone")

if __name__ == "__main__":
    main()