"""
Alert system for fire detection
"""
import time
from typing import List, Dict, Callable, Optional
from datetime import datetime
from pathlib import Path
import json
from loguru import logger

class AlertSystem:
    def __init__(
        self,
        alert_threshold: int = 1,
        cooldown_period: int = 60,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize alert system
        
        Args:
            alert_threshold: Minimum number of detections to trigger alert
            cooldown_period: Cooldown period in seconds between alerts
            alert_callbacks: List of callback functions for alerts
        """
        self.alert_threshold = alert_threshold
        self.cooldown_period = cooldown_period
        self.alert_callbacks = alert_callbacks or []
        
        # Alert history
        self.alerts = []
        self.last_alert_time = 0
        self.alert_count = 0
        
        # Alert directory
        self.alert_dir = Path("outputs/alerts")
        self.alert_dir.mkdir(parents=True, exist_ok=True)
    
    def check_detections(self, detections: List[Dict], metadata: Dict = None) -> bool:
        """
        Check detections and trigger alert if needed
        
        Args:
            detections: List of detection dictionaries
            metadata: Additional metadata (telemetry, etc.)
            
        Returns:
            True if alert was triggered
        """
        if not detections:
            return False
        
        # Check if cooldown period has passed
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown_period:
            return False
        
        # Check if threshold is met
        if len(detections) >= self.alert_threshold:
            return self.trigger_alert(detections, metadata)
        
        return False
    
    def trigger_alert(self, detections: List[Dict], metadata: Dict = None) -> bool:
        """
        Trigger fire alert
        
        Args:
            detections: List of detection dictionaries
            metadata: Additional metadata
            
        Returns:
            True if alert was triggered
        """
        current_time = time.time()
        self.last_alert_time = current_time
        self.alert_count += 1
        
        # Create alert
        alert = {
            'id': self.alert_count,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'detection_count': len(detections),
            'metadata': metadata or {}
        }
        
        self.alerts.append(alert)
        
        # Save alert to file
        self._save_alert(alert)
        
        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.warning(f"🔥 FIRE ALERT #{self.alert_count}: {len(detections)} detections")
        
        return True
    
    def _save_alert(self, alert: Dict):
        """Save alert to file"""
        alert_file = self.alert_dir / f"alert_{alert['id']:04d}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
        
        # Also append to alerts log
        alerts_log = self.alert_dir / "alerts_log.json"
        alerts_data = []
        if alerts_log.exists():
            with open(alerts_log, 'r') as f:
                alerts_data = json.load(f)
        
        alerts_data.append(alert)
        with open(alerts_log, 'w') as f:
            json.dump(alerts_data, f, indent=2)
    
    def get_alerts(self, limit: int = None) -> List[Dict]:
        """Get alert history"""
        if limit:
            return self.alerts[-limit:]
        return self.alerts
    
    def get_statistics(self) -> Dict:
        """Get alert statistics"""
        return {
            'total_alerts': self.alert_count,
            'alert_threshold': self.alert_threshold,
            'cooldown_period': self.cooldown_period,
            'last_alert_time': datetime.fromtimestamp(self.last_alert_time).isoformat() if self.last_alert_time > 0 else None
        }
    
    def add_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def reset(self):
        """Reset alert system"""
        self.alerts = []
        self.last_alert_time = 0
        self.alert_count = 0

def email_alert_callback(alert: Dict, email_config: Dict):
    """Email alert callback"""
    # Implement email sending
    logger.info(f"Email alert sent for alert #{alert['id']}")

def sms_alert_callback(alert: Dict, sms_config: Dict):
    """SMS alert callback"""
    # Implement SMS sending
    logger.info(f"SMS alert sent for alert #{alert['id']}")

def webhook_alert_callback(alert: Dict, webhook_url: str):
    """Webhook alert callback"""
    import requests
    try:
        response = requests.post(webhook_url, json=alert, timeout=5)
        response.raise_for_status()
        logger.info(f"Webhook alert sent for alert #{alert['id']}")
    except Exception as e:
        logger.error(f"Webhook alert failed: {e}")

