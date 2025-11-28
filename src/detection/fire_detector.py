import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import torch
from loguru import logger
import time

class FireDetector:
    """
    Enhanced Fire Detector with hybrid detection:
    1. YOLO-based object detection
    2. Pixel-based color segmentation for fire/smoke
    """
    
    def __init__(
        self,
        model_path: str = "Model/best.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        img_size: int = 640,
        use_half: bool = True,
        use_tta: bool = False,
        agnostic_nms: bool = False,
        max_det: int = 300,
        use_pixel_detection: bool = True
    ):
        """
        Initialize enhanced fire detector with YOLO and pixel-based detection
        
        Args:
            model_path: Path to YOLOv8 model weights or Hugging Face model ID
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            device: Device to run inference ('cpu', 'cuda', 'auto')
            img_size: Input image size for model (640, 1280, etc.)
            use_half: Use FP16 half-precision inference (faster on GPU)
            use_tta: Use Test Time Augmentation for better accuracy
            agnostic_nms: Class-agnostic NMS
            max_det: Maximum detections per image
            use_pixel_detection: Enable pixel-based fire detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.use_half = use_half
        self.use_tta = use_tta
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.use_pixel_detection = use_pixel_detection
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        self.model = None
        self.model_type = None
        self.class_names = []
        
        # Load YOLO model
        self.load_model()
        self._warmup_model()
        
        # Statistics
        self.detection_count = 0
        self.total_frames = 0
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0
        
        # Preprocessing settings
        self.preprocessing_enabled = True
        self.denoise_enabled = False
        
        # Pixel-based detection parameters (HSV color ranges)
        # Fire color range (red-orange-yellow)
        self.fire_lower1 = np.array([0, 50, 50])      # Lower red
        self.fire_upper1 = np.array([10, 255, 255])
        self.fire_lower2 = np.array([170, 50, 50])    # Upper red
        self.fire_upper2 = np.array([180, 255, 255])
        
        # Smoke color range (gray-white)
        self.smoke_lower = np.array([0, 0, 100])
        self.smoke_upper = np.array([180, 50, 255])
        
        # Minimum area for pixel-based detection (to filter noise)
        self.min_fire_area = 500
        self.min_smoke_area = 1000
        
        logger.info(f"FireDetector initialized - Device: {self.device}, Half: {self.use_half}, TTA: {self.use_tta}")
        logger.info(f"Pixel-based detection: {'Enabled' if self.use_pixel_detection else 'Disabled'}")
        
    def load_model(self):
        """Load YOLO model"""
        try:        
            if self._is_huggingface_model(self.model_path):
                logger.info(f"Loading model from Hugging Face: {self.model_path}")
                self.model = self._load_from_huggingface(self.model_path)
                self.model_type = 'huggingface'
            else:
                if not Path(self.model_path).exists():
                    logger.warning(f"Model not found at {self.model_path}")
                    logger.info("Attempting to use Hugging Face YOLOv8 model...")
                    try:
                        self.model = self._load_from_huggingface("Ultralytics/YOLOv8")
                        self.model_type = 'huggingface'
                    except:
                        logger.warning("Falling back to default yolov8n.pt")
                        self.model_path = "yolov8n.pt"
                        self.model = YOLO(self.model_path)
                        self.model_type = 'local'
                else:
                    self.model = YOLO(self.model_path)
                    self.model_type = 'local'
            
            self.model.to(self.device)
            
            if self.use_half and self.device == "cuda":
                try:
                    self.model.model.half()
                    logger.info("Half-precision (FP16) enabled")
                except Exception as e:
                    logger.warning(f"Could not enable half precision: {e}")
            
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            
            logger.info(f"Model loaded successfully from {self.model_path} on {self.device}")
            logger.info(f"Model classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Continuing with pixel-based detection only")
            self.model = None
    
    def _is_huggingface_model(self, model_path: str) -> bool:
        """Check if model path is a Hugging Face model ID"""
        return "/" in model_path and not any(x in model_path for x in ['.pt', '.pth', '\\', 'Model'])
    
    def _load_from_huggingface(self, model_id: str):
        """Load model from Hugging Face Hub"""
        try:
            from huggingface_hub import hf_hub_download
            
            if "Ultralytics/YOLOv8" in model_id or "ultralytics" in model_id.lower():
                model_variant = "yolov8n.pt"  
                if "yolov8s" in model_id.lower():
                    model_variant = "yolov8s.pt"
                elif "yolov8m" in model_id.lower():
                    model_variant = "yolov8m.pt"
                elif "yolov8l" in model_id.lower():
                    model_variant = "yolov8l.pt"
                elif "yolov8x" in model_id.lower():
                    model_variant = "yolov8x.pt"
                
                logger.info(f"Loading YOLOv8 variant: {model_variant}")
                return YOLO(model_variant)
            else:
                model_file = hf_hub_download(repo_id=model_id, filename="best.pt")
                return YOLO(model_file)
                
        except Exception as e:
            logger.error(f"Failed to load from Hugging Face: {e}")
            raise
    
    def _warmup_model(self, warmup_iterations: int = 3):
        """Warmup the model"""
        if self.model is None:
            return
            
        try:
            logger.info("Warming up model...")
            dummy_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
            for _ in range(warmup_iterations):
                _ = self.model(
                    dummy_img,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.img_size,
                    device=self.device,
                    verbose=False
                )
            
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better detection
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        if not self.preprocessing_enabled:
            return image
        
        processed = image.copy()
        
        # Denoising for noisy drone footage
        if self.denoise_enabled:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # CLAHE for better contrast
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        return processed
    
    def detect_fire_pixels(self, image: np.ndarray) -> List[Dict]:
        """
        Detect fire using pixel-based color segmentation
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create masks for fire (red-orange-yellow colors)
        fire_mask1 = cv2.inRange(hsv, self.fire_lower1, self.fire_upper1)
        fire_mask2 = cv2.inRange(hsv, self.fire_lower2, self.fire_upper2)
        fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
        
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for fire
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_fire_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and color intensity
                roi = hsv[y:y+h, x:x+w]
                mean_val = np.mean(roi[:, :, 2])  # V channel (brightness)
                confidence = min(0.95, (area / 10000) * 0.5 + (mean_val / 255) * 0.5)
                
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': confidence,
                    'class': 0,
                    'class_name': 'fire',
                    'area': area,
                    'detection_method': 'pixel'
                })
        
        # Detect smoke (gray-white colors)
        smoke_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for smoke
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_smoke_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence
                confidence = min(0.85, (area / 15000) * 0.6)
                
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': confidence,
                    'class': 1,
                    'class_name': 'smoke',
                    'area': area,
                    'detection_method': 'pixel'
                })
        
        return detections
    
    def detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Detect fire using YOLO model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None:
            return []
        
        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                half=self.use_half,
                augment=self.use_tta,
                agnostic_nms=self.agnostic_nms,
                max_det=self.max_det,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                        
                        area = (x2 - x1) * (y2 - y1)
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class': class_id,
                            'class_name': class_name,
                            'area': int(area),
                            'detection_method': 'yolo'
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def merge_detections(self, yolo_dets: List[Dict], pixel_dets: List[Dict]) -> List[Dict]:
        """
        Merge YOLO and pixel-based detections, removing duplicates
        
        Args:
            yolo_dets: YOLO detections
            pixel_dets: Pixel-based detections
            
        Returns:
            Merged list of detections
        """
        if not pixel_dets:
            return yolo_dets
        if not yolo_dets:
            return pixel_dets
        
        merged = yolo_dets.copy()
        
        for pixel_det in pixel_dets:
            px1, py1, px2, py2 = pixel_det['bbox']
            pixel_area = (px2 - px1) * (py2 - py1)
            
            # Check if this pixel detection overlaps significantly with any YOLO detection
            is_duplicate = False
            for yolo_det in yolo_dets:
                yx1, yy1, yx2, yy2 = yolo_det['bbox']
                
                # Calculate IoU
                x1 = max(px1, yx1)
                y1 = max(py1, yy1)
                x2 = min(px2, yx2)
                y2 = min(py2, yy2)
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    yolo_area = (yx2 - yx1) * (yy2 - yy1)
                    union = pixel_area + yolo_area - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    # If IoU > 0.3, consider it a duplicate
                    if iou > 0.3:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                merged.append(pixel_det)
        
        return merged
    
    def detect(self, image: np.ndarray, preprocess: bool = True) -> List[Dict]:
        """
        Detect fire in image using hybrid approach (YOLO + pixel-based)
        
        Args:
            image: Input image (BGR format)
            preprocess: Apply preprocessing
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: Confidence score
            - class: Class ID
            - class_name: Class name
            - area: Bounding box area
            - detection_method: 'yolo' or 'pixel'
        """
        self.total_frames += 1
        
        try:
            # Preprocess image
            if preprocess:
                processed_image = self.preprocess_image(image)
            else:
                processed_image = image
            
            start_time = time.time()
            
            # YOLO detection
            yolo_detections = self.detect_yolo(processed_image)
            
            # Pixel-based detection
            pixel_detections = []
            if self.use_pixel_detection:
                pixel_detections = self.detect_fire_pixels(processed_image)
            
            # Merge detections
            detections = self.merge_detections(yolo_detections, pixel_detections)
            
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.avg_inference_time = self.total_inference_time / self.total_frames
            
            # Add inference time to each detection
            for det in detections:
                det['inference_time'] = inference_time
                self.detection_count += 1
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def detect_batch(self, images: List[np.ndarray], preprocess: bool = True) -> List[List[Dict]]:
        """
        Detect fire in batch of images
        
        Args:
            images: List of input images
            preprocess: Apply preprocessing
            
        Returns:
            List of detection lists for each image
        """
        all_detections = []
        
        for image in images:
            detections = self.detect(image, preprocess)
            all_detections.append(detections)
        
        return all_detections
    
    def get_statistics(self) -> Dict:
        """Get comprehensive detection statistics"""
        detection_rate = (self.detection_count / self.total_frames * 100) if self.total_frames > 0 else 0
        fps = 1.0 / self.avg_inference_time if self.avg_inference_time > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'total_detections': self.detection_count,
            'detection_rate': detection_rate,
            'avg_inference_time': self.avg_inference_time,
            'avg_fps': fps,
            'model_path': self.model_path,
            'model_type': self.model_type,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'img_size': self.img_size,
            'use_half': self.use_half,
            'use_tta': self.use_tta,
            'pixel_detection_enabled': self.use_pixel_detection
        }
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_count = 0
        self.total_frames = 0
        self.total_inference_time = 0.0
        self.avg_inference_time = 0.0
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to {self.confidence_threshold}")
    
    def update_iou_threshold(self, threshold: float):
        """Update IoU threshold for NMS"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"IoU threshold updated to {self.iou_threshold}")
    
    def enable_preprocessing(self, enable: bool = True):
        """Enable/disable image preprocessing"""
        self.preprocessing_enabled = enable
        logger.info(f"Preprocessing {'enabled' if enable else 'disabled'}")
    
    def enable_denoising(self, enable: bool = True):
        """Enable/disable denoising for noisy drone footage"""
        self.denoise_enabled = enable
        logger.info(f"Denoising {'enabled' if enable else 'disabled'}")
    
    def enable_pixel_detection(self, enable: bool = True):
        """Enable/disable pixel-based detection"""
        self.use_pixel_detection = enable
        logger.info(f"Pixel-based detection {'enabled' if enable else 'disabled'}")


def main():
    """Test fire detector with enhanced features"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Fire Detector Test')
    parser.add_argument('--model', type=str, default='Model/best.pt',
                       help='Path to model weights or Hugging Face model ID')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (camera index, file path, or RTSP URL)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 half-precision')
    parser.add_argument('--tta', action='store_true',
                       help='Use Test Time Augmentation')
    parser.add_argument('--denoise', action='store_true',
                       help='Enable denoising')
    parser.add_argument('--no-pixel', action='store_true',
                       help='Disable pixel-based detection')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FireDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
        img_size=args.img_size,
        use_half=args.half,
        use_tta=args.tta,
        use_pixel_detection=not args.no_pixel
    )
    
    if args.denoise:
        detector.enable_denoising(True)
    
    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {args.source}")
        return
    
    logger.info("Starting detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect fire
        detections = detector.detect(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            method = det.get('detection_method', 'unknown')
            
            # Different colors for different detection methods
            if method == 'yolo':
                color = (0, 255, 0)  # Green for YOLO
            else:
                color = (255, 0, 0)  # Blue for pixel-based
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f} [{method}]"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show statistics
        stats = detector.get_statistics()
        stats_text = f"FPS: {stats['avg_fps']:.1f} | Detections: {stats['total_detections']} | Frames: {stats['total_frames']}"
        cv2.putText(frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Enhanced Fire Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    stats = detector.get_statistics()
    logger.info("="*60)
    logger.info("Final Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)


if __name__ == "__main__":
    main()