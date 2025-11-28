import os
import sys
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from ultralytics import YOLO
import torch
from typing import List, Dict, Optional, Union

class FireDetectionPredictionPipeline:
    """Complete prediction pipeline for fire detection"""
    
    def __init__(
        self,
        model_path: str = "Model/best.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        img_size: int = 640
    ):
        """
        Initialize prediction pipeline
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use (auto/cpu/cuda)
            img_size: Image size for inference
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.class_names = []
        
        logger.info("="*70)
        logger.info("Fire Detection Prediction Pipeline Initialized")
        logger.info("="*70)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Confidence Threshold: {self.confidence_threshold}")
        logger.info(f"IoU Threshold: {self.iou_threshold}")
        logger.info("="*70)
        
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                logger.info(f"Classes: {self.class_names}")
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_image(self, image_path: str, save_output: bool = True, output_dir: str = "outputs/predictions"):
        logger.info(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
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
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class': class_id,
                        'class_name': class_name
                    })
                    
                    color = (0, 0, 255) if class_name == 'fire' else (255, 0, 0)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image, label, (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        logger.info(f"Found {len(detections)} detections")
        
        if save_output and detections:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, image)
            logger.info(f"✓ Saved output to: {output_path}")
        
        return detections, image
    
    def predict_video(self, video_path: str, save_output: bool = True, output_dir: str = "outputs/predictions"):
        """
        Predict on a video file
        
        Args:
            video_path: Path to input video
            save_output: Whether to save annotated output
            output_dir: Directory to save outputs
        """
        logger.info(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))    
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        writer = None
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{os.path.basename(video_path)}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                        
                        detection_count += 1
                        
                        color = (0, 0, 255) if class_name == 'fire' else (255, 0, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if writer:
                writer.write(frame)
            
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames, {detection_count} detections")
        
        cap.release()
        if writer:
            writer.release()
            logger.info(f"✓ Saved output to: {output_path}")
        
        logger.info(f"✓ Video processing complete: {frame_count} frames, {detection_count} detections")
    
    def predict_camera(self, camera_index: int = 0, display: bool = True):
        """
        Predict on camera feed
        
        Args:
            camera_index: Camera index (0 for default)
            display: Whether to display output
        """
        logger.info(f"Starting camera feed (index: {camera_index})")
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera: {camera_index}")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False
            )
            
            # Draw detections
            frame_detections = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id] if hasattr(result, 'names') else str(class_id)
                        
                        detection_count += 1
                        frame_detections += 1
                        
                        color = (0, 0, 255) if class_name == 'fire' else (255, 0, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display stats
            stats_text = f"Frame: {frame_count} | Detections: {frame_detections} | Total: {detection_count}"
            cv2.putText(frame, stats_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if display:
                cv2.imshow('Fire Detection - Press q to quit, s to save', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    os.makedirs("outputs/screenshots", exist_ok=True)
                    screenshot_path = f"outputs/screenshots/frame_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"✓ Camera feed stopped: {frame_count} frames, {detection_count} detections")
    
    def predict_batch(self, image_dir: str, save_output: bool = True, output_dir: str = "outputs/predictions"):
        """
        Predict on a batch of images
        
        Args:
            image_dir: Directory containing images
            save_output: Whether to save outputs
            output_dir: Directory to save outputs
        """
        logger.info(f"Processing batch from: {image_dir}")
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images")
        
        total_detections = 0
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
            detections, _ = self.predict_image(str(image_path), save_output, output_dir)
            total_detections += len(detections)
        
        logger.info(f"✓ Batch processing complete: {len(image_files)} images, {total_detections} detections")


def main():
    """Main function to run prediction pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fire Detection Prediction Pipeline')
    parser.add_argument('--model', type=str, default='Model/best.pt',
                       help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image path, video path, camera index (0), or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')
    parser.add_argument('--save', action='store_true',
                       help='Save output')
    parser.add_argument('--output-dir', type=str, default='outputs/predictions',
                       help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = FireDetectionPredictionPipeline(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        img_size=args.img_size
    )
    
    source = args.source
    
    if source.isdigit():
        pipeline.predict_camera(camera_index=int(source), display=True)
    elif os.path.isfile(source):
        if source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            pipeline.predict_video(source, save_output=args.save, output_dir=args.output_dir)
        else:
            detections, image = pipeline.predict_image(source, save_output=args.save, output_dir=args.output_dir)
            cv2.imshow('Detection Result - Press any key to close', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif os.path.isdir(source):
        pipeline.predict_batch(source, save_output=args.save, output_dir=args.output_dir)
    else:
        logger.error(f"Invalid source: {source}")


if __name__ == "__main__":
    main()
