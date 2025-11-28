import argparse
import cv2
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detection.fire_detector import FireDetector
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description='Run Fire Detection Inference')
    parser.add_argument('--source', type=str, default='0', help='Source: "0" for camera, path to image or video')
    parser.add_argument('--model', type=str, default='Model/best.pt', help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save output')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = FireDetector(
            model_path=args.model,
            confidence_threshold=args.conf
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        return

    source = args.source
    
    # Check if source is image or video/camera
    is_image = False
    if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        is_image = True
    
    if is_image:
        # Process image
        if not os.path.exists(source):
            logger.error(f"Image not found: {source}")
            return
            
        img = cv2.imread(source)
        if img is None:
            logger.error(f"Failed to read image: {source}")
            return
            
        detections = detector.detect(img)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        cv2.imshow('Fire Detection', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if args.save:
            output_path = f"outputs/detected_{os.path.basename(source)}"
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite(output_path, img)
            logger.info(f"Saved output to {output_path}")
            
    else:
        # Process video/camera
        if source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open source: {source}")
            return
            
        logger.info("Starting detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            detections = detector.detect(frame)
            
            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Show stats
            stats = detector.get_statistics()
            cv2.putText(frame, f"FPS: {stats['avg_fps']:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Fire Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
