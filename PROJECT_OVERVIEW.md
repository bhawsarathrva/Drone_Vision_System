# 🚁 Drone Fire Detection System - Project Analysis & Documentation

## 📖 Project Overview
The **Drone Fire Detection System** is a comprehensive solution designed for aerial surveillance and early fire detection. It leverages **YOLOv8** (You Only Look Once) deep learning models combined with **traditional computer vision techniques** (pixel-based color segmentation) to achieve high accuracy and robustness. The system is built to run on drones, providing real-time detection, geolocation of fire targets, and alert generation.

## 🏗️ System Architecture

The project follows a modular architecture, separating concerns into distinct components:

1.  **Core Detection Engine**: Handles image processing and object detection.
2.  **Real-time Processing**: Manages video streams, threading, and performance.
3.  **Drone Interface**: Communicates with flight controllers (MAVLink/Simulator).
4.  **Geolocation**: Maps detections from image pixels to GPS coordinates.
5.  **Alerting**: Triggers notifications based on detection confidence and persistence.
6.  **Training & MLOps**: Pipelines for dataset management and model training.

---

## 📂 Directory Structure & File Analysis

### 1. Root Directory (Entry Points)

*   **`main.py`**: 
    *   **Role**: The central command center.
    *   **Functionality**: Initializes all subsystems (`RealtimeDetector`, `DroneController`, `AlertSystem`, `MapAPI`). It supports three modes:
        *   `realtime`: Runs the detection loop with GUI.
        *   `api`: Starts a FastAPI server for remote interaction.
        *   `both`: Runs both simultaneously.
    *   **Key Features**: Handles configuration loading, logging setup, and graceful shutdown.

*   **`predict_pipeline.py`**:
    *   **Role**: Standalone inference script.
    *   **Functionality**: Runs predictions on static images, video files, or camera streams.
    *   **Key Features**: Visualizes results with bounding boxes and labels, saves outputs to `outputs/predictions`.

*   **`train_pipeline.py`**:
    *   **Role**: End-to-end training script.
    *   **Functionality**: Automates the training workflow: verifying datasets, loading models (local or Hugging Face), training, validating, and exporting.
    *   **Key Features**: tailored for the specific project structure.

*   **`train_model.py`**:
    *   **Role**: Wrapper for modular training.
    *   **Functionality**: Utilizes `src.training.train.FireDetectionTrainer` to execute training based on `configs/training_config.yaml`.

### 2. `src/` Directory (Core Modules)

#### `src/detection/` - The Vision Core
*   **`fire_detector.py`**:
    *   **Class**: `FireDetector`
    *   **Functionality**: Implements the **Hybrid Detection Strategy**:
        1.  **YOLOv8**: Deep learning detection for 'fire' and 'smoke' classes.
        2.  **Pixel-based**: HSV color segmentation to detect fire colors (red/orange) and smoke (gray/white).
    *   **Logic**: Merges detections from both methods, removing duplicates based on IoU (Intersection over Union).
    *   **Preprocessing**: Includes denoising and CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility in drone footage.

#### `src/realtime/` - Stream Management
*   **`realtime_detector.py`**:
    *   **Class**: `RealtimeDetector`
    *   **Functionality**: Wraps `FireDetector` for continuous stream processing.
    *   **Key Features**: 
        *   Runs detection in a separate thread to maintain UI responsiveness.
        *   Integrates `ConfidenceFilter` to reduce false positives.
        *   Uses `SizeClassifier` to categorize fire size (Small, Medium, Large).
        *   Syncs with `DroneController` to attach telemetry data to detections.

#### `src/drone/` - Flight Control
*   **`drone_controller.py`**:
    *   **Class**: `DroneController`
    *   **Functionality**: Interfaces with the drone hardware or simulator.
    *   **Protocols**: Supports **MAVLink** (via `pymavlink`) for ArduPilot/PX4 and a built-in **Simulator**.
    *   **Data**: Provides real-time telemetry: GPS (Lat/Lon/Alt), Heading, Battery status, and Speed.
    *   **Commands**: Can send basic commands like ARM, TAKEOFF, LAND, RTL.

#### `src/geolocation/` - Mapping
*   **`coordinate_mapper.py`**:
    *   **Class**: `CoordinateMapper`
    *   **Functionality**: Solves the "Pixel-to-GPS" problem.
    *   **Math**: Uses trigonometry and drone telemetry (Altitude, Heading, Camera FOV) to calculate the real-world coordinates of a detected fire.
    *   **Output**: Returns Latitude/Longitude for the center and corners of the detection bounding box.

#### `src/utils/` - Utilities
*   **`alert_system.py`**:
    *   **Class**: `AlertSystem`
    *   **Functionality**: Manages incident reporting.
    *   **Logic**: Triggers alerts only when specific thresholds are met (e.g., X detections in Y seconds) to prevent alarm fatigue.
    *   **Storage**: Saves alerts as JSON files in `outputs/alerts`.

#### `src/training/` - Modular Training
*   **`train.py`**:
    *   **Class**: `FireDetectionTrainer`
    *   **Functionality**: Encapsulates the training logic using the Ultralytics API.
    *   **Config**: Driven by `configs/training_config.yaml` for flexible experiment management.

---

## 🚀 Workflows

### 1. Training a Model
To train a new model on your dataset:
```bash
# Using the pipeline script (Recommended)
python train_pipeline.py --epochs 100 --model yolov8s.pt

# OR using the modular script
python train_model.py
```

### 2. Running Real-time Detection
To start the system with your webcam and simulated drone data:
```bash
python main.py --mode realtime --camera 0
```

### 3. Running Predictions on Media
To detect fire in a video file:
```bash
python predict_pipeline.py --source path/to/video.mp4 --save
```

### 4. Adding Negative Samples
To improve model robustness by adding non-fire images:
```bash
python scripts/add_negatives.py --source path/to/images --category sun
```

## ⚙️ Configuration
The system is highly configurable via `configs/config.yaml`:
*   **Model**: Path to weights, confidence thresholds.
*   **Camera**: Source ID/URL, FPS.
*   **Drone**: Connection string (e.g., `udp:127.0.0.1:14550`), type.
*   **Geolocation**: Camera FOV settings.
*   **Alerts**: Thresholds and cooldowns.
