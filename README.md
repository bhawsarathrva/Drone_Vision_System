# 🔥 Drone Fire Detection System - Complete Setup Guide

An end-to-end fire detection system for drones using YOLOv8 from Hugging Face. This system provides real-time fire and smoke detection with training and prediction pipelines optimized for aerial surveillance.

---

## 📋 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training Pipeline](#-training-pipeline)
- [Prediction Pipeline](#-prediction-pipeline)
- [Troubleshooting](#-troubleshooting)
- [Project Structure](#-project-structure)
- [Advanced Usage](#-advanced-usage)

---

## ✨ Features

### Core Capabilities
- **Hybrid Detection**: YOLO model + pixel-based color segmentation
- **End-to-End Pipeline**: Complete training and prediction workflows
- **Hugging Face Integration**: Automatic model loading from HF Hub
- **Real-time Detection**: Camera, video, and RTSP stream support
- **GPU Acceleration**: FP16 half-precision for 2x speed boost
- **Advanced Preprocessing**: CLAHE, denoising for drone footage
- **Batch Processing**: Process multiple images efficiently

### Detection Methods
1. **YOLO Model Detection**: Deep learning-based object detection
2. **Pixel-Based Detection**: HSV color segmentation (works without trained model)
3. **Hybrid Mode**: Combines both methods for maximum accuracy

---

## 🔧 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or 3.11 (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA (optional but recommended)

### Required Software
- Python 3.10/3.11
- pip or uv package manager
- Git (for cloning repository)

---

## 📦 Installation

### Step 1: Clone or Navigate to Project

```powershell
cd "c:\Users\athrv\OneDrive\Desktop\Drone System"
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venv\Scripts\activate.bat

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

#### Option A: Using pip (Standard)

```powershell
# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install ultralytics opencv-python loguru pyyaml roboflow

# Or install all from requirements.txt
pip install -r requirements.txt
```

#### Option B: Using uv (10-100x Faster)

```powershell
# Install uv if not already installed
pip install uv

# Install PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
uv pip install -r requirements.txt
```

#### Option C: GPU Support (NVIDIA CUDA)

```powershell
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Verify Installation

```powershell
# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"

# Test Ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## 🚀 Quick Start

### 1. Test Pixel-Based Detection (No Model Required)

```powershell
# Test with webcam
python src/detection/fire_detector.py --source 0

# Test with image
python src/detection/fire_detector.py --source path/to/image.jpg
```

### 2. Train Your Model

```powershell
# Train with default settings (YOLOv8s, 100 epochs)
python train_pipeline.py

# Train with custom settings
python train_pipeline.py --model yolov8m.pt --epochs 150 --batch 8
```

### 3. Run Predictions

```powershell
# Predict on image
python predict_pipeline.py --source path/to/image.jpg --save

# Real-time camera detection
python predict_pipeline.py --source 0

# Process video
python predict_pipeline.py --source path/to/video.mp4 --save
```

---

## 🎓 Training Pipeline

### Dataset Information

Your Roboflow dataset is located at `Dataset/fire.yolov8/`:
- **Training**: 2,469 images
- **Validation**: 702 images
- **Test**: 347 images
- **Classes**: fire, smoke

### Training Commands

#### Basic Training

```powershell
# Default: YOLOv8s, 100 epochs, auto device
python train_pipeline.py
```

#### Custom Training

```powershell
# High accuracy (YOLOv8 Medium)
python train_pipeline.py --model yolov8m.pt --epochs 150 --batch 8

# Fast training (YOLOv8 Nano)
python train_pipeline.py --model yolov8n.pt --epochs 100 --batch 32

# Maximum accuracy (YOLOv8 Large)
python train_pipeline.py --model yolov8l.pt --epochs 200 --batch 4
```

### Model Variants

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| yolov8n.pt | 6MB | Fastest | Good | Drones, edge devices |
| yolov8s.pt | 22MB | Fast | Better | **Recommended (80%+ accuracy)** |
| yolov8m.pt | 52MB | Moderate | High | Maximum accuracy |
| yolov8l.pt | 87MB | Slow | Very High | Research |
| yolov8x.pt | 136MB | Slowest | Highest | Benchmarking |

### Training Parameters

```powershell
python train_pipeline.py \
  --data Dataset/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --batch 16 \
  --img-size 640 \
  --device auto \
  --project fire_detection_training \
  --name yolov8s_fire
```

### Training Output

After training completes:
- **Best model**: `Model/best.pt` (automatically saved)
- **Training results**: `fire_detection_training/yolov8s_fire/`
  - `weights/best.pt` - Best model weights
  - `weights/last.pt` - Last checkpoint
  - `results.png` - Training metrics
  - `confusion_matrix.png` - Confusion matrix
  - `val_batch0_pred.jpg` - Validation predictions

### Expected Training Time

- **YOLOv8n**: ~2-3 hours (100 epochs, GPU)
- **YOLOv8s**: ~3-4 hours (100 epochs, GPU)
- **YOLOv8m**: ~5-6 hours (100 epochs, GPU)

---

## 🔮 Prediction Pipeline

### Single Image Prediction

```powershell
# Basic prediction
python predict_pipeline.py --source path/to/image.jpg

# Save output
python predict_pipeline.py --source path/to/image.jpg --save

# Custom confidence threshold
python predict_pipeline.py --source path/to/image.jpg --conf 0.3 --save
```

### Video Processing

```powershell
# Process video file
python predict_pipeline.py --source path/to/video.mp4 --save

# Higher confidence threshold
python predict_pipeline.py --source path/to/video.mp4 --conf 0.5 --save
```

### Real-time Camera Detection

```powershell
# Default webcam
python predict_pipeline.py --source 0

# External camera
python predict_pipeline.py --source 1

# RTSP stream (drone camera)
python predict_pipeline.py --source "rtsp://192.168.1.100:8554/stream"
```

**Camera Controls:**
- Press `q` to quit
- Press `s` to save screenshot

### Batch Processing

```powershell
# Process all images in a folder
python predict_pipeline.py --source path/to/image/folder --save
```

### Prediction Parameters

```powershell
python predict_pipeline.py \
  --source 0 \
  --model Model/best.pt \
  --conf 0.25 \
  --iou 0.45 \
  --img-size 640 \
  --device auto \
  --save \
  --output-dir outputs/predictions
```

---

## 🐛 Troubleshooting

### Issue 1: PyTorch DLL Error (Windows)

**Error:**
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll"
```

**Solution 1: Install Visual C++ Redistributables (Recommended)**

1. Download installer:
   ```powershell
   Start-Process "https://aka.ms/vs/17/release/vc_redist.x64.exe"
   ```

2. Run the installer and follow prompts

3. **Restart your computer**

4. Test PyTorch:
   ```powershell
   python -c "import torch; print('Success!')"
   ```

**Solution 2: Fresh Virtual Environment**

```powershell
# Remove old venv
Remove-Item -Recurse -Force venv

# Create new venv
python -m venv venv_fixed

# Activate
.\venv_fixed\Scripts\Activate.ps1

# Install stable PyTorch version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install ultralytics opencv-python loguru pyyaml roboflow
```

**Solution 3: Use Conda**

```powershell
# Create conda environment
conda create -n fire_detection python=3.11 -y

# Activate
conda activate fire_detection

# Install PyTorch (conda handles DLLs automatically)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other requirements
pip install ultralytics opencv-python loguru pyyaml roboflow
```

### Issue 2: CUDA Out of Memory

```powershell
# Reduce batch size
python train_pipeline.py --batch 8

# Or use CPU
python train_pipeline.py --device cpu
```

### Issue 3: Low Training Accuracy

```powershell
# Train longer with larger model
python train_pipeline.py --model yolov8m.pt --epochs 150

# Adjust confidence threshold during prediction
python predict_pipeline.py --source test.jpg --conf 0.3
```

### Issue 4: Slow Inference

```powershell
# Use smaller model
python predict_pipeline.py --model Model/best.pt --img-size 416

# Enable GPU if available
python predict_pipeline.py --source 0 --device cuda
```

### Issue 5: Camera Not Found

```powershell
# Try different camera index
python predict_pipeline.py --source 1

# Or use video file for testing
python predict_pipeline.py --source test_video.mp4
```

---

## 📁 Project Structure

```
Drone System/
├── Dataset/
│   ├── data.yaml                    # Dataset configuration
│   └── fire.yolov8/                 # Roboflow dataset
│       ├── train/                   # Training images & labels
│       ├── valid/                   # Validation images & labels
│       └── test/                    # Test images & labels
│
├── Model/
│   ├── best.pt                      # Best trained model
│   └── last.pt                      # Last checkpoint
│
├── fire_detection_training/         # Training outputs
│   └── yolov8s_fire/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       ├── results.png              # Training curves
│       └── confusion_matrix.png     # Confusion matrix
│
├── outputs/
│   ├── predictions/                 # Prediction outputs
│   └── screenshots/                 # Camera screenshots
│
├── src/
│   ├── detection/
│   │   └── fire_detector.py         # Hybrid detector (YOLO + Pixel)
│   ├── training/
│   │   └── train.py                 # Training script
│   └── realtime/
│       └── realtime_detector.py     # Real-time detection
│
├── configs/
│   ├── config.yaml                  # Main configuration
│   └── training_config.yaml         # Training configuration
│
├── train_pipeline.py                # Training pipeline
├── predict_pipeline.py              # Prediction pipeline
├── main.py                          # Full system entry point
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🎯 Advanced Usage

### Full System with Drone Integration

```powershell
# Real-time detection with API server
python main.py --mode both --camera 0 --model Model/best.pt

# API server only
python main.py --mode api --model Model/best.pt

# Headless mode (no video display)
python main.py --mode realtime --no-video --camera rtsp://drone-ip/stream
```

### Custom Configuration

Edit `configs/config.yaml`:

```yaml
model:
  path: "Model/best.pt"
  confidence_threshold: 0.5
  use_half: true

camera:
  source: "0"
  fps: 30

detection:
  preprocessing_enabled: true
  temporal_filter_enabled: true
```

### Hybrid Detection Features

The fire detector supports both YOLO and pixel-based detection:

```powershell
# YOLO + Pixel detection (default)
python src/detection/fire_detector.py --source 0

# YOLO only
python src/detection/fire_detector.py --source 0 --no-pixel

# Enable denoising for noisy footage
python src/detection/fire_detector.py --source 0 --denoise
```

---

## 📊 Performance Metrics

### Validation Metrics

After training, check these metrics:

- **mAP50**: Mean Average Precision at IoU=0.50 (target: >0.80)
- **mAP50-95**: Mean Average Precision at IoU=0.50:0.95 (target: >0.60)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### View Results

```powershell
# View training plots
start fire_detection_training/yolov8s_fire/results.png

# View confusion matrix
start fire_detection_training/yolov8s_fire/confusion_matrix.png
```

---

## 🎓 Tips for 80%+ Accuracy

1. **Use YOLOv8s or larger** (not nano)
2. **Train for 100-150 epochs** minimum
3. **Use data augmentation** (enabled by default)
4. **Adjust confidence threshold** during prediction (0.25-0.35)
5. **Monitor validation metrics** during training

### If Accuracy is Low

- **Overfitting**: Increase augmentation, reduce epochs
- **Underfitting**: Train longer, use larger model
- **False positives**: Increase confidence threshold
- **Missing detections**: Decrease confidence threshold

---

## 🔄 Complete Workflow Example

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Train the model
python train_pipeline.py --model yolov8s.pt --epochs 100

# 3. Test on sample image
python predict_pipeline.py --source Dataset/fire.yolov8/test/images/sample.jpg --save

# 4. Run real-time detection
python predict_pipeline.py --source 0

# 5. Process video
python predict_pipeline.py --source test_video.mp4 --save
```

---

## 📚 Additional Resources

### Model Information

This pipeline uses **YOLOv8** from Ultralytics:
- **Hugging Face**: https://huggingface.co/Ultralytics/YOLOv8
- **Pretrained on**: COCO dataset
- **Fine-tuned on**: Your fire/smoke dataset from Roboflow

### Documentation

- **Ultralytics YOLOv8**: https://docs.ultralytics.com/
- **PyTorch**: https://pytorch.org/docs/
- **Roboflow**: https://roboflow.com/

---

## 🆘 Getting Help

### Check Installation

```powershell
# Python version
python --version

# PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Ultralytics version
python -c "from ultralytics import __version__; print(f'Ultralytics: {__version__}')"
```

### Common Issues

1. **PyTorch DLL Error**: See [Troubleshooting - Issue 1](#issue-1-pytorch-dll-error-windows)
2. **CUDA Out of Memory**: Reduce batch size or use CPU
3. **Model Not Found**: Check path to `Model/best.pt`
4. **Camera Not Working**: Try different camera index or use video file

---

## 📝 Quick Reference

### Essential Commands

```powershell
# Train model
python train_pipeline.py

# Predict on image
python predict_pipeline.py --source image.jpg --save

# Real-time camera
python predict_pipeline.py --source 0

# Process video
python predict_pipeline.py --source video.mp4 --save

# Full system
python main.py --mode both
```

### File Locations

- **Trained Model**: `Model/best.pt`
- **Dataset**: `Dataset/fire.yolov8/`
- **Training Results**: `fire_detection_training/yolov8s_fire/`
- **Predictions**: `outputs/predictions/`
- **Configuration**: `configs/config.yaml`

---

## ✅ System Requirements Checklist

- [ ] Python 3.10 or 3.11 installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed (test with `import torch`)
- [ ] Ultralytics installed (test with `from ultralytics import YOLO`)
- [ ] Dataset present in `Dataset/fire.yolov8/`
- [ ] Visual C++ Redistributables installed (Windows)
- [ ] GPU drivers installed (if using CUDA)

---

## 🎉 Ready to Start!

You're all set! Follow these steps:

1. **Activate environment**: `.\venv\Scripts\Activate.ps1`
2. **Train model**: `python train_pipeline.py`
3. **Test predictions**: `python predict_pipeline.py --source 0`

**Happy Fire Detecting! 🔥🚁**

---

**Version**: 2.0  
**Last Updated**: 2025-11-28  
**License**: See LICENSE file
