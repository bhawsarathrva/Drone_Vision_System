"""
Setup script for Fire Detection System
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README if available
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="fire-detection-system",
    version="1.0.0",
    description="Fire Detection System using YOLOv8 and Drones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fire Detection Team",
    author_email="",
    url="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "python-multipart>=0.0.6",
        "geopy>=2.4.0",
        "folium>=0.15.0",
        "requests>=2.31.0",
        "pymavlink>=2.4.40",
        "loguru>=0.7.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fire-detection=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

