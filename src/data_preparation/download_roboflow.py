"""
Download dataset from Roboflow
"""
import os
from pathlib import Path
from roboflow import Roboflow
from loguru import logger

def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    dataset_format: str = "yolov8",
    output_dir: str = "Dataset"
):
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
        dataset_format: Dataset format (yolov8, coco, etc.)
        output_dir: Output directory
    """
    try:
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(dataset_format)
        
        logger.info(f"Dataset downloaded to {dataset.location}")
        
        # Move to output directory if needed
        if output_dir and output_dir != dataset.location:
            import shutil
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(dataset.location, output_path, dirs_exist_ok=True)
            logger.info(f"Dataset copied to {output_dir}")
        
        return dataset.location
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def main():
    """Main function for downloading dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download dataset from Roboflow')
    parser.add_argument('--api-key', type=str, required=True,
                       help='Roboflow API key')
    parser.add_argument('--workspace', type=str, required=True,
                       help='Roboflow workspace name')
    parser.add_argument('--project', type=str, required=True,
                       help='Project name')
    parser.add_argument('--version', type=int, required=True,
                       help='Dataset version')
    parser.add_argument('--format', type=str, default='yolov8',
                       help='Dataset format')
    parser.add_argument('--output', type=str, default='Dataset',
                       help='Output directory')
    
    args = parser.parse_args()
    
    download_roboflow_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        dataset_format=args.format,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()

