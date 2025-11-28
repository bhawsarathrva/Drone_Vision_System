import os
import sys
import argparse
from pathlib import Path
from loguru import logger
import requests
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ModelDownloader:
    def __init__(self, output_dir: str = "Model"):
        """
        Initialize model downloader
        
        Args:
            output_dir: Directory to save downloaded models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ultralytics_models = {
            'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt',
            'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt',
            'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt',
            'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt',
            'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt',
        }
        
        self.model_sizes = {
            'yolov8n.pt': '6 MB',
            'yolov8s.pt': '22 MB',
            'yolov8m.pt': '52 MB',
            'yolov8l.pt': '87 MB',
            'yolov8x.pt': '136 MB',
        }
        
        logger.info(f"Model Downloader initialized - Output: {self.output_dir}")
    
    def download_file(self, url: str, filename: str) -> bool:
        output_path = self.output_dir / filename
        
        if output_path.exists():
            logger.info(f"✓ {filename} already exists, skipping download")
            return True
        
        try:
            logger.info(f"Downloading {filename} from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"✓ Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def download_ultralytics_model(self, model_name: str) -> bool:
        """
        Download a specific Ultralytics YOLOv8 model
        
        Args:
            model_name: Name of the model to download (e.g., 'yolov8n.pt')
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.ultralytics_models:
            logger.error(f"Unknown model: {model_name}")
            logger.info(f"Available models: {list(self.ultralytics_models.keys())}")
            return False
        
        url = self.ultralytics_models[model_name]
        size = self.model_sizes.get(model_name, 'Unknown size')
        
        logger.info(f"Downloading {model_name} ({size})")
        return self.download_file(url, model_name)
    
    def download_all_ultralytics_models(self) -> dict:
        """
        Download all Ultralytics YOLOv8 models
        
        Returns:
            Dictionary with model names and download status
        """
        logger.info("="*70)
        logger.info("Downloading All Ultralytics YOLOv8 Models")
        logger.info("="*70)
        
        results = {}
        for model_name in self.ultralytics_models.keys():
            success = self.download_ultralytics_model(model_name)
            results[model_name] = success
        
        logger.info("="*70)
        logger.info("Download Summary")
        logger.info("="*70)
        for model_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"{status} {model_name}")
        
        return results
    
    def download_from_huggingface(self, model_id: str, filename: str = "best.pt") -> bool:
        """
        Download a model from Hugging Face Hub
        
        Args:
            model_id: Hugging Face model ID (e.g., 'username/model-name')
            filename: Filename to save as
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Downloading from Hugging Face: {model_id}")
            
            model_path = hf_hub_download(   
                repo_id=model_id,
                filename=filename,
                cache_dir=str(self.output_dir / "hf_cache")
            )
            
            import shutil
            output_path = self.output_dir / f"{model_id.replace('/', '_')}_{filename}"
            shutil.copy(model_path, output_path)
            
            logger.info(f"✓ Successfully downloaded to {output_path}")
            return True
            
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to download from Hugging Face: {e}")
            return False
    
    def download_from_url(self, url: str, filename: str = None) -> bool:
        if filename is None:
            filename = url.split('/')[-1]
            if not filename.endswith('.pt'):
                filename = 'custom_model.pt'
        
        return self.download_file(url, filename)
    
    def list_downloaded_models(self):
        """List all downloaded models in the output directory"""
        logger.info("="*70)
        logger.info("Downloaded Models")
        logger.info("="*70)
        
        models = list(self.output_dir.glob('*.pt'))
        
        if not models:
            logger.info("No models found in output directory")
            return
        
        for model_path in sorted(models):
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {model_path.name} ({size_mb:.1f} MB)")
        
        logger.info(f"\nTotal: {len(models)} models")
    
    def verify_model(self, model_name: str) -> bool:
        model_path = self.output_dir / model_name
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        try:
            from ultralytics import YOLO
            
            logger.info(f"Verifying {model_name}...")
            model = YOLO(str(model_path))
            
            logger.info(f"✓ {model_name} loaded successfully")
            logger.info(f"  Classes: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Download YOLOv8 models for fire detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 'all'],
        help='Ultralytics model to download'
    )
    
    parser.add_argument(
        '--hf-model',
        type=str,
        help='Hugging Face model ID (e.g., username/model-name)'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        help='Custom URL to download model from'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='Model',
        help='Output directory for models (default: Model)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all downloaded models'
    )
    
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify that a model can be loaded'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        help='Custom filename for downloaded model'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(output_dir=args.output_dir)
    
    # List models
    if args.list:
        downloader.list_downloaded_models()
        return
    
    # Verify model
    if args.verify:
        success = downloader.verify_model(args.verify)
        sys.exit(0 if success else 1)
    
    # Download operations
    success = False
    
    if args.model:
        if args.model == 'all':
            results = downloader.download_all_ultralytics_models()
            success = all(results.values())
        else:
            success = downloader.download_ultralytics_model(args.model)
    
    elif args.hf_model:
        filename = args.filename or 'best.pt'
        success = downloader.download_from_huggingface(args.hf_model, filename)
    
    elif args.url:
        success = downloader.download_from_url(args.url, args.filename)
    
    else:
        parser.print_help()
        logger.info("\nNo download option specified. Use --model, --hf-model, or --url")
        return
    
    # List downloaded models after download
    if success:
        logger.info("")
        downloader.list_downloaded_models()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
