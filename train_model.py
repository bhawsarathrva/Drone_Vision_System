import sys
import os
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.train import FireDetectionTrainer

def main():
    logger.info("Starting Fire Detection Model Training...")
    
    try:
        # Initialize trainer with config
        trainer = FireDetectionTrainer(config_path="configs/training_config.yaml")
        
        # Train the model
        results = trainer.train()
        
        # Save the best model
        trainer.save_model("Model/best.pt")
        
        logger.info("Training completed successfully!")
        logger.info("Model saved to Model/best.pt")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
