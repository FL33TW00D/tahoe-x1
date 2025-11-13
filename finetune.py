import os
import sys
from omegaconf import OmegaConf as om
sys.path.insert(0, os.path.abspath('./scripts'))
from train import main

# Load configuration for fine-tuning
finetune_cfg = om.load("./configs/finetune_orion_3b.yaml")

# Set checkpoint path
checkpoint_path = "./best-model-3b.pt"  # Or local path
finetune_cfg.load_path = checkpoint_path
finetune_cfg.load_strict_model_weights = False

# Update save folder
finetune_cfg.save_folder = "./checkpoints/finetuned_{run_name}"

print("Fine-tuning configuration:")
print(om.to_yaml(finetune_cfg))
finetune_trainer = main(finetune_cfg)
print("Fine-tuning completed!")
