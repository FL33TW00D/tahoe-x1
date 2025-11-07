import os
import sys
from omegaconf import OmegaConf as om
sys.path.insert(0, os.path.abspath('./scripts'))
from train import main


# Load configuration for fine-tuning
finetune_cfg = om.load("./configs/test_run.yaml")

# Set checkpoint path
checkpoint_path = "./best-model-70m.pt"  # Or local path
finetune_cfg.load_path = checkpoint_path

# Adjust learning rate for fine-tuning and schedular for finetuning
finetune_cfg.optimizer.lr = 1.0e-5
finetune_cfg.optimizer.weight_decay = 1.0e-6
finetune_cfg.scheduler = {}
finetune_cfg.scheduler.name = 'constant_with_warmup'
finetune_cfg.scheduler.t_warmup = '0ba'

finetune_cfg.max_duration = "30ba"

# Update save folder
finetune_cfg.save_folder = "./checkpoints/finetuned_{run_name}"

print("Fine-tuning configuration:")
print(om.to_yaml(finetune_cfg))
finetune_trainer = main(finetune_cfg)
print("Fine-tuning completed!")
