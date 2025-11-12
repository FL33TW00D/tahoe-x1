"""
Pretrained models may not have been trained with <chem> or <genetic> token.
Add 1 if required by resizing the embedding matrix and saving a new checkpoint.
"""

import sys
import torch

checkpoint_path = sys.argv[1]
new_vocab_size = int(sys.argv[2])

print(f"Loading {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Find model state
model_state = checkpoint.get('state', {}).get('model', checkpoint)

# Resize embeddings
for key in list(model_state.keys()):
    if 'embedding.weight' in key:
        old_weight = model_state[key]
        old_size = old_weight.shape[0]
        if old_size < new_vocab_size:
            padding = torch.zeros(new_vocab_size - old_size, old_weight.shape[1])
            model_state[key] = torch.cat([old_weight, padding], dim=0)
            print(f"Resized {key}: {old_size} -> {new_vocab_size}")

# Save
output_path = checkpoint_path.replace('.pt', '-resized.pt')
torch.save(checkpoint, output_path)
print(f"Saved to {output_path}")
