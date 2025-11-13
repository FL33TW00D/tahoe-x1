import torch
import os
from fetch_gps import fetch
import json

if __name__ == "__main__":
    if not os.path.exists("./ESM2_pert_features_ensembl_22631.pt"): 
        fetch()
    gps = torch.load("./ESM2_pert_features_ensembl_22631.pt")
    print(gps.keys())
    embedding_dim = 5120
    
    # First, create the mapping
    gp_to_id = {"<pad>": 0, "non-targeting": 1}
    i = 2
    for id, _ in gps.items():
        gp_to_id[id] = i
        i += 1
    
    # Create a tensor with the correct shape
    num_gps = len(gp_to_id)
    gp_tensor = torch.zeros(num_gps, embedding_dim)
    
    # Fill in the tensor at the correct indices
    gp_tensor[0] = torch.zeros(embedding_dim)  # <pad>
    gp_tensor[1] = torch.zeros(embedding_dim)  # non-targeting
    
    for gp_id, embedding in gps.items():
        idx = gp_to_id[gp_id]
        gp_tensor[idx] = embedding
    
    print("Total number of gps: ", num_gps)
    print("Tensor shape: ", gp_tensor.shape)
    print("Sample gp_to_id items: ", list(gp_to_id.items())[:5])
    
    # Save both the mapping and the tensor
    with open("gp_to_id.json", "w") as f:
        json.dump(gp_to_id, f)

    print(gp_tensor[0:2])
    torch.save(gp_tensor, "gp_features.pt")
