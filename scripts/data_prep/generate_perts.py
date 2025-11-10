import torch
import os
from fetch_perts import fetch
import json

if __name__ == "__main__":
    if not os.path.exists("./ESM2_pert_features_ensembl_22631.pt"): 
        fetch()
    perts = torch.load("./ESM2_pert_features_ensembl_22631.pt")
    print(perts.keys())
    embedding_dim = 5120
    
    # First, create the mapping
    pert_to_id = {"<pad>": 0, "non-targeting": 1}
    i = 2
    for id, _ in perts.items():
        pert_to_id[id] = i
        i += 1
    
    # Create a tensor with the correct shape
    num_perts = len(pert_to_id)
    pert_tensor = torch.zeros(num_perts, embedding_dim)
    
    # Fill in the tensor at the correct indices
    pert_tensor[0] = torch.zeros(embedding_dim)  # <pad>
    pert_tensor[1] = torch.zeros(embedding_dim)  # non-targeting
    
    for pert_id, embedding in perts.items():
        idx = pert_to_id[pert_id]
        pert_tensor[idx] = embedding
    
    print("Total number of perts: ", num_perts)
    print("Tensor shape: ", pert_tensor.shape)
    print("Sample pert_to_id items: ", list(pert_to_id.items())[:5])
    
    # Save both the mapping and the tensor
    with open("pert_to_id.json", "w") as f:
        json.dump(pert_to_id, f)

    print(pert_tensor[0:2])
    torch.save(pert_tensor, "pert_features.pt")
