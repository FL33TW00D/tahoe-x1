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

    pert_to_id = {"<pad>": 0, "non-targeting": 1}
    i = 2

    for id, _ in perts.items():
        pert_to_id[id] = i
        i += 1
    perts["non-targeting"] = torch.zeros(embedding_dim)
    perts["<pad>"] = torch.zeros(embedding_dim)

    print("Total number of perts: ", len(pert_to_id))
    print("Sample pert_to_id items: ", list(pert_to_id.items())[:5])

    with open("pert_to_id.json", "w") as f:
        json.dump(pert_to_id, f)

    torch.save(perts, "pert_features.pt")
