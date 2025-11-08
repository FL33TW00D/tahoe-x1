import anndata as ad
import numpy as np

# Read in backed mode
adata = ad.read_h5ad("./QC_kd_unified_hct_merged_updated.h5ad", backed='r')
print(f"Total cells: {adata.n_obs}")

# Get mask without loading full data
mask = adata.obs["kd_eff"].values >= 0.3
keep_indices = np.where(mask)[0]
print(f"Keeping {len(keep_indices)} cells ({100*len(keep_indices)/adata.n_obs:.1f}%)")

# Process in chunks
chunk_size = 10000  # Adjust based on available RAM
n_chunks = (len(keep_indices) + chunk_size - 1) // chunk_size

chunks = []
for i in range(n_chunks):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, len(keep_indices))
    chunk_indices = keep_indices[start:end]
    
    print(f"Processing chunk {i+1}/{n_chunks}...")
    chunk = adata[chunk_indices, :].to_memory()
    chunks.append(chunk)

# Concatenate and save
print("Concatenating chunks...")
adata_filtered = ad.concat(chunks, axis=0, merge="same")
adata_filtered.write_h5ad("./QC_kd_unified_hct_merged_updated_filtered.h5ad")
print("✓ Done!")
