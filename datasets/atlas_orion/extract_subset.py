import numpy as np
import anndata as ad

# Load the data in backed mode (read-only, memory efficient)
adata = ad.read_h5ad("QC_kd_unified_hct_merged_updated_filtered.h5ad", backed="r")

print(f"Original dataset: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"Unique perturbations: {adata.obs['gene_id'].nunique()}")

# Get all unique perturbations
all_perts = adata.obs["gene_id"].unique()
print(all_perts)

# Separate non-targeting from other perturbations
non_targeting = [p for p in all_perts if isinstance(p, str) and "non-targeting" in p.lower()]
other_perts = [p for p in all_perts if p not in non_targeting]

print(f"\nFound {len(non_targeting)} non-targeting control(s)")
print(f"Found {len(other_perts)} other perturbations")

# Randomly select 49 other perturbations (to make 50 total with non-targeting)
np.random.seed(42)  # For reproducibility
n_to_select = min(49, len(other_perts))
selected_perts = list(np.random.choice(other_perts, size=n_to_select, replace=False))

# Add non-targeting control(s)
selected_perts.extend(non_targeting)

print(f"\nSelected {len(selected_perts)} perturbations total")

# Filter the dataset and load into memory
adata_filtered = adata[adata.obs["gene_id"].isin(selected_perts)].to_memory()

print(f"\nFiltered dataset: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes")
print(f"Perturbations in filtered data: {adata_filtered.obs['gene_id'].nunique()}")

# Save the filtered dataset
output_file = "QC_kd_unified_hct_merged_updated_filtered_subset.h5ad"
adata_filtered.write(output_file)
print(f"\nSaved filtered dataset to: {output_file}")

# Show distribution of cells per perturbation
print("\nCells per perturbation:")
print(adata_filtered.obs["gene_id"].value_counts().head(10))
