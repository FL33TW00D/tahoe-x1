import anndata as ad

adata = ad.read_h5ad("./QC_kd_unified_hct_merged_updated.h5ad")
print(f"Before filtering: {adata.shape}")
adata = adata[adata.obs["kd_eff"] >= 0.3, :]
print(f"After filtering: {adata.shape}")

print(adata)

print(adata.var.head(20))
print(adata.obs.head(20))
print(adata.obs["kd_eff"].value_counts())

adata.write_h5ad("./QC_kd_unified_hct_merged_updated_filtered.h5ad")
