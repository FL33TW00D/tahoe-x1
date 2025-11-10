import anndata as ad

adata = ad.read_h5ad('./QC_kd_unified_hct_merged_filtered_cast.h5ad', backed="r")
print(adata)

print(adata.obs["kd_eff"].value_counts())

nt = adata[adata.obs["target_gene"] == "non-targeting"]
print(nt.obs.head())


print(adata.obs["kd_eff"].dtype)
