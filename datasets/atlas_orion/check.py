import anndata as ad

adata = ad.read_h5ad('./QC_kd_unified_hct_merged_updated_filtered.h5ad', backed="r")
print(adata)
