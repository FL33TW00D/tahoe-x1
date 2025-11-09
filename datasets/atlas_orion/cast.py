import anndata as ad
import numpy as np

adata = ad.read_h5ad("./QC_kd_unified_hct_merged_updated_filtered_subset.h5ad")

adata.X = adata.X.astype(np.int32)

adata.write_h5ad("./QC_kd_unified_hct_merged_updated_filtered_subset.h5ad")
