import anndata as ad
import numpy as np

adata = ad.read_h5ad("./QC_kd_unified_vcc_emb.h5ad")
adata.obs["cell_type"] = "h1"
adata.X = adata.X.astype(np.int32)
adata.write_h5ad("./QC_kd_unified_vcc_emb.h5ad")
