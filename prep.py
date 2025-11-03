import anndata as ad
import numpy as np

adata = ad.read_h5ad("./QC_kd_unified_targeted_emb_uint16_pretrain.h5ad")
adata.obs["cell_type"] = "iPSC"
adata.X = adata.X.astype(np.int32)
adata.write_h5ad("./QC_kd_unified_targeted_emb_uint16_pretrain.h5ad")
