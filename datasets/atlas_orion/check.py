import anndata as ad

x = ad.read_h5ad("./hek_chunks/QC_kd_unified_hek_001.h5ad", backed="r+")

print(x.X[0:50, 0:50])
