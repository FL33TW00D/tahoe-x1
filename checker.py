import anndata as ad
import numpy as np

x = ad.read_h5ad("./QC_kd_unified_vcc_state_tx.h5ad")
print(x)
embeddings = x.obsm["Tx1-3b"]

print("\n" + "="*60)
print("EMBEDDING STATISTICS")
print("="*60)

# Basic shape information
print(f"\nShape: {embeddings.shape}")
print(f"Number of observations: {embeddings.shape[0]}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Data type: {embeddings.dtype}")
print(f"Memory usage: {embeddings.nbytes / (1024**2):.2f} MB")

# Distribution statistics
print(f"\n{'Statistic':<20} {'Value':<15}")
print("-" * 35)
print(f"{'Mean':<20} {np.mean(embeddings):.6f}")
print(f"{'Std deviation':<20} {np.std(embeddings):.6f}")
print(f"{'Min':<20} {np.min(embeddings):.6f}")
print(f"{'Max':<20} {np.max(embeddings):.6f}")
print(f"{'Median':<20} {np.median(embeddings):.6f}")
print(f"{'25th percentile':<20} {np.percentile(embeddings, 25):.6f}")
print(f"{'75th percentile':<20} {np.percentile(embeddings, 75):.6f}")

# Per-dimension statistics
print(f"\nPer-dimension statistics:")
print(f"{'Statistic':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 68)
dim_means = np.mean(embeddings, axis=0)
dim_stds = np.std(embeddings, axis=0)
dim_mins = np.min(embeddings, axis=0)
dim_maxs = np.max(embeddings, axis=0)

print(f"{'Across dimensions':<20} {np.mean(dim_means):.6f}    {np.mean(dim_stds):.6f}    {np.mean(dim_mins):.6f}    {np.mean(dim_maxs):.6f}")
print(f"{'Range (min-max)':<20} [{dim_means.min():.4f}, {dim_means.max():.4f}]")

# Check for special values
n_nan = np.isnan(embeddings).sum()
n_inf = np.isinf(embeddings).sum()
n_zero = (embeddings == 0).sum()

print(f"\nSpecial values:")
print(f"{'NaN values':<20} {n_nan} ({n_nan/embeddings.size*100:.4f}%)")
print(f"{'Inf values':<20} {n_inf} ({n_inf/embeddings.size*100:.4f}%)")
print(f"{'Zero values':<20} {n_zero} ({n_zero/embeddings.size*100:.4f}%)")

# Norm statistics
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nL2 norm statistics (per observation):")
print(f"{'Mean norm':<20} {np.mean(norms):.6f}")
print(f"{'Std norm':<20} {np.std(norms):.6f}")
print(f"{'Min norm':<20} {np.min(norms):.6f}")
print(f"{'Max norm':<20} {np.max(norms):.6f}")

# Sparsity
sparsity = (embeddings == 0).sum() / embeddings.size * 100
print(f"\nSparsity: {sparsity:.4f}%")

# Variance explained by dimensions
variances = np.var(embeddings, axis=0)
variance_sum = np.sum(variances)
print(f"\nVariance per dimension:")
print(f"{'Total variance':<20} {variance_sum:.6f}")
print(f"{'Top 5 dimensions':<20} {np.sort(variances)[-5:][::-1]}")
cumulative_var = np.cumsum(np.sort(variances)[::-1]) / variance_sum * 100
dims_for_90pct = np.argmax(cumulative_var >= 90) + 1
print(f"{'Dims for 90% var':<20} {dims_for_90pct} / {embeddings.shape[1]}")

print("\n" + "="*60)
