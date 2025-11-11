"""CLI for predicting embeddings with Tahoe-x1 models."""

from pathlib import Path
from typing import Annotated, Optional
import anndata as ad

import cyclopts
from omegaconf import OmegaConf as om

from scripts.inference.predict_embeddings import predict_embeddings

app = cyclopts.App(help="Predict embeddings using Tahoe-x1 models")


@app.default
def predict(
    file_name: Annotated[str, cyclopts.Parameter(help="Input .h5ad file path")],
    model_size: Annotated[
        str,
        cyclopts.Parameter(help="Model size: 70m, 1b, or 3b"),
    ] = "70m",
    model_dir: Annotated[
        Optional[str],
        cyclopts.Parameter(help="Local model directory")
    ] = None
) -> None:
    """Predict embeddings and save to output file.
    
    Args:
        file_name: Path to input .h5ad file
        model_size: Model size (70m, 1b, or 3b)
        output_suffix: Suffix for output file (default: _embeddings)
    """
    valid_sizes = ["70m", "1b", "3b"]
    if model_size not in valid_sizes:
        raise ValueError(f"model_size must be one of {valid_sizes}, got: {model_size}")
    
    cfg = {
        "model_name": f"Tx1-{model_size}",
        "paths": {
            "hf_repo_id": "tahoebio/Tahoe-x1",
            "hf_model_size": model_size,
            "adata_input": file_name,
            "model_dir": model_dir
        },
        "data": {
            "cell_type_key": "cell_type",
            "gene_id_key": "gene_id"
        },
        "predict": {
            "seq_len_dataset": 2048,
            "return_gene_embeddings": False,
            "use_pert_inf": True
        }
    }
    cfg = om.create(cfg)

    adata = ad.read_h5ad(file_name) 
    
    print(f"Running prediction with model size: {model_size}")
    print(f"Input file: {file_name}")
    adata_w_emb = predict_embeddings(cfg)

    # `predict_embeddings` removes any genes not within their dataset
    # this is undesirable, so we copy over the embeddings to the original adata
    adata.obsm[f"Tx1-{model_size}"] = adata_w_emb.obsm[f"Tx1-{model_size}"]
    
    input_path = Path(file_name)
    output_name = f"{input_path.stem}_tx{input_path.suffix}"
    output_path = input_path.parent / output_name
    
    print(f"Saving to: {output_path}")
    adata.write_h5ad(output_path)
    print("Done!")


if __name__ == "__main__":
    app()
