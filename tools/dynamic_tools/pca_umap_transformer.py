from pathlib import Path
from typing import Dict, Any
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

def pca_umap_transformer(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Apply Principal Component Analysis (PCA) to reduce dimensionality and then 
    transform the data using Uniform Manifold Approximation and Projection (UMAP) 
    for visualization purposes.
    
    Args:
        data: Normalized data
    
    Returns:
        Dictionary with results
    """
    try:
        # Perform PCA on the normalized data
        pca = PCA(n_components=0.95, random_state=42)
        pca_data = pca.fit_transform(data)
        
        # Apply UMAP for further visualization
        umap_transformer = UMAP(random_state=42)
        transformed_data = umap_transformer.fit_transform(pca_data)
        
        return {
            "success": True,
            "transformed_data": transformed_data
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }