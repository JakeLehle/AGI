from pathlib import Path
from typing import Dict, Any
import pandas as pd
import leidenalg
import networkx as nx

def LeidenClusteringTool(data: pd.DataFrame, resolution: float = 1.0) -> Dict[str, Any]:
    """
    Apply the Leiden clustering algorithm to group cells based on their transcriptomic profiles.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the transcriptomic profiles of cells.
        resolution (float, optional): The resolution parameter for the Leiden clustering algorithm. Defaults to 1.0.

    Returns:
        Dict[str, Any]: A dictionary with 'success' key and relevant data.
    """
    try:
        # Create a graph from the data
        G = nx.Graph()
        G.add_nodes_from(range(len(data)))
        
        # Add edges between cells based on their similarities
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                similarity = 1 - ((data.iloc[i] != data.iloc[j]).sum() / len(data.columns))
                if similarity > 0:
                    G.add_edge(i, j, weight=similarity)
        
        # Apply the Leiden clustering algorithm
        partition = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, resolution=resolution)
        
        # Get cluster labels for each cell
        clusters = [partition[i] for i in range(len(data))]
        
        return {
            "success": True,
            "clusters": clusters
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }