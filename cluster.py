import ollama
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import logging

def cluster_themes_with_embeddings(themes, model_name="mxbai-embed-large",
                                    method="hdbscan", **kwargs):
    """
    Unified clustering interface with multiple algorithm options.

    Args:
        themes: List of theme strings
        model_name: Ollama embedding model
        method: 'dbscan', 'hdbscan', or 'louvain'
        **kwargs: Algorithm-specific parameters

    Returns:
        A list of lists, where each inner list represents a cluster of theme indices.
        Example return: [[0, 2], [1, 4]] means themes at index 0 and 2 belong to one cluster,
        and themes at index 1 and 4 belong to another cluster.
    """
    if not themes:
        return []

    # Generate embeddings (shared across all methods)
    embeddings = _generate_embeddings(themes, model_name)

    # Route to appropriate clustering method
    if method == "hdbscan":
        return _cluster_hdbscan(embeddings, **kwargs)
    elif method == "louvain":
        return _cluster_louvain(embeddings, **kwargs)
    else:  # dbscan (fallback)
        return _cluster_dbscan(embeddings, **kwargs)

def _generate_embeddings(themes, model_name):
    """Generate embeddings using Ollama."""
    embeddings = []
    for theme in themes:
        resp = ollama.embeddings(model=model_name, prompt=theme)
        embeddings.append(resp["embedding"])
    return np.array(embeddings)

def _cluster_dbscan(embeddings, eps=0.3, min_samples=2, **kwargs):
    """DBSCAN clustering (original implementation)."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(embeddings)
    return _labels_to_clusters(labels)

def _cluster_hdbscan(embeddings, min_cluster_size=2, min_samples=1, **kwargs):
    """HDBSCAN clustering with automatic density handling."""
    try:
        import hdbscan
    except ImportError:
        logging.error("hdbscan not installed. Install with: pip install hdbscan")
        logging.info("Falling back to DBSCAN")
        return _cluster_dbscan(embeddings)

    distance_matrix = cosine_distances(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(distance_matrix)

    # Log outlier scores for debugging
    outlier_scores = clusterer.outlier_scores_
    high_outliers = np.where(outlier_scores > 0.7)[0]
    if len(high_outliers) > 0:
        logging.debug(f"High outlier themes (score > 0.7): {high_outliers.tolist()}")

    return _labels_to_clusters(labels)

def _cluster_louvain(embeddings, threshold=0.5, resolution=1.0, **kwargs):
    """Graph-based Louvain community detection."""
    try:
        import networkx as nx
        from community import community_louvain
    except ImportError:
        logging.error("networkx or python-louvain not installed")
        logging.info("Install with: pip install networkx python-louvain")
        logging.info("Falling back to DBSCAN")
        return _cluster_dbscan(embeddings)

    # Build similarity graph
    similarity_matrix = cosine_similarity(embeddings)
    n = len(embeddings)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges above similarity threshold
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            if sim >= threshold:
                G.add_edge(i, j, weight=sim)

    # Check for isolated nodes
    isolated = list(nx.isolates(G))
    if len(isolated) > 0:
        logging.debug(f"Isolated themes (no connections): {isolated}")

    if G.number_of_edges() == 0:
        logging.warning("No edges in similarity graph - all themes below threshold")
        return []

    # Run Louvain community detection
    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)

    # Convert partition to cluster lists
    clusters_dict = {}
    for node, community_id in partition.items():
        if community_id not in clusters_dict:
            clusters_dict[community_id] = []
        clusters_dict[community_id].append(node)

    # Filter out singleton clusters from isolated nodes
    clusters = [c for c in clusters_dict.values()
                if not (len(c) == 1 and c[0] in isolated)]

    logging.debug(f"Louvain found {len(clusters)} communities")
    return clusters

def _labels_to_clusters(labels):
    """Convert label array to list of cluster indices."""
    clusters = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # Noise
            continue
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        clusters.append(cluster_indices)
    return clusters
