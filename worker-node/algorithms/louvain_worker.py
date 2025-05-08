import numpy as np
import time
import networkx as nx
import community as community_louvain
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score
)

def run_louvain_clustering(task_data):
    try:
        matrix = np.array(task_data['consensus_matrix'])  # expected to be similarity matrix
        N = matrix.shape[0]

        # Build weighted undirected graph
        G = nx.Graph()
        for i in range(N):
            for j in range(i + 1, N):
                weight = matrix[i, j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)

        start_time = time.time()
        partition = community_louvain.best_partition(G, weight='weight')
        exec_time = time.time() - start_time

        labels = np.zeros(N, dtype=int)
        for node, cluster in partition.items():
            labels[node] = cluster

        result = {
            'labels': labels.tolist(),
            'n_clusters': len(np.unique(labels)),
            'execution_time': exec_time
        }

        # External metrics
        if 'ground_truth' in task_data:
            gt = np.array(task_data['ground_truth'])
            result.update({
                'ari': adjusted_rand_score(gt, labels),
                'nmi': normalized_mutual_info_score(gt, labels),
                'fmi': fowlkes_mallows_score(gt, labels)
            })

        # Internal metrics (only if >1 cluster)
        if len(set(labels)) > 1 and 'original_data' in task_data:
            data = np.array(task_data['original_data'])
            result.update({
                'silhouette': silhouette_score(data, labels),
                'calinski_harabasz': calinski_harabasz_score(data, labels)
            })
        else:
            result.update({
                'silhouette': None,
                'calinski_harabasz': None
            })

        return result

    except Exception as e:
        return {'error': str(e)}
