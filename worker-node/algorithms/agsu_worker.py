import numpy as np
import time
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score
)

def compute_summary_score(matrix, clusters):
    """Compute total summary score (sum of intra-cluster similarities)"""
    total = 0.0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        indices = list(cluster)
        submatrix = matrix[np.ix_(indices, indices)]
        total += np.sum(submatrix)
    return total

def run_agsu(task_data):
    try:
        # 1) Load and shift the consensus matrix so its mean is 0
        matrix = np.array(task_data['consensus_matrix'])
        matrix = matrix - np.mean(matrix)

        N = matrix.shape[0]
        clusters = [{i} for i in range(N)]  # start with singletons

        start_time = time.time()
        improved = True

        # 2) Greedy merge with early stopping
        while improved and len(clusters) > 1:
            improved = False
            current_score = compute_summary_score(matrix, clusters)
            best_score = current_score
            best_pair = None

            # Try every pair to see if merging improves the summary score
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    merged = clusters[i] | clusters[j]
                    test_clusters = [
                        clusters[k]
                        for k in range(len(clusters))
                        if k not in (i, j)
                    ] + [merged]
                    score = compute_summary_score(matrix, test_clusters)
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
                        improved = True

            # If an improving merge was found, apply it
            if improved and best_pair:
                i, j = best_pair
                merged = clusters[i] | clusters[j]
                clusters = [
                    clusters[k]
                    for k in range(len(clusters))
                    if k not in (i, j)
                ] + [merged]

        # 3) Build final label array
        final_labels = np.zeros(N, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                final_labels[idx] = label

        exec_time = time.time() - start_time

        # 4) Assemble result dict
        result = {
            'labels': final_labels.tolist(),
            'n_clusters': len(np.unique(final_labels)),
            'execution_time': exec_time
        }

        if 'ground_truth' in task_data:
            gt = np.array(task_data['ground_truth'])
            result.update({
                'ari': adjusted_rand_score(gt, final_labels),
                'nmi': normalized_mutual_info_score(gt, final_labels),
                'fmi': fowlkes_mallows_score(gt, final_labels)
            })

        if len(set(final_labels)) > 1 and 'original_data' in task_data:
            data = np.array(task_data['original_data'])
            result.update({
                'silhouette': silhouette_score(data, final_labels),
                'calinski_harabasz': calinski_harabasz_score(data, final_labels)
            })
        else:
            result.update({
                'silhouette': None,
                'calinski_harabasz': None
            })

        return result

    except Exception as e:
        return {'error': str(e)}
