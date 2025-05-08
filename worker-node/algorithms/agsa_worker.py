import numpy as np
import time
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score
)
from .matrix_utils import scale_shift, modularity_shift

def compute_semi_average_score(matrix, clusters):
    """Compute total semi-average similarity across all clusters"""
    total = 0.0
    for cluster in clusters:
        size = len(cluster)
        if size < 2:
            continue
        indices = list(cluster)
        submatrix = matrix[np.ix_(indices, indices)]
        intra_sum = np.sum(submatrix) - np.trace(submatrix)
        total += intra_sum / (size * (size - 1))
    return total

def run_agsa(task_data):
    try:
        matrix = np.array(task_data['consensus_matrix'])

        shift_type = task_data.get('shift_type', 'none').lower()
        if shift_type == 'scale':
            matrix = scale_shift(matrix)
        elif shift_type == 'modularity':
            matrix = modularity_shift(matrix)
        # else: no shift

        np.fill_diagonal(matrix, 0)

        N = matrix.shape[0]
        clusters = [{i} for i in range(N)]
        start_time = time.time()
        improved = True

        while improved and len(clusters) > 1:
            improved = False
            current_score = compute_semi_average_score(matrix, clusters)
            best_score = current_score
            best_pair = None

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    merged = clusters[i] | clusters[j]
                    test_clusters = [
                        clusters[k]
                        for k in range(len(clusters))
                        if k not in (i, j)
                    ] + [merged]
                    score = compute_semi_average_score(matrix, test_clusters)
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
                        improved = True

            if improved and best_pair:
                i, j = best_pair
                merged = clusters[i] | clusters[j]
                clusters = [
                    clusters[k]
                    for k in range(len(clusters))
                    if k not in (i, j)
                ] + [merged]

        final_labels = np.zeros(N, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                final_labels[idx] = label

        exec_time = time.time() - start_time

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
