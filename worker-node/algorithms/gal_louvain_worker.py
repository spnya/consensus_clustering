import numpy as np
import time
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score
)

def compute_semi_average_similarity(matrix, labels):
    """Compute total semi-average within-cluster similarity"""
    total = 0.0
    for cluster_id in np.unique(labels):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) < 2:
            continue
        submatrix = matrix[np.ix_(indices, indices)]
        sim = np.sum(submatrix) / (len(indices) * (len(indices) - 1))
        total += sim
    return total


def run_gal_louvain(task_data):
    try:
        matrix = np.array(task_data['consensus_matrix'])
        N = matrix.shape[0]
        labels = np.arange(N)

        start_time = time.time()

        improved = True
        while improved:
            improved = False
            for i in range(N):
                current_label = labels[i]
                best_label = current_label
                best_score = compute_semi_average_similarity(matrix, labels)

                neighbor_labels = set(labels[j] for j in range(N) if j != i)
                for new_label in neighbor_labels:
                    if new_label == current_label:
                        continue
                    labels[i] = new_label
                    new_score = compute_semi_average_similarity(matrix, labels)
                    if new_score > best_score:
                        best_label = new_label
                        best_score = new_score
                        improved = True
                    labels[i] = current_label

                labels[i] = best_label

        exec_time = time.time() - start_time
        final_labels = labels.copy()

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
