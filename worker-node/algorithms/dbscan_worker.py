import numpy as np
import time
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score
)
from .matrix_utils import scale_shift, modularity_shift

def run_dbscan_clustering(task_data):
    try:
        matrix = np.array(task_data['data'])
        eps = float(task_data.get('eps', 0.5))
        min_samples = int(task_data.get('min_samples', 5))

        shift_type = task_data.get('shift_type', 'none').lower()
        if shift_type == 'scale':
            matrix = scale_shift(matrix)
        elif shift_type == 'modularity':
            matrix = modularity_shift(matrix)

        start_time = time.time()
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(matrix)
        exec_time = time.time() - start_time

        result = {
            'labels': labels.tolist(),
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'execution_time': exec_time
        }

        if 'ground_truth' in task_data:
            gt = np.array(task_data['ground_truth'])
            result.update({
                'ari': adjusted_rand_score(gt, labels),
                'nmi': normalized_mutual_info_score(gt, labels),
                'fmi': fowlkes_mallows_score(gt, labels)
            })

        if len(set(labels)) > 1 and 'original_data' in task_data:
            data = np.array(task_data['original_data'])
            result.update({
                'silhouette': silhouette_score(data, labels),
                'calinski_harabasz': calinski_harabasz_score(data, labels)
            })
        else:
            result.update({'silhouette': None, 'calinski_harabasz': None})

        return result

    except Exception as e:
        return {'error': str(e)}
