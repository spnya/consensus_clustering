import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score
)
from .matrix_utils import scale_shift, modularity_shift

def run_kmeans_clustering(task_data):
    try:
        data = np.array(task_data['data'])

        # pull n_clusters from top-level or from params
        n_clusters = task_data.get('n_clusters')
        if n_clusters is None:
            n_clusters = task_data.get('params', {}).get('n_clusters')
        if n_clusters is None:
            return {'error': "Missing 'n_clusters' parameter for kmeans"}
        n_clusters = int(n_clusters)

        # optional shifts
        shift_type = task_data.get('shift_type', 'none').lower()
        if shift_type == 'scale':
            data = scale_shift(data)
        elif shift_type == 'modularity':
            data = modularity_shift(data)

        start_time = time.time()
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(data)
        exec_time = time.time() - start_time

        # build result
        result = {
            'labels': labels.tolist(),
            'n_clusters': len(np.unique(labels)),
            'execution_time': exec_time
        }

        # supervised metrics
        if 'ground_truth' in task_data:
            gt = np.array(task_data['ground_truth'])
            result.update({
                'ari': adjusted_rand_score(gt, labels),
                'nmi': normalized_mutual_info_score(gt, labels),
                'fmi': fowlkes_mallows_score(gt, labels)
            })

        # other internal metrics
        if len(set(labels)) > 1:
            result.update({
                'silhouette': silhouette_score(data, labels),
                'calinski_harabasz': calinski_harabasz_score(data, labels)
            })
        else:
            result.update({'silhouette': None, 'calinski_harabasz': None})

        return result

    except Exception as e:
        return {'error': str(e)}
