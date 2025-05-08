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

def run_kmeans_clustering(task_data):
    try:
        data = np.array(task_data['data'])
        n_clusters = int(task_data['n_clusters'])

        start_time = time.time()
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(data)
        exec_time = time.time() - start_time

        result = {
            'labels': labels.tolist(),
            'n_clusters': len(np.unique(labels)),
            'execution_time': exec_time
        }

        if 'ground_truth' in task_data:
            ground_truth = np.array(task_data['ground_truth'])
            result.update({
                'ari': adjusted_rand_score(ground_truth, labels),
                'nmi': normalized_mutual_info_score(ground_truth, labels),
                'fmi': fowlkes_mallows_score(ground_truth, labels)
            })

        if len(set(labels)) > 1:
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
