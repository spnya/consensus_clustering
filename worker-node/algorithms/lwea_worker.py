import numpy as np
import time
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score
)
from .matrix_utils import scale_shift, modularity_shift  


def compute_eci_weights(ensemble, theta=0.5):
    M = len(ensemble)
    N = len(ensemble[0])

    cluster_members = {}
    global_id = 0
    for m, labels in enumerate(ensemble):
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            cluster_members[(m, c)] = idx
            global_id += 1

    partition_lookup = []
    for labels in ensemble:
        label2indices = defaultdict(np.ndarray)
        for lab in np.unique(labels):
            label2indices[lab] = np.where(labels == lab)[0]
        partition_lookup.append(label2indices)

    cluster_eci = {}
    for (m, c), idx in cluster_members.items():
        size = len(idx)
        if size < 1:
            cluster_eci[(m, c)] = 0.0
            continue

        entropy = 0.0
        for m_prime, labels_prime in enumerate(ensemble):
            sub_labels = labels_prime[idx]
            counts = np.bincount(sub_labels)
            p = counts[counts > 0] / size
            entropy -= np.sum(p * np.log(p))

        entropy /= M
        eci = np.exp(-entropy / (theta * M))
        cluster_eci[(m, c)] = eci

    return cluster_eci, cluster_members


def build_lwca(cluster_eci, cluster_members, N, M):
    lwca = np.zeros((N, N), dtype=float)

    for (m, c), idx in cluster_members.items():
        w = cluster_eci[(m, c)] / M 
        if len(idx) < 2:
            continue
        rows, cols = np.meshgrid(idx, idx, indexing='ij')
        lwca[rows, cols] += w

    np.fill_diagonal(lwca, 1.0)
    return lwca


def run_lwea(task_data):
    try:
        ensemble = [np.array(p) for p in task_data['ensemble']]
        n_clusters = int(task_data['n_clusters'])
        theta = float(task_data.get('theta', 0.5))
        shift_type = task_data.get('shift_type', 'none').lower()

        M, N = len(ensemble), len(ensemble[0])

        t0 = time.time()
        cluster_eci, cluster_members = compute_eci_weights(ensemble, theta=theta)

        lwca = build_lwca(cluster_eci, cluster_members, N, M)

        if shift_type == 'scale':
            lwca = scale_shift(lwca)
        elif shift_type == 'modularity':
            lwca = modularity_shift(lwca)

        dist = 1.0 - lwca
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(dist)
        exec_time = time.time() - t0

        result = {
            'labels': labels.tolist(),
            'n_clusters': len(np.unique(labels)),
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
