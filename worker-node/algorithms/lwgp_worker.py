import numpy as np
import time
from collections import defaultdict
from sklearn.cluster import SpectralClustering
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
    cluster_members = {}
    for m, labels in enumerate(ensemble):
        for c in np.unique(labels):
            cluster_members[(m, c)] = np.where(labels == c)[0]

    cluster_eci = {}
    for (m, c), idx in cluster_members.items():
        size = len(idx)
        if size < 1:
            cluster_eci[(m, c)] = 0.0
            continue

        entropy = 0.0
        for labels_prime in ensemble:
            p, _ = np.histogram(labels_prime[idx],
                                bins=np.arange(labels_prime.max() + 2),
                                density=True)
            p = p[p > 0]
            entropy -= np.sum(p * np.log(p))
        entropy /= M
        cluster_eci[(m, c)] = np.exp(-entropy / (theta * M))

    return cluster_eci, cluster_members


def build_bipartite(cluster_eci, cluster_members, N):
    cluster_ids = list(cluster_members.keys())
    C = len(cluster_ids)

    cluster_index = {cid: N + i for i, cid in enumerate(cluster_ids)}

    size = N + C
    adj = np.zeros((size, size), dtype=float)

    for cid, members in cluster_members.items():
        w = cluster_eci[cid]
        c_idx = cluster_index[cid]
        adj[members, c_idx] = w
        adj[c_idx, members] = w

    return adj[:N + C, :N + C] 


def run_lwgp(task_data):
    """
    task_data:
      ensemble      : list of base clusterings (each length N)
      n_clusters    : desired number of object clusters
      theta         : ECI parameter (default 0.5)
      shift_type    : "scale" | "modularity" | "none"  (applied to bipartite)
      ground_truth  : optional labels for external metrics
      original_data : optional feature matrix for internal metrics
    """
    try:
        ensemble = [np.array(p) for p in task_data['ensemble']]
        n_clusters = int(task_data['n_clusters'])
        theta = float(task_data.get('theta', 0.5))
        shift_type = task_data.get('shift_type', 'none').lower()

        M, N = len(ensemble), len(ensemble[0])

        t0 = time.time()

        cluster_eci, cluster_members = compute_eci_weights(ensemble, theta)

        bipartite = build_bipartite(cluster_eci, cluster_members, N)

        if shift_type == 'scale':
            bipartite = scale_shift(bipartite)
        elif shift_type == 'modularity':
            bipartite = modularity_shift(bipartite)

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        all_labels = sc.fit_predict(bipartite)
        labels = all_labels[:N]

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
