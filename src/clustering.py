import numpy as np
import random
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, MeanShift, SpectralClustering, OPTICS
import community as community_louvain
import networkx as nx

import numpy as np


def generate_ground_truth(N, K, use_power_law=False, alpha=1.5):
    """
    Generate a ground truth partition of N objects into K clusters.

    Parameters:
    - N: Number of objects.
    - K: Number of clusters.
    - use_power_law: Boolean flag to enable power law distribution. Default is False.
    - alpha: The exponent parameter of the power law distribution. Default is 1.5.

    Returns:
    - ground_truth: An array of length N with cluster labels.
    """
    min_value = 1  # Minimum size of each cluster
    max_value = N  # Maximum size of each cluster

    flag = True
    while flag:
        # Generate K-1 random sizes
        sizes = np.random.randint(min_value, max_value, K - 1)

        # Calculate the sum of generated sizes
        sum_sizes = np.sum(sizes)

        # Check if the sum of sizes is less than N
        if sum_sizes < (N - min_value):
            # Calculate the last size to make the total sum N
            last_size = N - sum_sizes
            sizes = np.append(sizes, last_size)
            flag = False

    # Assign cluster labels based on sizes
    ground_truth = np.zeros(N, dtype=int)
    current = 0
    for k in range(K):
        ground_truth[current:current + sizes[k]] = k
        current += sizes[k]
    np.random.shuffle(ground_truth)
    return ground_truth

def mutate_partition(partition, p):
    """
    Apply mutation to a partition to generate diverse partition ensembles.

    Parameters:
    - partition: The original partition.
    - p: Mutation probability.

    Returns:
    - mutated: A mutated partition.
    """
    N = len(partition)
    mutated = partition.copy()
    for i in range(N):
        if random.random() < p:
            mutated[i] = random.choice(np.unique(partition))
    return mutated


def compute_consensus_matrix(partitions):
    """
    Compute the consensus matrix from a list of partitions.

    Parameters:
    - partitions: List of partitions.

    Returns:
    - consensus_matrix: The consensus matrix.
    """
    N = len(partitions[0])
    M = len(partitions)
    consensus_matrix = np.zeros((N, N))
    for partition in partitions:
        for i in range(N):
            for j in range(N):
                if partition[i] == partition[j]:
                    consensus_matrix[i, j] += 1
    return consensus_matrix / M


def modularity_shift(consensus_matrix):
    """
    Apply modularity shift to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.

    Returns:
    - shifted_matrix: The shifted consensus matrix.
    """
    N = consensus_matrix.shape[0]
    row_sums = np.sum(consensus_matrix, axis=1)
    total_sum = np.sum(consensus_matrix)
    expected = np.outer(row_sums, row_sums) / total_sum
    return consensus_matrix - expected


def scale_shift(consensus_matrix):
    """
    Apply scale shift to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.

    Returns:
    - shifted_matrix: The shifted consensus matrix.
    """
    mean_value = np.mean(consensus_matrix)
    return consensus_matrix - mean_value


def agglomerative_clustering(consensus_matrix, K):
    """
    Apply agglomerative clustering to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.
    - K: Number of clusters.

    Returns:
    - labels: Cluster labels for each object.
    """
    clustering = AgglomerativeClustering(n_clusters=K, linkage='average')
    return clustering.fit_predict(1 - consensus_matrix)


def louvain_clustering(consensus_matrix, K):
    """
    Apply Louvain clustering to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.
    - K: Number of clusters (used for fallback in case Louvain does not find exactly K clusters).

    Returns:
    - labels: Cluster labels for each object.
    """
    G = nx.Graph()
    N = consensus_matrix.shape[0]

    # Add nodes
    for i in range(N):
        G.add_node(i)

    # Add edges with weights
    for i in range(N):
        for j in range(i + 1, N):
            if consensus_matrix[i, j] > 0:
                G.add_edge(i, j, weight=consensus_matrix[i, j])

    # Apply Louvain method
    partition = community_louvain.best_partition(G, weight='weight')

    # Convert partition to labels array
    labels = np.zeros(N, dtype=int)
    for node, cluster in partition.items():
        labels[node] = cluster

    return labels


def birch_clustering(consensus_matrix, K):
    """
    Apply BIRCH clustering to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.
    - K: Number of clusters.

    Returns:
    - labels: Cluster labels for each object.
    """
    clustering = Birch(n_clusters=K)
    return clustering.fit_predict(1 - consensus_matrix)


def optics_clustering(consensus_matrix):
    """
    Apply OPTICS clustering to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.

    Returns:
    - labels: Cluster labels for each object.
    """
    clustering = OPTICS()
    return clustering.fit_predict(1 - consensus_matrix)


def spectral_clustering(consensus_matrix, K):
    """
    Apply Spectral clustering to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.
    - K: Number of clusters.

    Returns:
    - labels: Cluster labels for each object.
    """
    clustering = SpectralClustering(n_clusters=K)
    return clustering.fit_predict(consensus_matrix)


def meanshift_clustering(consensus_matrix):
    """
    Apply MeanShift clustering to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.

    Returns:
    - labels: Cluster labels for each object.
    """
    clustering = MeanShift()
    return clustering.fit_predict(consensus_matrix)


def kmeans_clustering(consensus_matrix, K):
    """
    Apply KMeans clustering to the consensus matrix.

    Parameters:
    - consensus_matrix: The consensus matrix.
    - K: Number of clusters.

    Returns:
    - labels: Cluster labels for each object.
    """
    clustering = KMeans(n_clusters=K)
    return clustering.fit_predict(consensus_matrix)

