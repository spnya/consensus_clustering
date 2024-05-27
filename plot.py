import matplotlib.pyplot as plt
import numpy as np

# Data

algorithms = {
    "Agglomerative": [
        [0.91, 0.07],
        [0.62, 0.34],
        [0.57, 0.29],
        [0.37, 0.28],
        [0.08, 0.10],
        [0.33, 0.24],
        [0.10, 0.13],
        [0.02, 0.06],
        [0.01, 0.01]
    ],
    "Louvain": [
        [0.58, 0.37],
        [0.29, 0.31],
        [0.65, 0.37],
        [0.46, 0.41],
        [0.25, 0.31],
        [0.54, 0.39],
        [0.31, 0.29],
        [0.26, 0.10],
        [0.01, 0.01]
    ],
    "MeanShift": [
        [0.77, 0.15],
        [0.52, 0.43],
        [0.47, 0.28],
        [0.17, 0.34],
        [0.00, 0.00],
        [0.00, 0.00],
        [0.00, 0.00],
        [0.00, 0.00],
        [0.00, 0.00]
    ],
}

fig, ax = plt.subplots(figsize=(10, 6))

for alg, data in algorithms.items():
    data = np.array(data)
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y_mean = data[:, 0]
    y_std = data[:, 1]
    ax.errorbar(x, y_mean, yerr=y_std, label=alg, marker='o', capsize=5)

ax.set_xlabel('Mutation probability')
ax.set_ylabel('Average ARI')
ax.set_title("Average ARI with power law cluster sizes, modularity shift, N=100, n=500, k=20, M=20")
ax.legend()
plt.grid(True)
plt.show()