import numpy as np


def generate_clusters(alpha, k, V, Nk):
    cen = (alpha - 1) + 2 * (1 - alpha) * np.random.rand(k, V)
    X = np.empty((0, V))
    rf = np.zeros(sum(Nk), dtype=int)
    count = 0

    for k0 in range(k):
        nk = Nk[k0]
        R = np.arange(count, count + nk)
        rf[R] = k0 + 1
        count += nk
        sig = 0.05 + 0.05 * np.random.rand(1, V)
        Xk = np.random.randn(nk, V)
        Xk = Xk * sig
        Xk = Xk + cen[k0, :]
        X = np.vstack([X, Xk])

    return X, rf


alpha = 0.5
k = 3
V = 2
Nk = [10, 15, 20]

X, rf = generate_clusters(alpha, k, V, Nk)
print("Data points:\n", X)
print("labels:\n", rf)
