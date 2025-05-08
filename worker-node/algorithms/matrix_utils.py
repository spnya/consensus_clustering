import numpy as np

def scale_shift(mat: np.ndarray) -> np.ndarray:
    return mat - np.mean(mat)

def modularity_shift(mat: np.ndarray) -> np.ndarray:
    row_sums = mat.sum(axis=1, keepdims=True)
    total = mat.sum()
    if total == 0:
        return mat  
    expected = (row_sums @ row_sums.T) / total
    return mat - expected
