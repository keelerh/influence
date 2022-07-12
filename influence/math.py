import numpy as np


def generalized_kron(A: np.ndarray, B: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
    """Computes the generalized Kronecker product of an m x n matrix A and p x q matrices B.

    The generalized Kronecker product differs from the regular Kronecker product in that each matrix B in the
    (i,j)th block can be different.

    :param A: r x t matrix
    :param B: a matrix with dimension-compatible {B_{ij} | 1 <= i <= r, 1 <= j <= t} sub-matrices in each (i,j)th block
    :param num_rows: number of rows in each sub-matrix of B
    :param num_cols: number of columns in each sub-matrix of B
    :return: generalized Kronecker product of A and {B_{ij}}
    """
    r, t = A.shape
    C = np.zeros((r * num_rows, t * num_cols))
    for i in range(0, r):
        for j in range(0, t):
            B_ij = B[i * num_rows:(i * num_rows) + num_rows, j * num_cols:(j * num_cols) + num_cols]
            for k in range(0, num_rows):
                for l in range(0, num_cols):
                    C[(i * num_rows) + k, (j * num_cols) + l] = A[i][j] * B_ij[k][l]
    return C
