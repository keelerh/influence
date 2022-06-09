import numpy as np


def generalized_kron(A: np.ndarray, Bs: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Computes the generalized Kronecker product of an m x n matrix A and p x q matrices B.

    The generalized Kronecker product differs from the regular Kronecker product in that each matrix B in the
    (i,j)th block can be different.

    :param A: m x n matrix
    :param Bs: mapping from an (i,j) block to a matrix B
    :returns: generalized Kronecker product of A and {B_{ij}}
    :raises ValueError: if all B matrices do not have the same number of rows and columns as one another
    """
    arbitrary_B = next(iter(Bs.values()))  # arbitrarily select a B
    p, q = arbitrary_B.shape
    if not all(x.shape[0] == p and x.shape[1] == q for x in Bs.values()):
        raise ValueError('all B matrices should be have the same # of rows and same # of columns')

    m, n = A.shape
    C = np.zeros((m * p, n * q))
    for i in range(0, m):
        for j in range(0, n):
            B = Bs[(i, j)]
            for k in range(0, p):
                for l in range(0, q):
                    C[(i * p) + k, (j * q) + l] = A[i][j] * B[k][l]
    return C
