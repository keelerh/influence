import random

import numpy as np

from influence.site import Site


class InfluenceModel(object):
    """The ``InfluenceModel`` is a networked, discrete time stochastic model.
    """

    def __init__(self, sites: list[Site], D: np.ndarray, state_transition_matrices: dict[tuple[int, int], np.ndarray]):
        """Initializes an ``InfluenceModel`` with sites, a network influence matrix, and state transition matrices.

        At the network level, nodes are referred to as sites and their connections are described by the stochastic
        network matrix D. At the local level, every site has an internal Markov chain Γ(A) and at any given time
        is in one of the statuses of Γ(A). At time k, the status of site i is represented by a length-m status vector,
        an indicator vector containing a 1 in the position corresponding to its current status and a 0 everywhere else.
        For each pair of sites i and j, the state-transition matrix A_{ij} is an m_i x m_j stochastic matrix.

        :param sites: ordered list of all n sites
        :param D: network influence matrix, an n x n stochastic matrix
        :param state_transition_matrices: a state-transition matrix A_{ij} for each pair of sites i and j
        """
        for (i,j), A in state_transition_matrices.items():
            if not A.shape[0] == len(sites[i].s) and A.shape[1] == len(sites[j].s):
                raise ValueError(f'state transition matrix at ({i},{j}) block must be m_i x m_j')
            if not all(np.isclose(1, np.sum(A, axis=1))):
                raise ValueError(f'state-transition matrix at ({i},{j}) must be stochastic (all rows sum to 1)')
            if any(x < 0 for x in np.nditer(A)):
                raise ValueError(f'state-transition matrix at ({i},{j}) must be stochastic (non-negative)')

        D_transpose = np.transpose(D)
        H = self.generalized_kron(D_transpose, state_transition_matrices)

        self.sites = sites
        self.H = H  # influence matrix, i.e. generalized Kronecker product of D' and {A_{ij}}

    def __next__(self):
        """Applies the evolution equations for the influence model and updates the status of all sites.

        p'[k+1] = s'[k]H
        s'[k+1] = MultiRealize(p'[k+1])
        """
        p_transpose = np.transpose(self.get_state_vector()) @ self.H
        self.multi_realize(p_transpose)

    def get_state_vector(self) -> np.ndarray:
        """Gets a copy of the state vector formed by vector stacking the status vectors at each site.

        The state vector is a column vector of length (m_i + ... + m_n) representing the status of all
        sites in the network.

        :return: a copy of the state vector formed from the status vectors at each site
        """
        statuses = tuple(site.s for site in self.sites)
        return np.vstack(statuses).copy()

    def multi_realize(self, p_transpose: np.ndarray):
        """Performs a random realization for each row of P[k+1] and updates the status vector s at each site.

        P[k+1] is a vector stack of the p_i[k] PMF vectors that govern the status of site i at time k.

        :param arr: the transpose of the probability vector p
        """
        i = 0
        for site in self.sites:
            m = site.s.shape[0]
            rand_status = random.choices(range(m), p_transpose[0][i:i+m], k=1)[0]
            new_status = np.zeros((m,1))
            new_status[rand_status][0] = 1
            site.s = new_status
            i += m

    @staticmethod
    def generalized_kron(A: np.ndarray, Bs: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
        """Computes the generalized Kronecker product of an m x n matrix A and p x q matrices B.

        The generalized Kronecker product differs from the regular Kronecker product in that each matrix B in the
        (i,j)th block can be different.

        :param A: m x n matrix
        :param Bs: mapping from an (i,j) block to a matrix B
        :return: generalized Kronecker product of A and {B_{ij}}
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
