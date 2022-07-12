import random

import numpy as np

from influence.math import generalized_kron
from influence.site import Site


class InfluenceModel(object):
    """The ``InfluenceModel`` is discrete-time, dynamic model composed of a network of n interacting nodes (or sites).

    Site i assumes one of a finite number of possible statuses at each discrete-time instant. At time k, the status of
    site i is represented by a length-m status vector s, an indicator vector containing a single 1 in the position
    corresponding to the present status, and 0 everywhere else:
        s'_i[k] = [0 ... 010 ... 1].

    Updating the status of the ith site in the influence model takes place in three stages:
        (1) Site i randomly selects one of its neighboring sites to be its determining site; site j is selected with
            probability d_{ij}.
        (2) The status of the site j at time k, s_j[k], fixes the probability vector p_i[k+1] that is used in (3) to
            randomly select the next status of site i.
        (3) The next status s_i[k+1] is realized according to p_i[k+1].
    """

    def __init__(self, sites: list[Site], D: np.ndarray, A: np.ndarray):
        """Initializes an ``InfluenceModel`` with sites, a network influence matrix, and state transition matrices.

        At the network level, nodes are referred to as sites and their connections are described by the stochastic
        network matrix D. At the local level, every site has an internal Markov chain Γ(A) and at any given time
        is in one of the statuses of Γ(A). At time k, the status of site i is represented by a length-m status vector,
        an indicator vector containing a 1 in the position corresponding to its current status and a 0 everywhere else.
        For each pair of sites i and j, the state-transition matrix A_{ij} is an m_i x m_j non-negative matrix with rows
        summing to 1.

        :param sites: an ordered list of all n sites
        :param D: the network influence matrix, an n x n stochastic matrix
        :param A: state-transition matrix with the (i,j)th block representing probabilities for a pair of sites i and j
        :raises ValueError: if either the network matrix or the state-transition matrix is malformed
        """
        if not D.shape[0] == D.shape[1]:
            raise ValueError('network matrix must be n x n')
        if not all(np.isclose(1, np.sum(D, axis=1))):
            raise ValueError('network matrix must be stochastic (all rows sum to 1)')
        if any(x < 0 for x in np.nditer(D)):
            raise ValueError('network matrix must be stochastic (non-negative)')
        if any(x < 0 for x in np.nditer(A)):
            raise ValueError('state-transition matrix must be non-negative')
        if not self._all_rows_sum_to_one(sites, A):
            raise ValueError(f'all rows must sum to 1 for the state-transition matrix at each (i,j)th block')

        D_transpose = np.transpose(D)
        m = int(A.shape[0] / len(sites))
        n = int(A.shape[1] / len(sites))
        H = generalized_kron(D_transpose, A, m, n)

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

        :param p_transpose: the transpose of the probability vector p
        """
        i = 0
        for site in self.sites:
            m = site.s.shape[0]
            rand_status = random.choices(range(m), p_transpose[0][i:i + m], k=1)[0]
            new_status = np.zeros((m, 1), dtype=int)
            new_status[rand_status][0] = 1
            site.s = new_status
            i += m

    @staticmethod
    def _all_rows_sum_to_one(sites: list[Site], A: np.ndarray) -> bool:
        """Validates that all rows in the sub-matrices of the state-influence matrix sum to one.

        :param sites: a list of all sites in the network
        :param sites: state-transition matrix
        :return: True if all rows in the sub-matrices sum to one, else False
        """
        m, n = 0, 0
        for site_i in sites:
            m_i = site_i.s.shape[0]
            for site_j in sites:
                m_j = site_j.s.shape[0]
                A_ij = A[m:m + m_i, n:n + m_j]
                if not all(np.isclose(1, np.sum(A_ij, axis=1))):
                    return False
                n += m_j
            m += m_i
            n = 0
        return True
