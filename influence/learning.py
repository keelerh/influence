import collections

import numpy as np
from scipy import optimize

THRESHOLD_FOR_SELF_INFLUENCE = 0.001


def learn_state_transition_matrix(
        observations_i: np.ndarray, observations_j: np.ndarray, m_i: int, m_j: int) -> np.ndarray:
    """Uses a maximum-likelihood estimate to reconstruct the most probable state-transition matrices {A_{ij}}.

    :params observations_i: a sequence of observations of the status of site i
    :params observations_j: a sequence of observations of the status of site j
    :params m_i: the number of possible statuses for site i
    :params m_j: the number of possible statuses for site j
    :return: a state-transition matrix A_{ij} for the (i,j)th block
    """
    A = np.zeros((m_i, m_j))
    for i in range(m_i):
        status_count = collections.Counter()
        # Skip the first status because the initial state s[0]
        # is independently chosen from some fixed distribution
        for idx, s in enumerate(observations_j[1:]):
            if observations_i[idx] == i:
                status_count[s] += 1
        num_statuses = sum(status_count.values())
        for j in range(m_j):
            if num_statuses == 0:
                A[i][j] = 0
            else:
                A[i][j] = float(status_count[j]) / num_statuses
    return A


def learn_network_influence_matrix(observations: dict, A: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
    """Learns the network influence matrix D using constrained gradient ascent with full 1-D search.

    :param observations: a mapping from sites to observations
    :param A: state-transition matrix with the (i,j)th block representing probabilities for a pair of sites i and j
    :param num_rows: number of rows in each sub-matrix of A
    :param num_cols: number of columns in each sub-matrix of A
    :return: the network influence matrix D
    """
    def _f(x, P, B):
        inner_prod = np.inner(x, B)
        if inner_prod == 0:
            return 0
        return -1 * sum(P / inner_prod)

    num_sites = len(observations)
    D = np.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(num_sites):
            A_ji = A[j * num_rows:(j * num_rows) + num_rows, i * num_cols:(i * num_cols) + num_cols]
            # P(s_i[k]|s_j[k-1])
            P = np.array([A_ji[observations[j][k]][s] for (k,s) in enumerate(observations[i][:1])])
            A_ii = A[i * num_rows:(i * num_rows) + num_rows, i * num_cols:(i * num_cols) + num_cols]
            # B' = P(s_i[k]|s_i[0] ... s_i[k]|s_i[N])
            B = np.array([A_ii[observations[i][k]][s] for (k,s) in enumerate(observations[j][:1])])
            result = optimize.minimize_scalar(_f, args=(P,B), method='bounded', bounds=(0,1))
            D[i][j] = result.x

    for i, row in enumerate(D):
        # If no sizeable influence on site i, it must only have self-influence
        if sum(row) < THRESHOLD_FOR_SELF_INFLUENCE:
            D[i][i] = 1.
    return D
