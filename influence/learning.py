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


def learn_network_influence_matrix(observations: np.ndarray, state_transition_matrices: dict) -> np.ndarray:
    """Learns the network influence matrix D using constrained gradient ascent with full 1-D search.

    :param observations: a sequence of observations for each site in the network
    :param state_transition_matrices: a state-transition matrix A_{ij} for each (i,j) block
    :return: the network influence matrix D
    """
    def _f(x, P, B):
        inner_prod = np.inner(x, B)
        if inner_prod == 0:
            return 0
        return -1 * sum(P / inner_prod)

    n = len(observations)
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A_ji = state_transition_matrices[(j,i)]
            # P(s_i[k]|s_j[k-1])
            P = np.array([A_ji[observations[j][k]][s] for (k,s) in enumerate(observations[i][:1])])
            A_ii = state_transition_matrices[(i,i)]
            # B' = P(s_i[k]|s_i[0] ... s_i[k]|s_i[N])
            B = np.array([A_ii[observations[i][k]][s] for (k,s) in enumerate(observations[j][:1])])
            result = optimize.minimize_scalar(_f, args=(P,B), method='bounded', bounds=(0,1))
            D[i][j] = result.x

    for i, row in enumerate(D):
        # If no sizeable influence on site i, it must only have self-influence
        if sum(row) < THRESHOLD_FOR_SELF_INFLUENCE:
            D[i][i] = 1.
    return D
