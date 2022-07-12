import numpy as np


class Site(object):
    """The ``Site`` class is a representation of a node at the network level.
    """

    def __init__(self, label: str, s: np.ndarray):
        """Initializes a ``Site`` with a human-readable label and an indicator vector representing its status.

        Within each site is an m-status Markov chain Γ(A). At any given time, the site is in one of the statuses of
        Γ(A). The status of the site evolves with transition probabilities that depend on its current status and the
        current statuses of its surrounding neighbors.

        At time k, the status of site i is represented by a length-m status vector s, an indicator vector containing a
        single 1 in the position corresponding to the present status, and 0 everywhere else:
            s'_i[k] = [0 ... 010 ... 1].

        :param label: a label for the site
        :param s: the length-m indicator vector for the status of the site
        :raises ValueError: if the status vector s is not a column vector
        """
        if len(s.shape) < 2 or s.shape[1] != 1:
            raise ValueError('s must be a column vector')

        self.label = label
        self.s = s
