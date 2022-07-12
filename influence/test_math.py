import unittest

import numpy as np

from influence.math import generalized_kron


class TestGeneralizedKron(unittest.TestCase):

    def test_1x1_by_1x1_same_B(self):
        A = np.array([[2]])
        B = np.array([[3]])
        self.assertEqual(generalized_kron(A, B, 1, 1), np.kron(A, B))

    def test_2x2_by_2x2_same_B(self):
        A = np.array([
            [1, 2],
            [3, 4],
        ])
        B_ij = np.array([
            [0, 5],
            [6, 7],
        ])
        B = np.array([
            [0, 5, 0, 5],
            [6, 7, 6, 7],
            [0, 5, 0, 5],
            [6, 7, 6, 7],
        ])
        self.assertTrue(np.allclose(generalized_kron(A, B, 2, 2), np.kron(A, B_ij)))

    def test_2x2_by_3x3_same_B(self):
        A = np.array([
            [0.5, 1.],
            [0.5, 0.],
        ])
        B_ij = np.array([
            [1.,  0.,  0. ],
            [0.4, 0.2, 0.4],
            [0.,  0.,  1. ],
        ])
        B = np.concatenate((B_ij, B_ij), 0)
        B = np.concatenate((B, B), 1)
        self.assertTrue(np.allclose(generalized_kron(A, B, 3, 3), np.kron(A, B_ij)))

    def test_1x3_by_2x1_same_B(self):
        A = np.array([
            [1, 2, 3],
        ])
        B_ij = np.array([
            [4],
            [5],
        ])
        B = np.array([
            [4, 4, 4],
            [5, 5, 5],
        ])
        self.assertTrue(np.allclose(generalized_kron(A, B, 2, 1), np.kron(A, B_ij)))

    def test_2x2_by_2x2_different_B(self):
        A = np.array([
            [2, 8],
            [4, 5],
        ])
        B = np.array([
            [1, 2, 0, 0],
            [9, 3, 1, 1],
            [3, 2, 4, 0],
            [7, 3, 8, 0],
        ])
        C = np.array([
            [ 2,  4,  0, 0],
            [18,  6,  8, 8],
            [12,  8, 20, 0],
            [28, 12, 40, 0],
        ])
        self.assertTrue(np.allclose(generalized_kron(A, B, 2, 2), C))

if __name__ == '__main__':
    unittest.main()
