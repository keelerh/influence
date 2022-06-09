import unittest

import numpy as np

from influence.math import generalized_kron


class TestGeneralizedKron(unittest.TestCase):

    def test_1x1_by_1x1_same_B(self):
        A = np.array([[2]])
        B = np.array([[3]])
        Bs = {(0,0): B}
        self.assertEqual(generalized_kron(A, Bs), np.kron(A, B))

    def test_2x2_by_2x2_same_B(self):
        A = np.array([
            [1, 2],
            [3, 4],
        ])
        B = np.array([
            [0, 5],
            [6, 7],
        ])
        Bs = {(0,0): B, (0,1): B, (1,0): B, (1,1): B}
        self.assertTrue(np.allclose(generalized_kron(A, Bs), np.kron(A, B)))

    def test_1x3_by_2x1_same_B(self):
        A = np.array([
            [1, 2, 3],
        ])
        B = np.array([
            [4],
            [5],
        ])
        Bs = {(0,0): B, (0,1): B, (0,2): B}
        self.assertTrue(np.allclose(generalized_kron(A, Bs), np.kron(A, B)))

    def test_2x2_by_2x2_different_B(self):
        A = np.array([
            [2, 8],
            [4, 5],
        ])
        B1 = np.array([
            [1, 2],
            [9, 3],
        ])
        B2 = np.array([
            [0, 0],
            [1, 1],
        ])
        B3 = np.array([
            [3, 2],
            [7, 3],
        ])
        B4 = np.array([
            [4, 0],
            [8, 0],
        ])
        Bs = {(0,0): B1, (0,1): B2, (1,0): B3, (1,1): B4}
        C = np.array([
            [ 2,  4,  0, 0],
            [18,  6,  8, 8],
            [12,  8, 20, 0],
            [28, 12, 40, 0],
        ])
        self.assertTrue(np.allclose(generalized_kron(A, Bs), C))

    def test_differently_shaped_Bs(self):
        A = np.array([[2]])
        B1 = np.array([
            [1, 0],
        ])
        B2 = np.array([
            [0],
            [1],
        ])
        Bs = {(0,0): B1, (0,1): B2}
        self.assertRaises(ValueError, generalized_kron, A, Bs)

if __name__ == '__main__':
    unittest.main()
