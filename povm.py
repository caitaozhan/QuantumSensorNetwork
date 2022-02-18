'''
Positive operator valued measurement
'''

import numpy as np
from quantum_state import QuantumState
from qiskit.quantum_info.operators.operator import Operator

class Povm:
    '''encapsulate positive operator valued measurement
    '''
    def __init__(self, operators: list = None):
        self._operators = operators   # a list of Operator
        self._method = ''
        self._theoretical_error = -1

    @property
    def operators(self):
        return self._operators

    @property
    def theoretical_error(self):
        return self._theoretical_error

    @property
    def method(self):
        return self._method

    def __str__(self):
        string = ''
        for M in self._operators:
            string += str(M.data) + '\n\n'
        return string

    def computational_basis(self):
        M0 = np.outer([1, 0], [1, 0])
        M1 = np.outer([0, 1], [0, 1])
        self._operators = [Operator(M0), Operator(M1)]

    def two_state_minerror(self, quantum_states: list, priors: list, debug: bool = False):
        '''for two state discrimination and minimum error, the optimal measurement is known
        '''
        X = quantum_states[0].density_matrix * priors[0] - quantum_states[1].density_matrix * priors[1]
        eigenvals, eigenvectors = np.linalg.eigh(X)
        M0 = np.outer(eigenvectors[:, 0], eigenvectors[:, 0])
        M1 = np.outer(eigenvectors[:, 1], eigenvectors[:, 1])

        if debug:
            print('X\n', X)
            print('eigenvals\n', eigenvals)
            print('eigenvectors\n', eigenvectors)
            print('X v = e v')
            print('left: ', np.dot(X, eigenvectors[:, 0]))
            print('right:', np.dot(eigenvals[0], eigenvectors[:, 0]))
            print('left: ', np.dot(X, eigenvectors[:, 1].T))
            print('right:', np.dot(eigenvals[1], eigenvectors[:, 1]))
            print('M0\n', M0)
            print('M1\n', M1)
            print('M0 + M1\n', M0 + M1)
            print('eigenvals*(M0, M1)\n', eigenvals[0]*M0 + eigenvals[1]*M1)
            print('theoretical error 1 =', 0.5 - 0.5 * np.trace(np.dot((M0 - M1), X)))
            print('theoretical error 2 =', 1 - (1 + abs(eigenvals[0]) + abs(eigenvals[1])) / 2)

        if eigenvals[0] < 0:  # positive and negative parts
            M0, M1 = M1, M0
        self._operators = [Operator(M0), Operator(M1)]
        self._theoretical_error = 1 - (1 + abs(eigenvals[0]) + abs(eigenvals[1])) / 2
        self._method = 'Minimal Error'

    def two_state_unambiguous(self, quantum_states: list, priors: list):
        '''for two state discrimination and unambiguous, the optimal measurement is known
        '''
        pass
