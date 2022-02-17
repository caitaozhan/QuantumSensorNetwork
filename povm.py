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

    @property
    def operators(self):
        return self._operators

    def computational_basis(self):
        M0 = np.outer([1, 0], [1, 0])
        M1 = np.outer([0, 1], [0, 1])
        self._operators = [Operator(M0), Operator(M1)]

    def two_state_minerror(self, quantum_states: list, priors: list):
        '''for two state discrimination and minimum error, the optimal measurement is known
        '''
        X = quantum_states[0].density_matrix * priors[0] - quantum_states[1].density_matrix * priors[1]
        eigenvals, eigenvectors = np.linalg.eigh(X)   # eig?
        M0 = np.outer(eigenvectors[0], eigenvectors[0])
        M1 = np.outer(eigenvectors[1], eigenvectors[1])
        if eigenvals[0] < 0:
            M0, M1 = M1, M0
        self._operators = [Operator(M0), Operator(M1)]
        

    def two_state_unambiguous(self, quantum_states: list, priors: list):
        '''for two state discrimination and unambiguous, the optimal measurement is known
        '''
        pass
