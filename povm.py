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
        pass

    def two_state_unambiguous(self, quantum_states: list, priors: list):
        '''for two state discrimination and unambiguous, the optimal measurement is known
        '''
        pass
