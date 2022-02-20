'''
Positive operator valued measurement
'''

import math
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
        '''for two state (single detector) minimum error discrimination, the optimal POVM (projective or von Neumann) measurement is known.
           The implementation is from this paper: https://arxiv.org/pdf/1707.02571.pdf
        '''
        X = quantum_states[0].density_matrix * priors[0] - quantum_states[1].density_matrix * priors[1]
        eigenvals, eigenvectors = np.linalg.eigh(X)             # The eigen vectors are normalized 
        M0 = np.outer(eigenvectors[:, 0], eigenvectors[:, 0])
        M1 = np.outer(eigenvectors[:, 1], eigenvectors[:, 1])

        if eigenvals[0] < 0:  # positive and negative parts
            M0, M1 = M1, M0
        self._operators = [Operator(M0), Operator(M1)]
        self._theoretical_error = 1 - (1 + abs(eigenvals[0]) + abs(eigenvals[1])) / 2
        self._method = 'Minimum Error'

        if debug:
            print('\nDebug information inside self.two_state_minerror()')
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
            print('M0 * M1\n', np.dot(M0, M1))
            print('eigenvals*(M0, M1)\n', eigenvals[0]*M0 + eigenvals[1]*M1)
            print('theoretical error 1 =', 0.5 - 0.5 * np.trace(np.dot((M0 - M1), X)))
            print('theoretical error 2 =', 1 - (1 + abs(eigenvals[0]) + abs(eigenvals[1])) / 2)
            costheta = abs(np.dot(quantum_states[0].state_vector, quantum_states[1].state_vector))
            print('theoretical error 3 =', 0.5 * (1 - math.sqrt(1 - 4*priors[0]*priors[1]*costheta**2)) )
            # I found three different expressions for the theoretical value for minimum error. The three are equivalent

    def two_state_unambiguous(self, quantum_states: list, priors: list, debug=True):
        '''for two state discrimination (single detector) and unambiguous, the optimal POVM measurement is known
           The implementation is from this paper: https://iopscience.iop.org/article/10.1088/1742-6596/84/1/012001
        '''
        qs1 = quantum_states[0].state_vector
        qs2 = quantum_states[1].state_vector
        qs1_ortho = np.array([-qs1[1], qs1[0]])
        qs2_ortho = np.array([-qs2[1], qs2[0]])
        costheta = abs(np.dot(qs1, qs2))
        sintheta = abs(np.dot(qs1, qs2_ortho))
        left = costheta**2 / (1 + costheta**2)
        right = 1 / (1 + costheta**2)
        if left <= priors[0] <= right:
            q1_opt = math.sqrt(priors[1] / priors[0]) * costheta
            q2_opt = math.sqrt(priors[0] / priors[1]) * costheta
            M1 = (1 - q1_opt) / (sintheta**2) * np.outer(qs2_ortho, qs2_ortho)
            M2 = (1 - q2_opt) / (sintheta**2) * np.outer(qs1_ortho, qs1_ortho)
            identity = np.array([[1, 0], [0, 1]])
            M0 = identity - M1 - M2
            self._operators = [Operator(M1), Operator(M2), Operator(M0)]
            self._theoretical_error = 2 * math.sqrt(priors[0]*priors[1]) * costheta
        
        elif priors[0] < left:
            self._theoretical_error = priors[0] + priors[1]*costheta**2
            raise Exception('TODO')
        else: # priors[0] > right
            self._theoretical_error = priors[0]*costheta**2 + priors[1]
            raise Exception('TODO')
        self._method = 'Unambiguous'            

        if debug:
            print('\nDebug information inside self.two_state_unambiguous()')
            print('cosine  theta', costheta)
            print('sinuous theta', sintheta)
            print('qs1 ortho', qs1_ortho)
            print('qs2 ortho', qs2_ortho)
            print('q1 opt', q1_opt)
            print('q2 opt', q2_opt)
            print('M2 * qs1', M2.dot(qs1))
            print('M1 * qs2', M1.dot(qs2))
            print('M1 + M2 + M0\n', M1 + M2 + M0)
            print('left', left)
            print('right', right)
