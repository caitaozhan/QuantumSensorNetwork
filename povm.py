'''
Positive operator valued measurement
'''

import math
import numpy as np
from scipy.linalg import sqrtm
from qiskit.quantum_info.operators.operator import Operator
from utility import Utility

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
           Implementing paper: https://arxiv.org/pdf/1707.02571.pdf
        '''
        X = quantum_states[0].density_matrix * priors[0] - quantum_states[1].density_matrix * priors[1]
        eigenvals, eigenvectors = np.linalg.eig(X)             # The eigen vectors are normalized
        index = []
        for i, eigen in enumerate(eigenvals):
            if abs(eigen) > Utility.EPSILON:
                index.append(i)
        if len(index) != 2:
            raise Exception(f'There must be two non-zero eigenvalues, but the currently there are {len(index)} eigen values')
        eig1 = index[0]
        eig2 = index[1]

        M0 = np.outer(eigenvectors[:, eig1], np.conj(eigenvectors[:, eig1]))
        M1 = np.outer(eigenvectors[:, eig2], np.conj(eigenvectors[:, eig2]))

        if eigenvals[eig1] < 0:  # positive and negative parts
            M0, M1 = M1, M0
        self._operators = [Operator(M0), Operator(M1)]
        self._theoretical_error = 1 - (1 + abs(eigenvals[eig1]) + abs(eigenvals[eig2])) / 2
        self._method = 'Minimum Error'

        if debug:
            print('\nDebug information inside Povm.two_state_minerror()')
            print('X\n', X)
            print('eigenvals\n', eigenvals)
            print('eigenvectors\n', eigenvectors)
            print('X v = e v')
            print('left: ', np.dot(X, eigenvectors[:, eig1]))
            print('right:', np.dot(eigenvals[eig1], eigenvectors[:, eig1]))
            print('left: ', np.dot(X, eigenvectors[:, eig2]))
            print('right:', np.dot(eigenvals[eig2], eigenvectors[:, eig2]))
            print('M0\n', M0)
            print('M1\n', M1)
            print('M0 + M1\n', M0 + M1)
            print('M0 * M1\n', np.dot(M0, M1))
            print('eigenvals*(M0, M1)\n', eigenvals[eig1]*M0 + eigenvals[eig2]*M1)
            print('theoretical error 1 =', float(0.5 - 0.5 * np.trace(np.dot((M0 - M1), X))))
            print('theoretical error 2 =', 1 - (1 + abs(eigenvals[eig1]) + abs(eigenvals[eig2])) / 2)
            costheta = abs(np.dot(np.conj(quantum_states[0].state_vector), quantum_states[1].state_vector))
            print('theoretical error 3 =', 0.5 * (1 - math.sqrt(1 - 4*priors[0]*priors[1]*costheta**2)) )
            tmp = np.dot(quantum_states[0].density_matrix, quantum_states[1].density_matrix)
            print('theoretical error 4 =', 0.5 * (1 - math.sqrt(1 - 4*priors[0]*priors[1]*np.trace(tmp))) )
            # I found four different expressions for the theoretical value for minimum error. The four are equivalent
            print(f'check condition 1: M0*X*M1 = \n{np.dot(M0, np.dot(X, M1))}')
            gamma = priors[0]*np.dot(M0, quantum_states[0].density_matrix) + priors[1]*np.dot(M1, quantum_states[1].density_matrix)
            print('check condition 2: gamma - pipi^{hat}')
            for i in [0, 1]:
                print(gamma - priors[i]*quantum_states[i].density_matrix)


    def two_state_unambiguous(self, quantum_states: list, priors: list, debug=True):
        '''for two state discrimination (single detector) and unambiguous, the optimal POVM measurement is known
           Implementing paper: https://iopscience.iop.org/article/10.1088/1742-6596/84/1/012001
        '''
        qs1 = quantum_states[0].state_vector
        qs2 = quantum_states[1].state_vector
        qs1_ortho = np.array([-qs1[1], qs1[0]])
        qs2_ortho = np.array([-qs2[1], qs2[0]])
        costheta = abs(np.dot(np.conj(qs1), qs2))
        sintheta = abs(np.dot(np.conj(qs1), qs2_ortho))
        left = costheta**2 / (1 + costheta**2)
        right = 1 / (1 + costheta**2)
        if left <= priors[0] <= right:
            q1_opt = math.sqrt(priors[1] / priors[0]) * costheta
            q2_opt = math.sqrt(priors[0] / priors[1]) * costheta
            Pi1 = (1 - q1_opt) / (sintheta**2) * np.outer(qs2_ortho, np.conj(qs2_ortho))
            Pi2 = (1 - q2_opt) / (sintheta**2) * np.outer(qs1_ortho, np.conj(qs1_ortho))
            identity = np.array([[1, 0], [0, 1]])
            Pi0 = identity - Pi1 - Pi2
            self._operators = [Operator(Pi1), Operator(Pi2), Operator(Pi0)]
            self._theoretical_error = 2 * math.sqrt(priors[0]*priors[1]) * costheta
        
        elif priors[0] < left:
            self._theoretical_error = priors[0] + priors[1]*costheta**2
            print(f'left={left}, right={right}, priors[0]={priors[0]}')
            raise Exception('TODO')
        else: # priors[0] > right
            self._theoretical_error = priors[0]*costheta**2 + priors[1]
            print(f'left={left}, right={right}, priors[0]={priors[0]}')
            raise Exception('TODO')
        self._method = 'Unambiguous'            

        if debug:
            print('\nDebug information inside Povm.two_state_unambiguous()')
            print('cosine  theta', costheta)
            print('sinuous theta', sintheta)
            print('qs1 ortho', qs1_ortho)
            print('qs2 ortho', qs2_ortho)
            print('q1 opt', q1_opt)
            print('q2 opt', q2_opt)
            print('Pi2 * qs1', Pi2.dot(qs1))
            print('Pi1 * qs2', Pi1.dot(qs2))
            print('Pi1 + Pi2 + Pi0\n', Pi1 + Pi2 + Pi0)
            print('left', left)
            print('right', right)


    def pretty_good_measurement(self, quantum_states: list, priors: list, debug=True):
        '''For any given set of states, we can construct an associated measurement, the square root measurement
           Implementing paper: https://arxiv.org/pdf/0810.1970.pdf
        '''
        if len(quantum_states) != len(priors):
            raise Exception('length of quantum_states and priors are not equal')
        rho = 0
        for qs, p in zip(quantum_states, priors):
            rho += (p * qs.density_matrix)
        rho_invsqrt = np.linalg.inv(sqrtm(rho))
        self._operators = []
        for qs, p in zip(quantum_states, priors):
            Pi = p * np.dot(rho_invsqrt, np.dot(qs.density_matrix, rho_invsqrt))
            self._operators.append(Operator(Pi))
        self._method = 'Pretty Good'
        self._theoretical_error = None 
        
        if debug:
            print('\nDebug information inside Povm.pretty_good_measurement()')
            print(f'prior list {priors}')
            Utility.print_matrix('rho:', rho)
            Utility.print_matrix('rho_invsqrt:', rho_invsqrt)
            summ = 0
            string = ''
            for i, Pi in enumerate(self._operators):
                summ += Pi.data
                tmp_str = f'Pi{i}:'
                Utility.print_matrix(tmp_str, Pi.data)
                string += f'{tmp_str[:-1]} + '
            string = f'{string[:-2]}='
            Utility.print_matrix(string, summ)
            print(f'Check POVM optimality: {Utility.check_optimal(quantum_states, priors, self._operators)}')
