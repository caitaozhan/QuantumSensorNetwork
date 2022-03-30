'''
Optimizing initial state
'''

import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from quantum_state import QuantumState
from utility import Utility


class OptimizeInitialState(QuantumState):
    '''Optimize the initial state
    '''
    def __init__(self, num_sensor: int):
        super().__init__(num_sensor=num_sensor, state_vector=None)
        self._optimze_method = ''

    @property
    def optimize_method(self):
        return self._optimze_method

    def __str__(self):
        s = f'Optimization method is {self.optimize_method}.\nInitial state is:\n'
        parent = super().__str__()
        return s + parent

    def check_state(self):
        '''check if the amplitudes norm_squared add up to one
        '''
        summ = 0
        for amp in self._state_vector:
            summ += Utility.norm_squared(amp)
        return True if abs(summ - 1) < Utility.EPSILON else False

    def normalize_state(self):
        '''Normalize a state vector if not normalized
        '''
        pass


    def guess(self, unitary_operator: Operator):
        '''do an eigenvalue decomposition, the two eigen vectors are |v> and |u>,
           then the guessed initial state is 1/sqrt(2) * (|v>|u> + |u>|v>) 
        Args:
            unitary_opeartor: describe the interaction with the environment
        '''
        e_vals, e_vectors = np.linalg.eig(unitary_operator._data)
        theta1 = Utility.get_theta(e_vals[0].real, e_vals[0].imag)
        theta2 = Utility.get_theta(e_vals[1].real, e_vals[1].imag)
        v1 = e_vectors[:, 0]    # v1 is positive
        v2 = e_vectors[:, 1]    # v2 is negative
        if theta1 < theta2:
            v1, v2, = v2, v1

        tensors = []
        for i in range(self.num_sensor):
            j = 0
            tensor = 1
            while j < self.num_sensor:
                if j == i:
                    tensor = np.kron(tensor, v1)  # kron is tensor product
                else:
                    tensor = np.kron(tensor, v2)
                j += 1
            tensors.append(tensor)
        self._state_vector = 1/math.sqrt(self.num_sensor) * np.sum(tensors, axis=0)
        if self.check_state() is False:
            raise Exception(f'{self} is not a valid quantum state')
        self._optimze_method = 'Guess'
