'''
Optimizing initial state
'''

import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from utility import Utility

class InitialState:
    '''Optimize the initial state
    '''
    def __init__(self, num_sensor: int):
        self._num_sensor = num_sensor   # number of sensors = number of qubits
        self._initial_state = None      # the initial state for the sensors

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def num_sensor(self):
        return self._num_sensor

    def guess(self, unitary_operator: Operator):
        '''do an eigenvalue decomposition, the two eigen vectors are |v> and |u>,
           then the guessed initial state is 1/sqrt(2) * (|v>|u> + |u>|v>) 
        Args:
            unitary_opeartor: describe the interaction with the environment
        '''
        e_vals, e_vectors = np.linalg.eig(unitary_operator)
        v1 = e_vectors[:, 0]    # v1 is positive
        v2 = e_vectors[:, 1]    # v2 is negative
        theta1 = Utility.get_theta(e_vals[0].real, e_vals[0].imag)
        theta2 = Utility.get_theta(e_vals[1].real, e_vals[1].imag)
        if theta1 < theta2:
            v1, v2, = v2, v1

        # self._initial_state = 1/math.sqrt(2) * ()
        tensors = []
        for i in range(self.num_sensor):
            j = 0
            tensor = 1
            while j < self.num_sensor:
                if j == i:
                    tensor = np.kron(tensor, v1)
                else:
                    tensor = np.kron(tensor, v2)
                j += 1
            tensors.append(tensor)
        self._initial_state = 1/math.sqrt(self.num_sensor) * np.sum(tensors)
