import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from qiskit_textbook.tools import random_state
from utility import Utility


class QuantumState:
    '''Encapsulate a (pure) quantum state, i.e. a complex vector in the Hilbert space.
       Currently using two-level quantum state, i.e., qubits
       One quantum sensor is represented by a single qubit quantum state
       N quantum sensor are represented by a N qubit quantum state
    '''
    def __init__(self, num_sensor: int, state_vector: np.array = None):
        '''
        Args:
            num_sensor: number of sensor (i.e, detector)
            state_vector: a state vector of dimension 2**num_sensor
        '''
        self._num_sensor = num_sensor
        self._state_vector = state_vector
        self._density_matrix = None

    @property
    def num_sensor(self):
        return self._num_sensor

    @property
    def state_vector(self):
        return self._state_vector

    @property
    def density_matrix(self):
        if self._density_matrix is None:
            if self._state_vector is None:
                raise Exception('state_vector is None!')
            self._density_matrix = np.outer(self._state_vector, np.conj(self._state_vector))  # don't forget the conjugate ...
        return self._density_matrix

    def init_random_state(self, seed: int = None):
        '''init a random quantum state'''
        if seed is not None:
            np.random.seed(seed)
        self._state_vector = random_state(self.num_sensor)

    def evolve(self, operator: Operator):
        '''the evolution of a quantum state
        Args:
            operator: describe the interaction of the environment, essentily a matrix
        '''
        dim = self._state_vector.shape[0]  # for N qubits, the dimension is 2**N
        operator_dim = np.product(operator.input_dims()) # for N qubits, the input_dims() return (2, 2, ..., 2), N twos.
        if dim == operator_dim:
            self._state_vector = np.dot(operator._data, self._state_vector)
        else:
            raise Exception('state_vector and operator dimension not equal')

    def __str__(self):
        string = ''
        index = 0
        num_of_bit = math.ceil(math.log2(len(self.state_vector)))
        for index, amplitude in enumerate(self.state_vector):
            state = Utility.integer2bit(index, num_of_bit)
            if type(amplitude) is np.complex128:
                real = f'{amplitude.real:.5f}'
                imag = f'{amplitude.imag:.4f}'
                string += f'|{state}>: {real:>8}{imag:>8}i\n'
            else:
                string += f'|{state}>: {amplitude}\n'
        return string

