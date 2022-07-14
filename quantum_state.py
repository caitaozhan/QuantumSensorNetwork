import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from qiskit_textbook.tools import random_state
from utility import Utility
from input_output import Default


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

    @state_vector.setter
    def state_vector(self, vector: np.array):
        self._state_vector = vector

    @property
    def density_matrix(self):
        if self._density_matrix is None:
            if self._state_vector is None:
                raise Exception('state_vector is None!')
            self._density_matrix = np.outer(self._state_vector, np.conj(self._state_vector))  # don't forget the conjugate ...
        return self._density_matrix

    def check_state(self):
        '''check if the amplitudes norm_squared add up to one
        '''
        summ = 0
        for amp in self._state_vector:
            summ += Utility.norm_squared(amp)
        return True if abs(summ - 1) < Default.EPSILON else False

    def normalize_state(self, state: np.array):
        '''Normalize a state vector
        Return:
            np.array -- the normalized state
        '''
        state_copy = np.array(state)
        magnitude_squared = 0
        for a in state_copy:
            magnitude_squared += abs(a)**2
        return state_copy / np.sqrt(magnitude_squared)

    def generate_random_direction(self):
        '''generate a random direction
        '''
        real = 2 * np.random.random() - 1
        imag = 2 * np.random.random() - 1
        direction = real + 1j*imag
        direction /= abs(direction)     # normalize
        return direction

    def init_random_state(self, seed: int = None):
        '''init a random quantum state'''
        if seed is not None:
            np.random.seed(seed)
        self._state_vector = random_state(self.num_sensor)

    def init_random_state_realnumber(self, seed: int = None):
        '''init a random quantum state with real number amplitudes'''
        if seed is not None:
            np.random.seed(seed)
        self._state_vector = np.random.random(2**self.num_sensor)
        squared_sum = np.sum(np.power(self._state_vector, 2))
        self._state_vector /= np.sqrt(squared_sum)

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
                real = f'{amplitude.real:.6f}'
                imag = f'{amplitude.imag:.6f}'
                string += f'|{state}>: {real:>9} {imag:>9}i\n'
            else:
                string += f'|{state}>: {amplitude}\n'
        return string

    def set_statevector_from_str(self, s: str):
        '''set the self._state_vector from the __str__() string
        Args:
            s -- the __str__() string
            Example:

            Guess
            Initial state:
            |000>:  0.31088  0.3194i
            |001>:  0.42490 -0.0000i
            |010>:  0.42490 -0.0000i
            |011>: -0.19850  0.2039i
            |100>:  0.42490 -0.0000i
            |101>: -0.19850  0.2039i
            |110>: -0.19850  0.2039i
            |111>: -0.00349 -0.1295i
        '''
        statevector = []
        s = s.split('\n')
        for line in s:
            if '>:' in line:
                line = line.split()
                real = line[1]
                imag = line[2]
                real = float(real.strip())
                imag = float(imag[:-1].strip())
                statevector.append(complex(real, imag))
        self._state_vector = np.array(statevector)
