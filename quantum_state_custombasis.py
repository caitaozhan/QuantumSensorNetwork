import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from qiskit_textbook.tools import random_state
from utility import Utility
from input_output import Default
from equation_generator import EquationGenerator


class QuantumStateCustomBasis:
    '''Encapsulate a (pure) quantum state, i.e. a complex vector in the Hilbert space.
       Two-level quantum state, i.e., qubits
       One quantum sensor is represented by a single qubit quantum state
       N quantum sensor are represented by a N qubit quantum state
       The coefficients of this 
    '''
    def __init__(self, num_sensor: int, custom_basis: list, state_vector_custom: np.array = None, state_vector: np.array = None):
        '''
        Args:
            num_sensor: number of sensor (i.e, detector)
            custom_basis: custom basis (not the computationali basis)
            state_vector_custom: a state vector of dimension 2**num_sensor (coefficients in the custom basis)
            state_vector: a state vector of dimension 2**num_sensor (coefficients in the computational basis)
        '''
        self._num_sensor = num_sensor
        self._custom_basis = custom_basis
        self._state_vector_custom = state_vector_custom
        self._state_vector = state_vector
        self._density_matrix = np.outer(self._state_vector, np.conj(self._state_vector)) if state_vector is not None else None
        if self._state_vector_custom is not None:
            self.custom2computational()

    @property
    def num_sensor(self):
        return self._num_sensor

    @property
    def state_vector(self):
        return self._state_vector
    
    @property
    def state_vector_custom(self):
        return self._state_vector_custom
    
    @property
    def custom_basis(self):
        return self._custom_basis

    @state_vector_custom.setter
    def state_vector_custom(self, vector: np.array):
        self._state_vector_custom = vector

    @state_vector.setter
    def state_vector(self, vector: np.array):
        self._state_vector = vector
        self._density_matrix = np.outer(self._state_vector, np.conj(self._state_vector))

    @property
    def density_matrix(self):
        return self._density_matrix

    def custom2computational(self):
        '''from coefficients in the custom basis to the coefficients in the computational basis
        '''
        self._state_vector = self._state_vector_custom[0] * self._custom_basis[0]
        for i in range(1, 2**self.num_sensor):
            self._state_vector += self._state_vector_custom[i] * self._custom_basis[i]
        self._density_matrix = np.outer(self._state_vector, np.conj(self._state_vector))


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
        self._state_vector_custom = random_state(self.num_sensor)
        self.custom2computational()
    

    def init_random_state_realnumber(self, seed: int = None):
        '''init a random quantum state with real number amplitudes'''
        if seed is not None:
            np.random.seed(seed)
        self._state_vector_custom = np.random.random(2**self.num_sensor)
        squared_sum = np.sum(np.power(self._state_vector_custom, 2))
        self._state_vector_custom /= np.sqrt(squared_sum)
        self.custom2computational()
        # print(f'init_random_state_realnumber(): {self.check_state()}')

    def init_random_state_realnumber_partition(self, seed: int, partitions: list, varying: int):
        '''init a random quantum state with real number amplitudes
           coefficients at each partition except the partition varying are the same. In partition varying, the coefficients are different.
        '''
        if seed is not None:
            np.random.seed(seed)
        self._state_vector_custom = np.random.random(2**self.num_sensor)
        for i, partition in enumerate(partitions):
            if i != varying:
                fixed = self._state_vector_custom[partition[0]]   # every coefficient in the partition equals to the first coefficient
                for j in partition:
                    self._state_vector_custom[j] = fixed
        squared_sum = np.sum(np.power(self._state_vector_custom, 2))
        self._state_vector_custom /= np.sqrt(squared_sum)
        self.custom2computational()
        # print('init_random_state_realnumber_partition', self.check_state())

    def evolve(self, operator: Operator):
        '''the evolution of a quantum state
        Args:
            operator: describe the interaction of the environment, essentily a matrix
        '''
        dim = self._state_vector.shape[0]  # for N qubits, the dimension is 2**N
        operator_dim = np.product(operator.input_dims()) # for N qubits, the input_dims() return (2, 2, ..., 2), N twos.
        if dim == operator_dim:
            self._state_vector = np.dot(operator._data, self._state_vector)
            self._density_matrix = np.outer(self._state_vector, np.conj(self._state_vector))
        else:
            raise Exception('state_vector and operator dimension not equal')

    def custombasis2string(self):
        string = ''
        for vec in self.custom_basis:
            string += '['
            for c in vec:
                real = f'{c.real:.3f}'
                imag = f'{c.imag:.3f}'
                if imag[0] != '-':
                    imag = '+' + imag
                c_str = f'{real:>6}{imag:>6}i '
                string += c_str
            string += ']\n'
        return string


    def visualize_computation_in_custombasis(self, matplotlib: bool):
        '''measure the computational state vector in the custom basis, and visualize it
        Args:
            matplotlib -- if True, do a bar plot, else just print the probs
        '''
        amplitudes = []
        for i in range(2 ** self.num_sensor):
            amplitudes.append(np.dot(np.conj(self._custom_basis[i]), self._state_vector))
        probs = np.abs(amplitudes) ** 2
        print('\nProbabilities:')
        for i, prob in enumerate(probs):
            print(f'|{i}> : {prob:.5f}')
        if matplotlib:
            import matplotlib.pyplot as plt
            X = list(range(2**self.num_sensor))
            plt.bar(X, probs)
            plt.xticks(X)
            plt.show()


    def get_symmetry_index(self) -> float:
        '''a measure of symmetry -- the sum of pairwise difference in the coefficient-squares, i.e., probabilities, in each partition
        Return:
            sym_index -- if value is zero, then perfect symmetry
        '''
        amplitudes = []
        for i in range(2 ** self.num_sensor):
            amplitudes.append(np.dot(np.conj(self._custom_basis[i]), self._state_vector))
        probs = np.abs(amplitudes) ** 2
        eg = EquationGenerator(self._num_sensor)
        symmetry_index = 0
        for i in range(self.num_sensor + 1):
            partition = eg.get_partition(i)  # ['001', '010', '100']
            n = len(partition)
            for j in range(n):
                prob_1 = probs[int(partition[j], 2)]
                for k in range(j + 1, n):
                    prob_2 = probs[int(partition[k], 2)]
                    symmetry_index += abs(prob_1 - prob_2)
        return symmetry_index


    def __str__(self):
        string = '\nCustom Basis:\n'
        # string += self.custombasis2string()
        string += '\nCoefficients in custom basis:\n'
        index = 0
        num_of_bit = math.ceil(math.log2(len(self.state_vector_custom)))
        for index, amplitude in enumerate(self.state_vector_custom):
            # state = Utility.integer2bit(index, num_of_bit)
            if type(amplitude) is np.complex128:
                real = f'{amplitude.real:.6f}'
                imag = f'{amplitude.imag:.6f}'
                string += f'|{index}>: {real:>9} {imag:>9}i\n'
            else:
                string += f'|{index}>: {amplitude:.6f}\n'
        string += '\nCoefficients in computational basis:\n'
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
        string += '---'
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
        self._density_matrix = np.outer(self._state_vector, np.conj(self._state_vector))
