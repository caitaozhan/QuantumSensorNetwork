import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from qiskit_textbook.tools import random_state
from utility import Utility
from input_output import Default
from quantum_state import QuantumState
from quantum_noise import QuantumNoise


class QuantumStateNonPure:
    '''Encapsulate a non-pure quantum state, represented by a density matrix
       Using two-level quantum state, i.e., qubits
       One quantum sensor is represented by a single qubit quantum state
       N quantum sensor are represented by a N qubit quantum state
    '''
    def __init__(self, num_sensor: int, density_matrix: np.array = None):
        '''
        Args:
            num_sensor: number of sensor (i.e, detector)
        '''
        self._num_sensor = num_sensor
        self._density_matrix = density_matrix

    def __str__(self) -> str:
        return str(self.density_matrix)

    @property
    def num_sensor(self):
        return self._num_sensor
    
    @num_sensor.setter
    def num_sensor(self, num_sensor: int):
        self._num_sensor = num_sensor

    @property
    def density_matrix(self):
        return self._density_matrix

    @density_matrix.setter
    def density_matrix(self, density_matrix: np.array):
        self._density_matrix = density_matrix

    def check_matrix(self) -> bool:
        '''check if the trace of the density matrix equals 1
        '''
        return abs(np.trace(self.density_matrix) - 1) < Default.EPSILON

    def evolve(self, operator: Operator):
        '''the evolution of a mixed quantum state
           rho -> U rho U^{dagger}
        Args:
            operator: describe the interaction of the environment, essentily a matrix
        '''
        dim = self._density_matrix.shape[0]  # for N qubits, the dimension is 2**N
        operator_dim = np.product(operator.input_dims()) # for N qubits, the input_dims() return (2, 2, ..., 2), N twos.
        assert dim == operator_dim
        self._density_matrix = operator._data @ self._density_matrix @ np.transpose(np.conj(operator._data))
        assert self.check_matrix() == True

    def apply_quantum_noise(self, quantum_noise: QuantumNoise):
        '''apply the quantum state through a quantum noise channel
        '''
        new_density_matrix = np.zeros((2**self.num_sensor, 2**self.num_sensor), dtype=np.complex128)
        for k in quantum_noise.kraus:
            k_dagger = np.transpose(np.conj(k))
            new_density_matrix += k @ self._density_matrix @ k_dagger
        self._density_matrix = new_density_matrix
        assert self.check_matrix() == True
