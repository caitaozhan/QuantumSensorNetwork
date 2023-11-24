import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from qiskit_textbook.tools import random_state
from utility import Utility
from input_output import Default
from quantum_state import QuantumState
from depolarising_noise import DepolarisingNoise


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

    @property
    def num_sensor(self):
        return self._num_sensor

    @property
    def density_matrix(self):
        return self._density_matrix

    def set_dm_via_initstate_and_depolarising_noise(self, initstate: QuantumState, depolar_noise: DepolarisingNoise):
        '''set the density matrix by the initial state and passing through the depolarising noise
        Args:
            initstate: the initial state of the quantum sensor network
            depolar_noise: depolaring noise model
        '''
        self._density_matrix = np.dot(initstate.density_matrix, depolar_noise.get_matrix(self.num_sensor))

    def evolve(self, operator: Operator):
        '''the evolution of a quantum state
        Args:
            operator: describe the interaction of the environment, essentily a matrix
        '''
        dim = self._density_matrix.shape[0]  # for N qubits, the dimension is 2**N
        operator_dim = np.product(operator.input_dims()) # for N qubits, the input_dims() return (2, 2, ..., 2), N twos.
        if dim == operator_dim:
            self._density_matrix = np.dot(operator._data, self._density_matrix)
        else:
            raise Exception('density_matrix and operator dimension not equal')
