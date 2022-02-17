'''
Putting things together
'''

import numpy as np
import math
from quantum_state import QuantumState
from povm import Povm
from quantum_measurement import QuantumMeasurement


def test1():
    '''discriminate |0> and |1> in computational basis
    '''
    vector1 = np.array([1, 0])
    vector2 = np.array([0, 1])
    qs1 = QuantumState(num_detector=1, psi=vector1)
    qs2 = QuantumState(num_detector=1, psi=vector2)
    quantum_states = [qs1, qs2]
    priors = [0.5, 0.5]
    povm = Povm()
    povm.computational_basis()

    quantum_measurement = QuantumMeasurement()
    quantum_measurement.preparation(quantum_states, priors)
    quantum_measurement.povm = povm
    error = quantum_measurement.simulate(seed=0, repeat=10_000)
    print(f'error of discriminating |0> and |1> in computational basis through simulation is {error}')


def test2():
    '''discriminate |0> and |+> in computational basis
    '''
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, psi=vector1)
    qs2 = QuantumState(num_detector=1, psi=vector2)
    quantum_states = [qs1, qs2]
    priors = [0.5, 0.5]
    povm = Povm()
    povm.computational_basis()

    quantum_measurement = QuantumMeasurement()
    quantum_measurement.preparation(quantum_states, priors)
    quantum_measurement.povm = povm
    error = quantum_measurement.simulate(seed=1, repeat=10_000)
    print(f'error of discriminating |0> and |+> in computational basis through simulation is {error}')

def test3():
    '''discriminate |0> and |+> in computational basis
    '''
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, psi=vector1)
    qs2 = QuantumState(num_detector=1, psi=vector2)
    quantum_states = [qs1, qs2]
    priors = [0.5, 0.5]
    povm = Povm()
    povm.two_state_minerror(quantum_states, priors)

    quantum_measurement = QuantumMeasurement()
    quantum_measurement.preparation(quantum_states, priors)
    quantum_measurement.povm = povm
    error = quantum_measurement.simulate(seed=1, repeat=10_000)
    print(f'error of discriminating |0> and |+> in computational basis through simulation is {error}')


if __name__ == '__main__':
    test1()
    test2()
    # test3()