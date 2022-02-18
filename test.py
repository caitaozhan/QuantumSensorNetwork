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
    print(f'simulate: error of discriminating |0> and |1> in computational basis is {error}')


def test2():
    '''discriminate |0> and |+> in computational basis
    '''
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, psi=vector1)
    qs2 = QuantumState(num_detector=1, psi=vector2)
    quantum_states = [qs1, qs2]
    # priors = [0.75, 0.25]
    priors = [0.25, 0.75]
    povm = Povm()
    povm.computational_basis()

    quantum_measurement = QuantumMeasurement()
    quantum_measurement.preparation(quantum_states, priors)
    quantum_measurement.povm = povm
    error = quantum_measurement.simulate(seed=1, repeat=20_000)
    print(f'simulate: error of discriminating |0> and |+> in computational basis is {error}')

def test3():
    '''discriminate |0> and |+> in the optimal basis
    '''
    seed = 1
    repeat = 50_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, psi=vector1)
    qs2 = QuantumState(num_detector=1, psi=vector2)
    quantum_states = [qs1, qs2]
    priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    # priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors)

        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


if __name__ == '__main__':
    # test1()
    # test2()
    test3()