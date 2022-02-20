'''
Putting things together
'''

import numpy as np
import math
from quantum_state import QuantumState
from povm import Povm
from quantum_measurement import QuantumMeasurement


def test1():
    '''minimal error discriminatation
    '''
    seed = 1
    repeat = 10_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    # vector2 = np.array([0, 1])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    # priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors, debug=True)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test2():
    '''Unambiguous discrimination'''
    seed = 1
    repeat = 50_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.00001, 0.99999], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7]]
    priors_list = [[0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5]]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    # priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_unambiguous(quantum_states, priors, debug=False)

        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


if __name__ == '__main__':
    # test1()
    test2()