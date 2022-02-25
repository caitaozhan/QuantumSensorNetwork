'''
Putting things together
'''

import numpy as np
import math
from quantum_state import QuantumState
from povm import Povm
from quantum_measurement import QuantumMeasurement


def test1():
    '''minimal error discriminatation of |0> and |+>
    '''
    seed = 1
    repeat = 100_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    # vector2 = np.array([0, 1])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    priors_list = [[0.5, 0.5], [0.4, 0.6]]
    # priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors, debug=False)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test2():
    '''Unambiguous discrimination  of |0> and |+>
    '''
    seed = 1
    repeat = 100_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.00001, 0.99999], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7]]
    priors_list = [[0.4, 0.6], [0.5, 0.5]]
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


def test3():
    '''minimal error discriminatation of two random states (complex numbers introduced) don't work yet. Why?
    '''
    seed = 1
    repeat = 10_000
    # qs1 = QuantumState(num_detector=1)
    # qs2 = QuantumState(num_detector=1)
    # qs1.init_random_state(seed=1)
    # qs2.init_random_state(seed=2)
    vector1 = np.array([0.6, 0.8])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors, debug=False)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test4():
    '''Unambiguous discrimination of two random states (complex numbers introduced) don't work yet. Why?
    '''
    seed = 1
    repeat = 50_000
    # qs1 = QuantumState(num_detector=1)
    # qs2 = QuantumState(num_detector=1)
    # qs1.init_random_state(seed=1)
    # qs2.init_random_state(seed=2)
    vector1 = np.array([0.6, 0.8])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    priors_list = [[0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5]]
    priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_unambiguous(quantum_states, priors, debug=False)

        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test5():
    '''minimal error discriminatation of |00> and |++>
    '''
    seed = 1
    repeat = 50_000
    vector1 = np.array([1, 0, 0, 0])
    vector2 = np.array([0.5, 0.5, 0.5, 0.5])
    # vector1 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    # a = 1/math.sqrt(8)
    # vector2 = np.array([a, a, a, a, a, a, a, a])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors, debug=True)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    test5()
    
