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
    repeat = 50_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    # vector2 = np.array([0, 1])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    # priors_list = [[0.5, 0.5], [0.4, 0.6]]
    priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors, debug=True)
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
    # priors_list = [[0.5, 0.5], [0.4, 0.6]]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_unambiguous(quantum_states, priors, debug=False)

        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test3():
    '''minimal error discriminatation of two random states (complex numbers introduced)
    '''
    seed = 1
    repeat = 100_000
    qs1 = QuantumState(num_detector=1)
    qs2 = QuantumState(num_detector=1)
    qs1.init_random_state(seed=1)
    qs2.init_random_state(seed=2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    priors_list = [[0.4, 0.6]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors, debug=False)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test4():
    '''Unambiguous discrimination of two random states (complex numbers introduced)
    '''
    seed = 2
    repeat = 100_000
    qs1 = QuantumState(num_detector=1)
    qs2 = QuantumState(num_detector=1)
    qs1.init_random_state(seed=1)
    qs2.init_random_state(seed=2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.35, 0.65], [0.4, 0.6], [0.45, 0.55], [0.5, 0.5]]
    priors_list = [[0.4, 0.6]]
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
    seed = 2
    repeat = 100_000
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
        povm.two_state_minerror(quantum_states, priors, debug=False)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test6():
    '''pretty good measurement of |0> and |+>
    '''
    seed = 1
    repeat = 100_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_detector=1, state_vector=vector1)
    qs2 = QuantumState(num_detector=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    # priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    priors_list = [[0.5, 0.5], [0.4, 0.6]]
    for priors in priors_list:
        povm = Povm()
        povm.pretty_good_measurement(quantum_states, priors, debug=True)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


def test7():
    '''pretty good measurement of two random states, when is it optimal?
    '''
    for seed in range(10):
        qs1 = QuantumState(num_detector=1)
        qs2 = QuantumState(num_detector=1)
        qs1.init_random_state(seed=seed)
        qs2.init_random_state(seed=seed+10)
        quantum_states = [qs1, qs2]
        priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.7, 0.3], [0.95, 0.05]]
        for priors in priors_list:
            povm = Povm()
            povm.pretty_good_measurement(quantum_states, priors, debug=True)

def test8():
    '''pretty good measurement of three random states, when is it optimal?
    '''
    for seed in range(10):
        qs1 = QuantumState(num_detector=1)
        qs2 = QuantumState(num_detector=1)
        qs3 = QuantumState(num_detector=1)
        qs1.init_random_state(seed=seed)
        qs2.init_random_state(seed=seed+10)
        qs3.init_random_state(seed=seed+20)
        quantum_states = [qs1, qs2, qs3]
        priors_list = [[0.1, 0.1, 0.8], [0.2, 0.3, 0.5], [1/3, 1/3, 1/3], [0.7, 0.2, 0.1], [0.9, 0.06, 0.04]]
        for priors in priors_list:
            povm = Povm()
            povm.pretty_good_measurement(quantum_states, priors, debug=True)


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    test7()
    # test8()
    