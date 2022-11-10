'''
Putting things together
'''

import copy
import numpy as np
import math
from quantum_state import QuantumState
from povm import Povm
from quantum_measurement import QuantumMeasurement
from utility import Utility
from optimize_initial_state import OptimizeInitialState


# minimal error discriminatation of |0> and |+>
def test1():
    seed = 1
    repeat = 50_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    # vector2 = np.array([0, 1])
    qs1 = QuantumState(num_sensor=1, state_vector=vector1)
    qs2 = QuantumState(num_sensor=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    # priors_list = [[0.5, 0.5], [0.4, 0.6]]
    # priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.two_state_minerror(quantum_states, priors, debug=True)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


# Unambiguous discrimination of |0> and |+>
def test2():
    seed = 1
    repeat = 100_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_sensor=1, state_vector=vector1)
    qs2 = QuantumState(num_sensor=1, state_vector=vector2)
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


# minimal error discriminatation of two random states (complex numbers introduced)
def test3():
    seed = 1
    repeat = 100_000
    qs1 = QuantumState(num_sensor=1)
    qs2 = QuantumState(num_sensor=1)
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


# Unambiguous discrimination of two random states (complex numbers introduced)
def test4():
    seed = 2
    repeat = 100_000
    qs1 = QuantumState(num_sensor=1)
    qs2 = QuantumState(num_sensor=1)
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


# minimal error discriminatation of |00> and |++>
def test5():
    seed = 2
    repeat = 100_000
    vector1 = np.array([1, 0, 0, 0])
    vector2 = np.array([0.5, 0.5, 0.5, 0.5])
    # vector1 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    # a = 1/math.sqrt(8)
    # vector2 = np.array([a, a, a, a, a, a, a, a])
    qs1 = QuantumState(num_sensor=1, state_vector=vector1)
    qs2 = QuantumState(num_sensor=1, state_vector=vector2)
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


# pretty good measurement of |0> and |+>
def test6():
    seed = 1
    repeat = 100_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_sensor=1, state_vector=vector1)
    qs2 = QuantumState(num_sensor=1, state_vector=vector2)
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


# pretty good measurement of two random states, when is it optimal?
def test7():
    for seed in range(10):
        qs1 = QuantumState(num_sensor=4)
        qs2 = QuantumState(num_sensor=4)
        qs3 = QuantumState(num_sensor=4)
        qs4 = QuantumState(num_sensor=4)
        qs1.init_random_state(seed=seed)
        qs2.init_random_state(seed=seed+10)
        qs3.init_random_state(seed=seed+20)
        qs4.init_random_state(seed=seed+30)
        quantum_states = [qs1, qs2, qs3, qs4]
        priors_list = [[1/4] * 4]
        for priors in priors_list:
            povm = Povm()
            povm.pretty_good_measurement(quantum_states, priors, debug=True)


# pretty good measurement of three random states, when is it optimal?
def test8():
    seed = 0
    quantum_states = []
    for i in range(16):
    # for i in range(3):
    # for i in range(8):
    # for i in range(4):
        qs = QuantumState(num_sensor=5)
        qs.init_random_state(seed=seed + i)
        quantum_states.append(qs)
    priors_list = [[1/16] * 16]
    # priors_list = [[1/3] * 3]
    # priors_list = [[1/8] * 8]
    # priors_list = [[1/4] * 4]
    for priors in priors_list:
        povm = Povm()
        povm.pretty_good_measurement(quantum_states, priors, debug=True)


# semidefinite programming (min error) for |0> and |1>
def test9():
    seed = 1
    repeat = 100_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_sensor=1, state_vector=vector1)
    qs2 = QuantumState(num_sensor=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
    priors_list = [[0.5, 0.5], [0.4, 0.6]]
    for priors in priors_list:
        povm = Povm()
        povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


# semidefinite programming (min error) of two random states, when is it optimal?
def test10():
    for seed in range(10):
        qs1 = QuantumState(num_sensor=1)
        qs2 = QuantumState(num_sensor=1)
        qs1.init_random_state(seed=seed)
        qs2.init_random_state(seed=seed+10)
        quantum_states = [qs1, qs2]
        priors_list = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.7, 0.3], [0.95, 0.05]]
        for priors in priors_list:
            povm = Povm()
            povm.semidefinite_programming_minerror(quantum_states, priors, debug=True)


# semidefinite programming (min error) of three random states, when is it optimal?
def test11():
    for seed in range(10):
        qs1 = QuantumState(num_sensor=1)
        qs2 = QuantumState(num_sensor=1)
        qs3 = QuantumState(num_sensor=1)
        qs1.init_random_state(seed=seed)
        qs2.init_random_state(seed=seed+10)
        qs3.init_random_state(seed=seed+20)
        quantum_states = [qs1, qs2, qs3]
        priors_list = [[0.1, 0.1, 0.8], [0.2, 0.3, 0.5], [1/3, 1/3, 1/3], [0.7, 0.2, 0.1], [0.9, 0.06, 0.04]]
        for priors in priors_list:
            povm = Povm()
            povm.semidefinite_programming_minerror(quantum_states, priors, debug=True)


# unitary theta = 90 degrees, pretty good and SDP same?
def test12():
    num_sensor = 3
    seed = 2
    theta = 20
    priors = [1./3, 1./3, 1./3]
    # priors = [0.15, 0.3, 0.55]
    povm = Povm()
    unitary_operator = Utility.generate_unitary_operator(theta, seed)
    opt_initstate = OptimizeInitialState(num_sensor)
    opt_initstate.guess(unitary_operator)
    print(opt_initstate)
    print('success', opt_initstate.evaluate(unitary_operator, priors, povm, 'min error'))
    init_state = QuantumState(num_sensor, opt_initstate.state_vector)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(unitary_operator, num_sensor, i)
        init_state_copy = copy.deepcopy(init_state)
        init_state_copy.evolve(evolve_operator)
        quantum_states.append(init_state_copy)
    # povm.pretty_good_measurement(quantum_states, priors, debug=False)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
    repeat = 1_000_000
    qm = QuantumMeasurement()
    qm.preparation(quantum_states, priors)
    qm.povm = povm
    error = qm.simulate(seed, repeat)
    qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


# semidefinite programming (Unambiguous) of |0> and |+>
def test13():
    seed = 1
    repeat = 100_000
    vector1 = np.array([1, 0])
    vector2 = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
    qs1 = QuantumState(num_sensor=1, state_vector=vector1)
    qs2 = QuantumState(num_sensor=1, state_vector=vector2)
    quantum_states = [qs1, qs2]
    priors_list = [[0.5, 0.5]]
    for priors in priors_list:
        povm = Povm()
        povm.semidefinite_programming_unambiguous(quantum_states, priors, debug=False)

        qm = QuantumMeasurement()
        qm.preparation(quantum_states, priors)
        qm.povm = povm
        error = qm.simulate(seed, repeat)
        qm.simulate_report(quantum_states, priors, povm, seed, repeat, error)


# Unambiguous discrimination of two random states (complex numbers introduced)
def test14():
    seed = 2
    repeat = 100_000
    qs1 = QuantumState(num_sensor=1)
    qs2 = QuantumState(num_sensor=1)
    qs1.init_random_state(seed=1)
    qs2.init_random_state(seed=2)
    quantum_states = [qs1, qs2]
    priors_list = [[0.4, 0.6]]
    for priors in priors_list:
        povm = Povm()
        povm.semidefinite_programming_unambiguous(quantum_states, priors, debug=False)

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
    # test5()
    # test6()
    # test7()
    test8()
    # test9()
    # test10()
    # test11()
    # test12()
    # test13()
    # test14()
