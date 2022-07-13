'''
average the coefficients
'''

import numpy as np
from itertools import permutations
from quantum_state import QuantumState
from povm import Povm
from utility import Utility
from copy import deepcopy
from equation_generator import EquationGenerator


def permute_init_state(init_state: QuantumState, permutation: list) -> list:
    '''permute an initial state
    Args:
        init_state -- initial quantum state
        permutation -- eg. [[1,2,4], [1,4,2], [2,1,4], [2,4,1], [4,1,2], [4,2,1]]
    Return:
        a list of initial states by permutation        
    '''
    num_sensor = init_state.num_sensor
    n = len(permutation)
    init_states = []
    index = sorted(permutation[0])
    for k in range(n):
        state_vector_copy = np.array(init_state.state_vector)
        for i, j in zip(index, permutation[k]):
            state_vector_copy[i] = init_state.state_vector[j]
        init_state_new = QuantumState(num_sensor, state_vector_copy)
        init_states.append((init_state_new, permutation[k]))
    return init_states

def average_init_state(init_state: QuantumState, permutation: list) -> QuantumState:
    '''averaging the same partition coefficients for an initial state
    Args:
        init_state -- initial quantum state
        permutation -- eg. [[1,2,4], [1,4,2], [2,1,4], [2,4,1], [4,1,2], [4,2,1]]
    Return:
        a quantum state after averaging
    '''
    index = sorted(permutation[0])
    state_vector_copy = np.array(init_state.state_vector)
    coeff = init_state.state_vector[index]
    probs = [np.abs(co)**2 for co in coeff]
    abs_avg = np.sqrt(np.average(probs))
    for i in index:
        ratio = abs_avg / abs(state_vector_copy[i]) 
        state_vector_copy[i] *= ratio
    qstate = QuantumState(init_state.num_sensor, state_vector_copy)
    # print(qstate.check_state())
    return qstate


def main(debug, seed):
    print(f'seed is {seed}')
    unitary_theta = 44
    num_sensor = 3
    priors = [1/3, 1/3, 1/3]
    povm = Povm()
    # 1. random initial state and random unitary operator
    init_state = QuantumState(num_sensor)
    init_state.init_random_state(seed)
    U = Utility.generate_unitary_operator(theta=unitary_theta, seed=seed)
    print(f'Initial state:\n{init_state}')
    Utility.print_matrix('Unitary operator:', U.data)
    print()
    # 2. the initial state evolves to different quantum states to be discriminated and do SDP
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug)
    print(f'the probability of error is {povm.theoretical_error:.5f}')
    # 3. do permutations
    eg = EquationGenerator(num_sensor)
    init_state_permutation = []
    for i in range(num_sensor+1):
        partition = eg.get_partition(i)
        partition = [int(bin_string, 2) for bin_string in partition]
        permutation = list(permutations(partition))
        init_state_permutation.extend(permute_init_state(init_state, permutation))
    # 4. do SDP for each new permutated initial state
    for init_state, permutation in init_state_permutation:
        quantum_states = []
        for i in range(num_sensor):
            evolve_operator = Utility.evolve_operator(U, num_sensor, i)
            qstate = deepcopy(init_state)
            qstate.evolve(evolve_operator)
            quantum_states.append(qstate)
        povm.semidefinite_programming_minerror(quantum_states, priors, debug)
        print(f'the probability of error is {povm.theoretical_error:.5f} for permutation {permutation}')
    print()
    # 5. average the same partitions
    init_state_avg = deepcopy(init_state)
    for i in range(num_sensor+1):
        partition = eg.get_partition(i)
        partition = [int(bin_string, 2) for bin_string in partition]
        permutation = list(permutations(partition))
        init_state_avg = average_init_state(init_state_avg, permutation)
    # 6. do SDP for the new averaged initial state
    print('the averaged initial state:')
    print(init_state_avg)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_avg)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug)
    print(f'the probability of error is {povm.theoretical_error:.5f} for the averaged initial state')


if __name__ == '__main__':
    debug = False
    seed = 2           # when seed is 1 or 2, the averaged initial state gets lower error.
    main(debug, seed)  # but when seed is 3, the averaged initial state gets higher error


'''
seed is 2
Initial state:
|000>: -0.071703 -0.531088i
|001>:  0.055635 -0.072456i
|010>: -0.089209 -0.190070i
|011>: -0.330871  0.133615i
|100>: -0.224440 -0.261215i
|101>:  0.135702  0.032647i
|110>: -0.409367  0.015211i
|111>: -0.353511  0.319651i

Unitary operator:
( 0.71934+0.58648i) ( 0.26676-0.25968i) 
(-0.26676-0.25968i) ( 0.71934-0.58648i) 

the probability of error is 0.25056
the probability of error is 0.25056 for permutation (0,)
the probability of error is 0.25056 for permutation (1, 2, 4)
the probability of error is 0.23713 for permutation (1, 4, 2)
the probability of error is 0.24754 for permutation (2, 1, 4)
the probability of error is 0.23626 for permutation (2, 4, 1)
the probability of error is 0.25006 for permutation (4, 1, 2)
the probability of error is 0.25202 for permutation (4, 2, 1)
the probability of error is 0.25056 for permutation (3, 5, 6)
the probability of error is 0.24754 for permutation (3, 6, 5)
the probability of error is 0.23713 for permutation (5, 3, 6)
the probability of error is 0.23626 for permutation (5, 6, 3)
the probability of error is 0.25006 for permutation (6, 3, 5)
the probability of error is 0.25202 for permutation (6, 5, 3)
the probability of error is 0.25056 for permutation (7,)

the averaged initial state:
|000>: -0.071703 -0.531088i
|001>:  0.145417 -0.189383i
|010>: -0.101449 -0.216148i
|011>: -0.300282  0.121262i
|100>: -0.155607 -0.181104i
|101>:  0.314859  0.075748i
|110>: -0.323619  0.012025i
|111>: -0.353511  0.319651i

the probability of error is 0.19720 for the averaged initial state
'''


'''
seed is 3
Initial state:
|000>:  0.047057  0.192821i
|001>: -0.193699  0.010030i
|010>:  0.364013  0.367113i
|011>: -0.346845 -0.271200i
|100>: -0.415506 -0.054832i
|101>: -0.435507 -0.039988i
|110>:  0.138162 -0.205202i
|111>:  0.163277  0.084172i

Unitary operator:
( 0.71934+0.00927i) (-0.63356+0.28472i) 
( 0.63356+0.28472i) ( 0.71934-0.00927i) 

the probability of error is 0.12949
the probability of error is 0.12949 for permutation (0,)
the probability of error is 0.12949 for permutation (1, 2, 4)
the probability of error is 0.14862 for permutation (1, 4, 2)
the probability of error is 0.20373 for permutation (2, 1, 4)
the probability of error is 0.20409 for permutation (2, 4, 1)
the probability of error is 0.19440 for permutation (4, 1, 2)
the probability of error is 0.17742 for permutation (4, 2, 1)
the probability of error is 0.12949 for permutation (3, 5, 6)
the probability of error is 0.20373 for permutation (3, 6, 5)
the probability of error is 0.14862 for permutation (5, 3, 6)
the probability of error is 0.20409 for permutation (5, 6, 3)
the probability of error is 0.19440 for permutation (6, 3, 5)
the probability of error is 0.17742 for permutation (6, 5, 3)
the probability of error is 0.12949 for permutation (7,)

the averaged initial state:
|000>:  0.047057  0.192821i
|001>: -0.399693  0.020697i
|010>:  0.281802  0.284202i
|011>: -0.303851 -0.237583i
|100>: -0.396788 -0.052362i
|101>: -0.384093 -0.035267i
|110>:  0.215419 -0.319946i
|111>:  0.163277  0.084172i

the probability of error is 0.16932 for the averaged initial state
'''
