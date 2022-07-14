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


def permutation(init_state: QuantumState):
    '''001 --> 100, 010 --> 001, 100 --> 010
    '''
    n = init_state.num_sensor
    mapping = [0]*2**n
    for i in range(2**n):
        bin_str = bin(i)[2:]
        bin_str = '0' * (n-len(bin_str)) + bin_str
        bin_str_permute = bin_str[-1] + bin_str[:-1]
        j = int(bin_str_permute, 2)
        mapping[i] = j
    state_vector = np.array(init_state.state_vector)
    for i in range(2**n):
        state_vector[mapping[i]] = init_state.state_vector[i]
    return QuantumState(n, state_vector)


def average_init_state(init_state: QuantumState, partition: list) -> QuantumState:
    '''averaging the same partition coefficients for an initial state
    Args:
        init_state -- initial quantum state
        partition -- eg. [1,2,4]
    Return:
        a quantum state after averaging
    '''
    state_vector_copy = np.array(init_state.state_vector)
    coeff = init_state.state_vector[partition]
    probs = [np.abs(co)**2 for co in coeff]
    abs_avg = np.sqrt(np.average(probs))
    for i in partition:
        ratio = abs_avg / abs(state_vector_copy[i]) 
        state_vector_copy[i] *= ratio
    qstate = QuantumState(init_state.num_sensor, state_vector_copy)
    # print(qstate.check_state())
    return qstate


def main1(debug, seed):
    '''confirm that the permutations of an initial state has the same probability of error
    '''
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
    # 3. do permutation
    init_state_permutation = permutation(init_state)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_permutation)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug)
    print(f'the probability of error is {povm.theoretical_error:.5f}')
    init_state_permutation = permutation(init_state_permutation)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_permutation)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug)
    print(f'the probability of error is {povm.theoretical_error:.5f}')
    

def main2(debug, seed):
    '''test the averaging the coefficients in each partition will lead to a better initial state
    '''
    print(f'seed is {seed}')
    unitary_theta = 50
    num_sensor = 3
    priors = [1/3, 1/3, 1/3]
    povm = Povm()
    # 1. random initial state and random unitary operator
    init_state = QuantumState(num_sensor)
    eg = EquationGenerator(num_sensor)
    partitions = []
    for i in range(num_sensor+1):
        partition = eg.get_partition(i)
        partitions.append([int(bin_string, 2) for bin_string in partition])
    init_state.init_random_state_realnumber_partition(seed, partitions, varying=1)
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
    # 3. average the coefficients of same partitions
    init_state_avg = deepcopy(init_state)
    for i in range(num_sensor+1):
        partition = eg.get_partition(i)
        partition = [int(bin_string, 2) for bin_string in partition]
        init_state_avg = average_init_state(init_state_avg, partition)
    # 4. do SDP for the new averaged initial state
    print('\nthe averaged initial state:')
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
    seed = 3
    # main1(debug, seed)
    main2(debug, seed)

'''
seed is 2
Initial state:
|000>: 0.36932730689470566
|001>: 0.02196187462715846
|010>: 0.4656140740606537
|011>: 0.3687576302080315
|100>: 0.3560897329016312
|101>: 0.27982361540357986
|110>: 0.17335599223189854
|111>: 0.5245787900654866

Unitary operator:
( 0.64279+0.64675i) ( 0.29418-0.28636i) 
(-0.29418-0.28636i) ( 0.64279-0.64675i) 

the probability of error is 0.15527

the averaged initial state:
|000>: 0.36932730689470566
|001>: 0.3386633962006395
|010>: 0.3386633962006395
|011>: 0.2853861393602709
|100>: 0.3386633962006395
|101>: 0.2853861393602709
|110>: 0.2853861393602709
|111>: 0.5245787900654866

the probability of error is 0.15273 for the averaged initial state
'''

'''
seed is 3
Initial state:
|000>: 0.328587899151045
|001>: 0.42245768227297303
|010>: 0.1735441920996522
|011>: 0.30474293535918684
|100>: 0.532702761595381
|101>: 0.5346989553511666
|110>: 0.07492006257949849
|111>: 0.12363427969339942

Unitary operator:
( 0.64279+0.01022i) (-0.69867+0.31398i) 
( 0.69867+0.31398i) ( 0.64279-0.01022i) 

the probability of error is 0.14242

the averaged initial state:
|000>: 0.328587899151045
|001>: 0.40511739538653835
|010>: 0.4051173953865383
|011>: 0.3579498313459948
|100>: 0.40511739538653835
|101>: 0.3579498313459948
|110>: 0.3579498313459948
|111>: 0.12363427969339942

the probability of error is 0.17601 for the averaged initial state
'''