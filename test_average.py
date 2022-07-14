'''
average the coefficients
'''

import numpy as np
from qiskit.quantum_info.operators.operator import Operator
from quantum_state_custombasis import QuantumStateCustomBasis
from quantum_state import QuantumState
from povm import Povm
from utility import Utility
from copy import deepcopy
from equation_generator import EquationGenerator



def basis(v1: np.array, v2: np.array, string: str) -> np.array:
    '''generate the customized basis, e.g., |--+>
       the same as OptimizeInitState.eigenvector()
    Args:
        v1 -- u+
        v2 -- u-
        string -- the eigenvector to generate in binary string, e.g., '0001'
    Return:
        basis
    '''
    tensor = 1
    for i in string:
        if i == '1':
            tensor = np.kron(tensor, v1)
        else:
            tensor = np.kron(tensor, v2)
    return tensor


def generate_custombasis(num_sensor: int, U: Operator) -> list:
    '''generate a customized set of basis from unitary operator U
    Args:
        U -- unitary operator
    Return:
        a list, where each element is a np.array (a basis |j>)
    '''
    e_vals, e_vectors = np.linalg.eig(U.data)
    theta1 = Utility.get_theta(e_vals[0].real, e_vals[0].imag)
    theta2 = Utility.get_theta(e_vals[1].real, e_vals[1].imag)
    v1 = e_vectors[:, 0]  # v1 is positive
    v2 = e_vectors[:, 1]  # v2 is negative
    if theta1 < theta2:
        v1, v2 = v2, v1
    custombasis = []
    for i in range(2**num_sensor):
        j = bin(i)[2:]
        j = '0' * (num_sensor-len(j)) + j
        custombasis.append(basis(v1, v2, j))
    return custombasis


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


def average_init_state(init_state: QuantumStateCustomBasis, partition: list) -> QuantumState:
    '''averaging the same partition coefficients for an initial state
    Args:
        init_state -- initial quantum state
        partition -- eg. [1,2,4]
    Return:
        a quantum state after averaging
    '''
    state_vector_custom_copy = np.array(init_state.state_vector_custom)
    coeff = init_state.state_vector_custom[partition]
    probs = [np.abs(co)**2 for co in coeff]
    abs_avg = np.sqrt(np.average(probs))
    for i in partition:
        ratio = abs_avg / abs(state_vector_custom_copy[i]) 
        state_vector_custom_copy[i] *= ratio
    qstate = QuantumStateCustomBasis(init_state.num_sensor, init_state.custom_basis, state_vector_custom_copy)
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
    

def main2(debug, seed, unitary_theta):
    '''test the averaging the coefficients in each partition will lead to a better initial state
    '''
    print(f'unitary theta is {unitary_theta}, seed is {seed}', end=' ')
    num_sensor = 3
    priors = [1/3, 1/3, 1/3]
    povm = Povm()
    varying_partition = 1
    # 1. random initial state and random unitary operator
    U = Utility.generate_unitary_operator(theta=unitary_theta, seed=seed)
    if debug:
        Utility.print_matrix('\nUnitary operator:', U.data)
    custom_basis = generate_custombasis(num_sensor, U)
    init_state_custom = QuantumStateCustomBasis(num_sensor, custom_basis)
    eg = EquationGenerator(num_sensor)
    partitions = []
    for i in range(num_sensor+1):
        partition = eg.get_partition(i)
        partitions.append([int(bin_string, 2) for bin_string in partition])
    init_state_custom.init_random_state_realnumber_partition(seed, partitions, varying=varying_partition)
    if debug:
        print(f'Initial state:\n{init_state_custom}')
        print()
    # 2. the initial state evolves to different quantum states to be discriminated and do SDP
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_custom)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
    if debug:
        print(f'the probability of error is {povm.theoretical_error:.5f}')
    non_averaged = povm.theoretical_error
    # 3. average the coefficients of same partitions
    init_state_avg = deepcopy(init_state_custom)
    partition = eg.get_partition(varying_partition)
    partition = [int(bin_string, 2) for bin_string in partition]
    init_state_avg = average_init_state(init_state_avg, partition)
    # 4. do SDP for the new averaged initial state
    if debug:
        print('\nthe averaged initial state:')
        print(init_state_avg)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_avg)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
    if debug:
        print(f'the probability of error is {povm.theoretical_error:.5f} for the averaged initial state')
    averaged = povm.theoretical_error
    if non_averaged > averaged:
        print(True)
    else:
        print(False)


if __name__ == '__main__':
    debug = False
    # seed = 1
    # main1(debug, seed)
    # for unitary_theta in range(1, 90):
    #     for seed in range(10):
    #         main2(debug, seed, unitary_theta)  # all is True

    debug = True
    main2(debug, seed=2, unitary_theta=40)



'''

unitary theta is 40, seed is 2 
Unitary operator:
( 0.76604+0.54268i) ( 0.24684-0.24029i) 
(-0.24684-0.24029i) ( 0.76604-0.54268i) 
Initial state:

Custom Basis:
[ 0.886+0.000i -0.179+0.184i -0.179+0.184i -0.002-0.075i -0.179+0.184i -0.002-0.075i -0.002-0.075i  0.016+0.015i ]
[ 0.179+0.184i  0.886+0.000i -0.075+0.000i -0.179+0.184i -0.075+0.000i -0.179+0.184i  0.015-0.016i -0.002-0.075i ]
[ 0.179+0.184i -0.075+0.000i  0.886+0.000i -0.179+0.184i -0.075+0.000i  0.015-0.016i -0.179+0.184i -0.002-0.075i ]
[-0.002+0.075i  0.179+0.184i  0.179+0.184i  0.886+0.000i -0.015-0.016i -0.075+0.000i -0.075+0.000i -0.179+0.184i ]
[ 0.179+0.184i -0.075+0.000i -0.075+0.000i  0.015-0.016i  0.886+0.000i -0.179+0.184i -0.179+0.184i -0.002-0.075i ]
[-0.002+0.075i  0.179+0.184i -0.015-0.016i -0.075+0.000i  0.179+0.184i  0.886+0.000i -0.075+0.000i -0.179+0.184i ]
[-0.002+0.075i -0.015-0.016i  0.179+0.184i -0.075+0.000i  0.179+0.184i -0.075+0.000i  0.886+0.000i -0.179+0.184i ]
[-0.016+0.015i -0.002+0.075i -0.002+0.075i  0.179+0.184i -0.002+0.075i  0.179+0.184i  0.179+0.184i  0.886+0.000i ]

Coefficients in custom basis:
|000>: 0.342379
|001>: 0.020359
|010>: 0.431641
|011>: 0.341851
|100>: 0.330108
|101>: 0.341851
|110>: 0.341851
|111>: 0.486303

Coefficients in computational basis:
|000>:  0.433724  0.228029i
|001>:  0.016172  0.220220i
|010>:  0.411116  0.220220i
|011>:  0.262058  0.142276i
|100>:  0.313616  0.220220i
|101>:  0.281821  0.121975i
|110>:  0.201769  0.204212i
|111>:  0.250445  0.135679i


the probability of error is 0.20131

the averaged initial state:

Custom Basis:
[ 0.886+0.000i -0.179+0.184i -0.179+0.184i -0.002-0.075i -0.179+0.184i -0.002-0.075i -0.002-0.075i  0.016+0.015i ]
[ 0.179+0.184i  0.886+0.000i -0.075+0.000i -0.179+0.184i -0.075+0.000i -0.179+0.184i  0.015-0.016i -0.002-0.075i ]
[ 0.179+0.184i -0.075+0.000i  0.886+0.000i -0.179+0.184i -0.075+0.000i  0.015-0.016i -0.179+0.184i -0.002-0.075i ]
[-0.002+0.075i  0.179+0.184i  0.179+0.184i  0.886+0.000i -0.015-0.016i -0.075+0.000i -0.075+0.000i -0.179+0.184i ]
[ 0.179+0.184i -0.075+0.000i -0.075+0.000i  0.015-0.016i  0.886+0.000i -0.179+0.184i -0.179+0.184i -0.002-0.075i ]
[-0.002+0.075i  0.179+0.184i -0.015-0.016i -0.075+0.000i  0.179+0.184i  0.886+0.000i -0.075+0.000i -0.179+0.184i ]
[-0.002+0.075i -0.015-0.016i  0.179+0.184i -0.075+0.000i  0.179+0.184i -0.075+0.000i  0.886+0.000i -0.179+0.184i ]
[-0.016+0.015i -0.002+0.075i -0.002+0.075i  0.179+0.184i -0.002+0.075i  0.179+0.184i  0.179+0.184i  0.886+0.000i ]

Coefficients in custom basis:
|000>: 0.342379
|001>: 0.313953
|010>: 0.313953
|011>: 0.341851
|100>: 0.313953
|101>: 0.341851
|110>: 0.341851
|111>: 0.486303

Coefficients in computational basis:
|000>:  0.462397  0.257484i
|001>:  0.286158  0.220220i
|010>:  0.286158  0.220220i
|011>:  0.230241  0.174962i
|100>:  0.286158  0.220220i
|101>:  0.230241  0.174962i
|110>:  0.230241  0.174962i
|111>:  0.250124  0.123739i

the probability of error is 0.18641 for the averaged initial state

'''