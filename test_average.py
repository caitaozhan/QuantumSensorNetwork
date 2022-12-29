'''
average the coefficients
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from typing import List
from qiskit.quantum_info.operators.operator import Operator
from quantum_measurement import QuantumMeasurement
from quantum_state_custombasis import QuantumStateCustomBasis
from quantum_state import QuantumState
from povm import Povm
from utility import Utility
from copy import deepcopy
from equation_generator import EquationGenerator


plt.rcParams['font.size'] = 40
plt.rcParams['lines.linewidth'] = 4


def plot_errors(errors, i, j):
    X = [abs(c1 - c2) for c1, c2, _ in errors]
    y = [error for _, _, error in errors]
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.1, top=0.96)
    ax.set_xlabel(f'|C{i} - C{j}|')
    ax.set_ylabel('Error')
    ax.plot(X, y)
    fig.savefig(f'tmp-|C{i} - C{j}|')


def plot_errors_steps(errors: list, theta: int, seed: int):
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.1, top=0.9)
    X = [i for i in range(len(errors))]
    ax.plot(X, errors)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Error')
    ax.set_title(f'Theta={theta}, Seed={seed}')
    fig.savefig(f'tmp/babystep-theta{theta}-seed{seed}.png')


class Step:
    '''different coefficients will have a different step towards the partition average
    '''
    def __init__(self, total_step: int, init_state_custom: QuantumStateCustomBasis, eg: EquationGenerator):
        self.total_step = total_step
        self.init_state_custom = init_state_custom
        self.eg = eg
        self.stepsize_squared = None
        self.init_stepsize_squared()

    def init_stepsize_squared(self):
        num_sensor = self.init_state_custom.num_sensor
        self.stepsize_squared = [0] * 2**num_sensor
        for p in range(num_sensor+1):
            partition = self.eg.get_partition(p)
            partition = [int(bin_string, 2) for bin_string in partition]
            coeffs = self.init_state_custom.state_vector_custom[partition]
            coeff_squared_avg = np.average([coeff**2 for coeff in coeffs])
            for i in partition:
                self.stepsize_squared[i] = (coeff_squared_avg - self.init_state_custom.state_vector_custom[i]**2) / self.total_step

    def size_squared(self, i):
        return self.stepsize_squared[i]


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


def permutation(init_state: QuantumState) -> QuantumState:
    '''shift one bit to the right: 001 --> 100, 010 --> 001, 100 --> 010
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


def permutation_custombasis(init_state_custom: QuantumStateCustomBasis) -> List[QuantumStateCustomBasis]:
    '''generate all the permutations for a QuantumStateCustomBasis
       for num_sensor = 3, there will be 5 permutations (3! - 1)
    '''
    n = init_state_custom.num_sensor
    index = [i for i in range(n)]
    permutes = list(permutations(index))
    permutes.pop(0)
    init_state_custom_permutations = []
    for permute in permutes:
        mapping = [0]*2**n
        for i in range(2**n):
            bin_str = bin(i)[2:]
            bin_str = '0' * (n-len(bin_str)) + bin_str
            bin_str_permute = [''] * n
            for j in range(n):
                bin_str_permute[permute[j]] = bin_str[j]
            bin_str_permute = ''.join(bin_str_permute)
            mapping[i] = int(bin_str_permute, 2)
        state_vector_custom = np.array(init_state_custom.state_vector_custom)
        for i in range(2**n):
            state_vector_custom[mapping[i]] = init_state_custom.state_vector_custom[i]
        qstate = QuantumStateCustomBasis(n, init_state_custom.custom_basis, state_vector_custom)
        qstate.custom2computational()
        init_state_custom_permutations.append(qstate)
    return init_state_custom_permutations        


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



def evaluate(init_state_custom: QuantumStateCustomBasis, U: Operator, priors: list, povm: Povm, debug: bool = False) -> float:
    '''evaluate an initial state using SDP, return the probability of error
    '''
    quantum_states = []
    num_sensor = init_state_custom.num_sensor
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_custom)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug=debug)
    return povm.theoretical_error


def evaluate_povm(init_state_custom: QuantumStateCustomBasis, U: Operator, priors: list, povm: Povm, debug: bool = False) -> Povm:
    '''evaluate an initial state using SDP, return the Povm object
    '''
    quantum_states = []
    num_sensor = init_state_custom.num_sensor
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_custom)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    povm.semidefinite_programming_minerror(quantum_states, priors, debug=debug)
    return povm


def evaluate_simulation(init_state_custom: QuantumStateCustomBasis, U: Operator, priors: list, povm: Povm, seed: int, repeat: int ,debug: bool = False) -> float:
    '''evaluate an initial state using simulation, return the probability of error
    '''
    quantum_states = []
    num_sensor = init_state_custom.num_sensor
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(U, num_sensor, i)
        qstate = deepcopy(init_state_custom)
        qstate.evolve(evolve_operator)
        quantum_states.append(qstate)
    qm = QuantumMeasurement()
    qm.preparation(quantum_states, priors)
    qm.povm = povm
    error = qm.simulate(seed, repeat)
    return error


def modify(init_state_custom: QuantumStateCustomBasis, delta: float, i: float, j: float) -> bool:
    '''modify the coefficients at index i and j, make their squared values closer together by plus/misus delta
       return whether the coefficients at i and j are equaled
    '''
    equaled = False
    c1sq = init_state_custom.state_vector_custom[i] ** 2
    c2sq = init_state_custom.state_vector_custom[j] ** 2
    if c1sq < c2sq:
        c1sq += delta
        c2sq -= delta
        if c1sq > c2sq:
            equaled = True
            c1sq = (c1sq + c2sq) / 2
            c2sq = c1sq
        init_state_custom.state_vector_custom[i] = np.sqrt(c1sq)
        init_state_custom.state_vector_custom[j] = np.sqrt(c2sq)
    if c1sq > c2sq:
        c1sq -= delta
        c2sq += delta
        if c1sq < c2sq:
            equaled = True
            c1sq = (c1sq + c2sq) / 2
            c2sq = c1sq
        init_state_custom.state_vector_custom[i] = np.sqrt(c1sq)
        init_state_custom.state_vector_custom[j] = np.sqrt(c2sq)
    init_state_custom.custom2computational()
    # print(init_state_custom.check_state())
    return equaled


def average_two_states(qstate_custom1: QuantumStateCustomBasis, qstate_custom2: QuantumStateCustomBasis) -> QuantumStateCustomBasis:
    n = qstate_custom1.num_sensor
    state_vector_custom = [0]*2**n
    for i in range(2**n):
        coeff1 = qstate_custom1.state_vector_custom[i]
        coeff2 = qstate_custom2.state_vector_custom[i]
        state_vector_custom[i] = np.sqrt((coeff1**2 + coeff2**2) / 2)
    qstate = QuantumStateCustomBasis(n, qstate_custom1.custom_basis, np.array(state_vector_custom))
    qstate.custom2computational()
    # print(qstate.check_state())
    return qstate


# doing average in a single partition
def main2(debug, seed, unitary_theta):
    '''test the averaging the coefficients in each partition will lead to a better initial state
       here only one partition has varying coefficients
    '''
    print(f'unitary theta is {unitary_theta}, seed is {seed}', end=' ')
    num_sensor = 3
    priors = [1/3, 1/3, 1/3]
    povm = Povm()
    varying_partition = 2
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
        print(True, non_averaged - averaged )
    else:
        print(False)


# doing average (small step delta) in a single partition
def main2_delta(debug, seed, unitary_theta):
    '''instead of averaging in main2, here change small delta at a time
       here only one partition has varying coefficients
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
    errors = [] # (c1, c2, error)
    i, j = 2, 4
    equaled = False
    delta = 0.001   # difference in the coefficient squared
    while not equaled:
        error = evaluate(init_state_custom, U, priors, povm)
        c1sq = init_state_custom.state_vector_custom[i] ** 2
        c2sq = init_state_custom.state_vector_custom[j] ** 2
        errors.append((c1sq, c2sq, error))
        equaled = modify(init_state_custom, delta, i, j)
    
    for c1sq, c2sq, error in errors:
        print(c1sq, c2sq, abs(c1sq - c2sq), error)
    plot_errors(errors, i, j)

counter_false = 0
counter_true = 0


def main3(debug, seed, unitary_theta):
    '''here all parititions has varying coefficients
       do averaging all partitions
    '''
    print(f'unitary theta is {unitary_theta}, seed is {seed}', end=' ')
    num_sensor = 3
    priors = [1/3, 1/3, 1/3]
    povm = Povm()
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
    init_state_custom.init_random_state_realnumber(seed)
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
        print(f'the probability of error is {povm.theoretical_error:.7f}')
    non_averaged = povm.theoretical_error
    # 3. average the coefficients of same partitions
    init_state_avg = deepcopy(init_state_custom)
    for partition_to_average in [1, 2]:
        partition = eg.get_partition(partition_to_average)
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
        print(f'the probability of error is {povm.theoretical_error:.7f} for the averaged initial state')
    averaged = povm.theoretical_error
    if non_averaged > averaged:
        print(True, non_averaged - averaged)
        global counter_true
        counter_true += 1
    else:
        print(False, non_averaged - averaged)
        global counter_false
        counter_false += 1


def main3_delta(debug, seed, unitary_theta):
    '''instead of averaging in main3, here change small delta at a time
       here all parititions has varying coefficients
    '''
    print(f'unitary theta is {unitary_theta}, seed is {seed}', end=' ')
    num_sensor = 3
    priors = [1/3, 1/3, 1/3]
    povm = Povm()
    # 1. random initial state and random unitary operator
    U = Utility.generate_unitary_operator(theta=unitary_theta, seed=2)
    if debug:
        Utility.print_matrix('\nUnitary operator:', U.data)
    custom_basis = generate_custombasis(num_sensor, U)
    init_state_custom = QuantumStateCustomBasis(num_sensor, custom_basis)
    init_state_custom.init_random_state_realnumber(seed=seed)      # all coefficients are random
    if debug:
        print(f'Initial state:\n{init_state_custom}')
        print()
    # 2. the initial state evolves to different quantum states to be discriminated and do SDP
    errors = []
    total_step = 100
    eg = EquationGenerator(num_sensor)
    step = Step(total_step, init_state_custom, eg)
    errors = []
    errors.append(evaluate(init_state_custom, U, priors, povm))
    for i in range(total_step):
        if debug:
            print(i, end=' ')
        for i in range(2**num_sensor):
            current_squared = init_state_custom.state_vector_custom[i] ** 2
            nxt_squared = current_squared + step.size_squared(i)
            init_state_custom.state_vector_custom[i] = np.sqrt(nxt_squared)
        init_state_custom.custom2computational()
        errors.append(evaluate(init_state_custom, U, priors, povm))
    if debug:
        print(errors[0], errors[-1])
    plot_errors_steps(errors, theta, seed)


def validate_lemma2():
    debug = False
    seed = 0
    unitary_theta = 40
    file_perm = 'result2/12.28.2022/lemma2.n{}.perm.npy'
    file_avg = 'result2/12.28.2022/lemma2.n{}.avg.npy'
    
    errors_perm, errors_avg = lemma2(2, debug, seed, unitary_theta)
    np.save(file_perm.format(2), np.array(errors_perm))
    np.save(file_avg.format(2), np.array(errors_avg))
    
    errors_perm, errors_avg = lemma2(3, debug, seed, unitary_theta)
    np.save(file_perm.format(3), np.array(errors_perm))
    np.save(file_avg.format(3), np.array(errors_avg))
    
    errors_perm, errors_avg = lemma2(4, debug, seed, unitary_theta)
    np.save(file_perm.format(4), np.array(errors_perm))
    np.save(file_avg.format(4), np.array(errors_avg))
    
    errors_perm, errors_avg = lemma2(5, debug, seed, unitary_theta)
    np.save(file_perm.format(5), np.array(errors_perm))
    np.save(file_avg.format(5), np.array(errors_avg))
    

# validate lemma 2 about averaging two states, n=3
def lemma2(num_sensor, debug, seed, unitary_theta):
    '''use QuantumStateCustomBasis
       1) confirm that the permutations of an initial state has the same probability of error
       2) average --> better state (lower error)
    '''
    print(f'---\nunitary theta is {unitary_theta}, seed is {seed}')
    priors = [1/num_sensor] * num_sensor
    povm = Povm()
    # 1. random initial state and random unitary operator
    U = Utility.generate_unitary_operator(theta=unitary_theta, seed=2)
    if debug:
        Utility.print_matrix('\nUnitary operator:', U.data)
    custom_basis = generate_custombasis(num_sensor, U)
    init_state_custom = QuantumStateCustomBasis(num_sensor, custom_basis)
    init_state_custom.init_random_state_realnumber(seed)      # all coefficients are random
    print('\ninitial state:')
    print('error', evaluate(init_state_custom, U, priors, povm, debug=debug))
    if debug:
        print(f'Initial state:\n{init_state_custom}\n')
    # 2. purmutate the initial state and evaluate them
    init_state_custom_permute = permutation_custombasis(init_state_custom)
    print('\npermutated states:')
    errors_perm = []
    for i, qstate in enumerate(init_state_custom_permute):
        errors_perm.append(evaluate(qstate, U, priors, povm, debug=debug))
        print(i, 'error', errors_perm[-1])
    # 3. average the initial state and the permutated state and evaluate them
    print('\naverage of initial state and the permutated states')
    errors_avg = []
    for i, qstate in enumerate(init_state_custom_permute):
        qstate_avg = average_two_states(init_state_custom, qstate)
        errors_avg.append(evaluate(qstate_avg, U, priors, povm, debug=debug))
        print(i, 'error', errors_avg[-1])
    return errors_perm, errors_avg




if __name__ == '__main__':
    debug = False
    seed = 4
    # for unitary_theta in range(1, 90):
    #     for seed in range(20):
    #         main3(debug, seed, unitary_theta)  # all is True
    # print('false', counter_false)
    # debug = True

    # main2(debug, seed=2, unitary_theta=40)
    # main2_delta(debug, seed=2, unitary_theta=40)

    seed = 2
    theta = 41

    # main3(debug, seed=seed, unitary_theta=theta)
    # print('\n*********\n')
    # for theta in range(40, 41):
    #     for seed in range(7, 8):
    #         print(f'theta={theta}, seed={seed}')
    #         main3_delta(debug, seed=seed, unitary_theta=theta)

    validate_lemma2()

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