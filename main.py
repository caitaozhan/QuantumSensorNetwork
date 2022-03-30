'''The main
'''

import numpy as np
import random
import copy
from qiskit.quantum_info import random_unitary
from optimize_initial_state import OptimizeInitialState
from povm import Povm
from utility import Utility
import time

def main():
    print('Two sensors\n')
    seed = 2
    num_sensor = 2
    povm = Povm()
    unitary_operator = random_unitary(dims=2, seed=seed)
    Utility.print_matrix('Unitary matrix', unitary_operator._data)
    optis = OptimizeInitialState(num_sensor)

    # optimize the initial state: Guess
    optis.guess(unitary_operator)
    print(optis)
    # generate the quantums states for SDP, equal prior, then do the SDP
    # priors = [random.uniform(0.1, 0.9)]
    priors = [0.296]
    priors.append(1 - priors[0])
    print('priors:', priors)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(unitary_operator, num_sensor, i)
        optis_copy = copy.deepcopy(optis)
        optis_copy.evolve(evolve_operator)
        quantum_states.append(optis_copy)
    # Optimizing the POVM
    povm.semidefinite_programming(quantum_states, priors, debug=False)
    print(f'SDP error = {povm.theoretical_error}')
    povm.two_state_minerror(quantum_states, priors, debug=False)   # the measurement operators summation is not Identity... But the theoretical error is correct
    print(f'MED error = {povm.theoretical_error}')

    # do random guess for the initial state
    print('\nOptimization method is Random.\nRepeat 1000 times.')
    errors = []
    elapse = 0
    for i in range(1000):
        # random intial state
        optis.random(i, unitary_operator)
        quantum_states = []
        for i in range(num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, num_sensor, i)
            optis_copy = copy.deepcopy(optis)
            optis_copy.evolve(evolve_operator)
            quantum_states.append(optis_copy)
        # Optimizing the POVM
        start = time.time()
        povm.semidefinite_programming(quantum_states, priors, debug=False)
        elapse += (time.time() - start)
        errors.append(povm.theoretical_error)
    print(f'min error = {np.min(errors)}\nmax error = {np.max(errors)}\navg error = {np.average(errors)}')
    print(f'SDP time elpase = {elapse:.2}s')

if __name__ == '__main__':
    main()
