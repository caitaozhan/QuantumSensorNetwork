'''
Quantum measurment, POVM
'''

import random
import numpy as np
from itertools import accumulate
from bisect import bisect_left
from utility import Utility
from qiskit.quantum_info.operators.operator import Operator
from povm import Povm

class QuantumMeasurement:
    '''Encapsulate the quantum measurement process
    '''
    def __init__(self):
        self._quantum_states = None
        self._priors = None
        self._priors_prefix = None
        self._povm = None

    @property
    def quantum_states(self):
        return self._quantum_states
    
    @property
    def prior(self):
        return self._priors

    @property
    def prior_prefix(self):
        return self._priors_prefix

    @property
    def povm(self):
        return self._povm

    @povm.setter
    def povm(self, povm: Povm):
        self._povm = povm

    def preparation(self, quantum_states: list, prior: list):
        '''set up the quantum states to be discriminated and their prior probability
        Args:
            quantum_states: a list of QuantumState
            prior: a list of probabilities
        '''
        if len(quantum_states) != len(prior):
            raise Exception('Number of quantum states and prior probability not equal')
        if abs(sum(prior) - 1) > Utility.EPSILON:
            raise Exception('Prior probability summation not one.')

        self._quantum_states = quantum_states
        self._priors = prior
        self._priors_prefix = list(accumulate(self._priors))

    def _sample(self, prefix):
        '''sample from a prefix sum array (the total summation is one)
        Return:
            int: the index of the randomly picked quantum state
        '''
        pick = random.random()
        return bisect_left(prefix, pick)

    def simulate(self, seed: int = 0, repeat: int = 10_000):
        '''repeat the single-shot measurement many times
        Return:
            float: the error probability
        '''
        random.seed(seed)
        index = 0
        error_count = 0
        while index < repeat:
            # step 1: alice sample a quantum state during preparation, and send to bob
            pick = self._sample(self.prior_prefix)
            prepared_quantum_state = self.quantum_states[pick]

            # step 2: bob receives the quantum state and does the measurement
            probs = []
            for M in self.povm.operators:
                density_operator = Operator(prepared_quantum_state.density_matrix)
                tmp = M.dot(density_operator)
                prob = np.trace(tmp.data)
                probs.append(prob)
            
            # step 3: collect the error stats
            probs_prefix = list(accumulate(probs))
            measure = self._sample(probs_prefix)
            if pick != measure:
                error_count += 1

            index += 1
        
        return 1.*error_count / repeat

    def simulate_report(self, quantum_states, priors, povm, seed, repeat, error):
        '''Generate the report for the simulation
        '''
        metric = 'error' if povm.method == 'Minimum Error' else  'failure'
        print('\n-------SIMULATION REPORT------')
        print('Quantum states to be discriminated:')
        for state in quantum_states:
            print(state)
        print(f'The priors are:\n{priors}\n')
        print(f'The POVM ({povm.method}) is: \n{povm}', end='')
        print(f'The theoretical {metric} = {povm.theoretical_error}')
        print(f'The simulated {metric}   = {error}  (seed = {seed}, repeat = {repeat})')
        print('-------END OF REPORT------')
