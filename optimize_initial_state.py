'''
Optimizing initial state
'''

import copy
import numpy as np
import math
from qiskit_textbook.tools import random_state
from qiskit.quantum_info.operators.operator import Operator
from quantum_state import QuantumState
from utility import Utility
from povm import Povm
from input_output import Default


class OptimizeInitialState(QuantumState):
    '''Optimize the initial state
    '''
    def __init__(self, num_sensor: int):
        super().__init__(num_sensor=num_sensor, state_vector=None)
        self._optimze_method = ''

    @property
    def optimize_method(self):
        return self._optimze_method

    def __str__(self):
        s = f'\n{self.optimize_method}\nInitial state:\n'
        parent = super().__str__()
        return s + parent

    def check_state(self):
        '''check if the amplitudes norm_squared add up to one
        '''
        summ = 0
        for amp in self._state_vector:
            summ += Utility.norm_squared(amp)
        return True if abs(summ - 1) < Default.EPSILON else False

    def normalize_state(self, state: np.array):
        '''Normalize a state vector
        Return:
            np.array -- the normalized state
        '''
        state_copy = np.array(state)
        magnitude_squared = 0
        for a in state_copy:
            magnitude_squared += abs(a)**2
        return state_copy / np.sqrt(magnitude_squared)

    def random(self, seed, unitary_operator: Operator):
        '''ignore the unitary operator and randomly initialize a quantum state
        '''
        np.random.seed(seed)
        self._state_vector = random_state(nqubits=self.num_sensor)
        self._method = 'Random'

    def guess(self, unitary_operator: Operator):
        '''do an eigenvalue decomposition, the two eigen vectors are |v> and |u>,
           then the guessed initial state is 1/sqrt(2) * (|v>|u> + |u>|v>) 
        Args:
            unitary_opeartor: describe the interaction with the environment
        '''
        e_vals, e_vectors = np.linalg.eig(unitary_operator._data)
        theta1 = Utility.get_theta(e_vals[0].real, e_vals[0].imag)
        theta2 = Utility.get_theta(e_vals[1].real, e_vals[1].imag)
        v1 = e_vectors[:, 0]    # v1 is positive
        v2 = e_vectors[:, 1]    # v2 is negative
        if theta1 < theta2:
            v1, v2, = v2, v1

        tensors = []
        for i in range(self.num_sensor):
            j = 0
            tensor = 1
            while j < self.num_sensor:
                if j == i:
                    tensor = np.kron(tensor, v1)  # kron is tensor product
                else:
                    tensor = np.kron(tensor, v2)
                j += 1
            tensors.append(tensor)
        self._state_vector = 1/math.sqrt(self.num_sensor) * np.sum(tensors, axis=0)
        if self.check_state() is False:
            raise Exception(f'{self} is not a valid quantum state')
        self._optimze_method = 'Guess'

    def evaluate(self, unitary_operator: Operator, priors: list, povm: Povm, eval_metric: str):
        '''evaluate the self.state_vector
        '''
        qstate = QuantumState(self.num_sensor, self.state_vector)
        return self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)

    def _evaluate(self, init_state: QuantumState, unitary_operator: Operator, priors: list, povm: Povm, eval_metric: str):
        '''evaluate the initial state
        Args:
            init_state -- initial state
            unitary_operator -- unitary operator
            priors -- prior probabilities
            povm   -- positive operator valued measurement
            eval_metric -- 'min error' or 'unambiguous'
        Return:
            float -- evaluate score by SDP solver
        '''
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)
        if eval_metric == 'min error':
            povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
        elif eval_metric == 'unambiguous':
            povm.semidefinite_programming_unambiguous(quantum_states, priors, debug=False)
        else:
            raise Exception(f'unknown eval_metric: {eval_metric}!')
        return povm.therotical_success

    def find_neighbors(self, init_state: QuantumState, i: int, mod_step: list, amp_step: list):
        '''find four neighbors of the initial state
        Args:
            init_state -- initial state
            i        -- ith element of the state vector to be modified
            mod_step -- step size for modulus
            amp_step -- step size for amplitude
        Return:
            list -- a list of QuantumState object
        '''
        init_state_vector = init_state.state_vector.copy()
        z = init_state_vector[i]
        r = abs(z)
        theta = Utility.get_theta(z.real, z.imag)
        array = []
        init_state_vector[i] = (r + mod_step)*np.exp(complex(0, theta))
        array.append(QuantumState(self.num_sensor, self.normalize_state(init_state_vector)))
        init_state_vector[i] = (r - mod_step)*np.exp(complex(0, theta))
        array.append(QuantumState(self.num_sensor, self.normalize_state(init_state_vector)))
        init_state_vector[i] = r*np.exp(complex(0, theta + amp_step))
        array.append(QuantumState(self.num_sensor, np.array(init_state_vector)))
        init_state_vector[i] = r*np.exp(complex(0, theta - amp_step))
        array.append(QuantumState(self.num_sensor, np.array(init_state_vector)))
        return array

    def hill_climbing(self, startState: QuantumState, seed: int, unitary_operator: Operator, priors: list, epsilon: float, \
                            mod_step: list, amp_step: list, decrease_rate: float, min_iteration: int, eval_metric: str):
        '''use the good old hill climbing method to optimize the initial state
        Args:
            startState       -- the start point of hill climbing
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors   -- prior probabilities
            EPSILON  -- for hill climbing termination
            mod_step -- step size for modulus
            amp_step -- step size for amplitude
            decrease_rate -- decrease rate
            min_iteration -- minimal number of iteration
            eval_metric -- 'min error' or 'unambiguous'
        Return:
            dict -- a dict of hill climbing summary
        '''
        print('\nStart hill climbing...')
        qstate = None
        if startState is None:
            np.random.seed(seed)
            qstate = QuantumState(self.num_sensor, random_state(nqubits=self.num_sensor))
            # print(f'Random start:\n{qstate}')
        else:
            qstate = startState
            # print(f'Start from guess:\n{startState}')
        N = 2**self.num_sensor
        povm = Povm()
        best_score = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        scores = [round(best_score, 6)]
        terminate = False
        iteration = 0
        while terminate is False or iteration < min_iteration:
            iteration += 1
            before_score = best_score
            for i in range(N):
                neighbors = self.find_neighbors(qstate, i, mod_step[i], amp_step[i])
                best_step = -1
                for j in range(len(neighbors)):
                    score = self._evaluate(neighbors[j], unitary_operator, priors, povm, eval_metric)
                    if score > best_score:
                        best_score = score
                        best_step = j
                if best_step == -1:
                    mod_step[i] *= decrease_rate
                    amp_step[i] *= decrease_rate
                elif best_step in [0, 1]:
                    qstate = neighbors[best_step]
                    mod_step[i] *= decrease_rate
                else: # best_step in [2, 3]:
                    qstate = neighbors[best_step]
                    amp_step[i] *= decrease_rate
            scores.append(round(best_score, 6))
            if best_score - before_score < epsilon:
                terminate = True
            else:
                terminate = False

        self._state_vector = qstate.state_vector
        self._optimze_method = 'Hill climbing'
        return scores