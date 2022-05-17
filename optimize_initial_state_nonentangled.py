'''
Optimizing initial state assuming that the different sensors are separable (non-entangled)
'''

import numpy as np
import copy
from quantum_state import QuantumState
from qiskit_textbook.tools import random_state
from qiskit.quantum_info.operators.operator import Operator
from povm import Povm


class OptimizeInitialStateNonentangled(QuantumState):
    '''Similar to the class OptimizeInitalState, but this class operates on non-entangled states
    '''
    def __init__(self, num_sensor: int):
        super().__init__(num_sensor=num_sensor, state_vector=None)
        self.sensor_states = [QuantumState(num_sensor=1, state_vector=None) for _ in range(num_sensor)]
        self._optimize_method = ''

    @property
    def optimize_method(self):
        return self._optimize_method

    def set_sensor_state(self, qstate: QuantumState):
        '''set each to individual sensor state to the same qstate
        '''
        for i in range(len(self.sensor_states)):
            self.sensor_states[i] = qstate

    def init_random_state(self, seed: int = None):
        '''
        Args:
            seed -- if given, then every (separate) sensor will have a same quantum state
        '''
        for sensor_state in self.sensor_states:
            sensor_state.init_random_state(seed)
        self.set_state_vector()
        if self.check_state() is False:
            raise Exception('Not a valid state vector')

    def set_state_vector(self):
        self.state_vector = 1
        for sensor_state in self.sensor_states:
            if sensor_state.state_vector is not None:
                self.state_vector = np.kron(self.state_vector, sensor_state.state_vector)
            else:
                raise Exception('sensor state is None')

    def __str__(self):
        s = ''
        for i, sensor_state in enumerate(self.sensor_states):
            s += f'Sensor-{i+1}\n{sensor_state}\n'

        parent = super().__str__()
        return s + 'Tensor product:\n' + parent

    def find_neighbors(self, qstate: QuantumState, i: int, step_size: float):
        '''find four random neighbor of qstate
        Args:
            qstate -- initial state
            i        -- ith element of the state vector to be modified
            step_size -- step size for modulus
        Return:
            list -- a list of QuantumState object
        '''
        array = []
        for _ in range(4):
            state_vector = qstate.state_vector.copy()
            direction = self.generate_random_direction()
            state_vector[i] += direction * step_size
            array.append(QuantumState(num_sensor=1, state_vector=self.normalize_state(state_vector)))
        return array

    def _evaluate(self, one_sensor_state: QuantumState, unitary_operator: Operator, priors: list, povm: Povm, eval_metric: str):
        '''evaluate the initial state where each individual sensor is the same
        Args:
            one_sensor_state -- initial state
            unitary_operator -- unitary operator
            priors -- prior probabilities
            povm   -- positive operator valued measurement
            eval_metric -- 'min error' or 'unambiguous'
        Return:
            float -- evaluate score by SDP solver and some probability calculating
        '''
        # do SDP over U (tensor) one_sensor_state and one_sensor_state
        quantum_states = [one_sensor_state]
        one_sensor_state_copy = copy.deepcopy(one_sensor_state)
        one_sensor_state_copy.evolve(unitary_operator)
        quantum_states.append(one_sensor_state_copy)
        if eval_metric == 'min error':
            povm.semidefinite_programming_minerror(quantum_states, priors=[0.5, 0.5], debug=False)
        elif eval_metric == 'unambiguous':
            povm.semidefinite_programming_unambiguous(quantum_states, priors=[0.5, 0.5], debug=False)

        # calculate probabilities...
        rho0 = one_sensor_state.density_matrix
        rho1 = one_sensor_state_copy.density_matrix
        tmp = povm.operators[0].dot(Operator(rho0))
        prob0 = np.trace(tmp.data).real
        tmp = povm.operators[1].dot(Operator(rho1))
        prob1 = np.trace(tmp.data).real
        num_sensor = len(priors)
        return prob0**(num_sensor - 1) * prob1

    def hill_climbing(self, seed: int, unitary_operator: Operator, priors: list, epsilon: float, \
                      step_size: float, decrease_rate: float, min_iteration: int, eval_metric: str):
        '''This is the hill climbing (random neighbor version) for the non-entangled sensor case
        Args:
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors   -- prior probabilities
            epsilon  -- for termination
            step_size -- step size for the random direction
            decrease_rate -- decrease rate
            min_iteration -- minimal number of iteration
            eval_metric -- 'min error' or 'unambiguous'
        Return:
            list -- a list of scores at each iteration
        '''
        print('\nStart hill climbing for the non-entangled case...')
        np.random.seed(seed)
        qstate = QuantumState(num_sensor=1, state_vector=random_state(nqubits=1))
        N = 2
        povm = Povm()
        best_score = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        scores = [round(best_score, 7)]
        terminate = False
        iteration = 0
        while terminate is False or iteration < min_iteration:
            iteration += 1
            before_score = best_score
            for i in range(N):
                neighbors = self.find_neighbors(qstate, i, step_size[i])
                best_step = -1
                for j in range(len(neighbors)):
                    score = self._evaluate(neighbors[j], unitary_operator, priors, povm, eval_metric)
                    if score > best_score:
                        best_score = score
                        best_step = j
                if best_step == -1:
                    step_size[i] *= decrease_rate
                else:
                    qstate = neighbors[best_step]
                    step_size[i] *= decrease_rate
            scores.append(round(best_score, 7))
            if best_score - before_score < epsilon:
                terminate = True
            else:
                terminate = False

        self.set_sensor_state(qstate)
        self.set_state_vector()
        self._optimize_method = 'Hill climbing (NE)'
        return scores




