'''
Optimizing initial state
'''

import time
import copy
import numpy as np
import math
from itertools import accumulate
import random
from bisect import bisect_left
from qiskit_textbook.tools import random_state
from qiskit.quantum_info.operators.operator import Operator
from input_output import Default
from quantum_state import QuantumState
from quantum_state_custombasis import QuantumStateCustomBasis
from utility import Utility
from povm import Povm
from equation_generator import EquationGenerator


class OptimizeInitialState(QuantumState):
    '''A quantum state that has optimization capabilities, i.e., optimizing the initial state.
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

    def upperbound(self, unitary_operator: Operator, priors: list):
        '''an upper bound from equation (10) of paper: https://arxiv.org/pdf/1509.04592.pdf
        '''
        init_state = QuantumState(self.num_sensor, self.state_vector)
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)
        summ = 0
        for i in range(self.num_sensor):
            for j in range(self.num_sensor):
                psi_i = quantum_states[i].state_vector
                psi_j = quantum_states[j].state_vector
                summ += 2 * np.sqrt(abs(((priors[i] + priors[j])/2)**2 - priors[i]*priors[j]*(abs(np.dot(np.conj(psi_i), psi_j)))**2))
        return 1./self.num_sensor + 1./(2*self.num_sensor)*summ
    
    def upperbound_new(self, unitary_operator: Operator, priors: list):
        '''a new upper bound from equation (53) of paper: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.105.032410
        '''
        init_state = QuantumState(self.num_sensor, self.state_vector)
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)
        summ = 0
        for i in range(self.num_sensor):
            for j in range(i + 1, self.num_sensor):
                rho_i = quantum_states[i].density_matrix
                rho_j = quantum_states[j].density_matrix                
                density_matrix = priors[i] * rho_i - priors[j] * rho_j
                summ += Utility.trace_norm(density_matrix)
        return 1/self.num_sensor * (1 + summ)

    def random(self, seed, unitary_operator: Operator):
        '''ignore the unitary operator and randomly initialize a quantum state
        '''
        np.random.seed(seed)
        self._state_vector = random_state(nqubits=self.num_sensor)
        self._method = 'Random'

    def theorem(self, unitary_operator: Operator, unitary_theta: float, partition_i: int):
        '''implementing the theorem (corollary + conjecture)
        '''
        e_vals, e_vectors = np.linalg.eig(unitary_operator._data)
        theta1 = Utility.get_theta(e_vals[0].real, e_vals[0].imag)
        theta2 = Utility.get_theta(e_vals[1].real, e_vals[1].imag)
        v1 = e_vectors[:, 0]    # v1 is positive
        v2 = e_vectors[:, 1]    # v2 is negative
        if theta1 < theta2:
            v1, v2, = v2, v1

        eg = EquationGenerator(self.num_sensor)
        RAD = 180 / np.pi
        T = 0.5 * np.arccos(-(1 - 1/math.ceil(self.num_sensor/2)))
        T *= RAD
        if T - Default.EPSILON <= unitary_theta <= 180 - T + Default.EPSILON:  # mutual orthogonal situation (corollary)
            a, b, c, partition = eg.optimal_solution_nomerge()
            coeff1 = np.sqrt(1 / (c - a*np.cos(2*unitary_theta/RAD) - b))                                   # for the symmetric partition, no merging
            coeff2squared = (-a*np.cos(2*unitary_theta/RAD) - b) / (c - a*np.cos(2*unitary_theta/RAD) - b)  # for partition 0, no merging
            coeff2squared = 0 if coeff2squared < 0 else coeff2squared
            coeff2 = np.sqrt(coeff2squared)
            states = []
            for ev in partition:
                e_vector = Utility.eigenvector(v1, v2, ev)
                states.append(coeff1 * e_vector)
            for ev in ['0'*self.num_sensor]:
                e_vector = Utility.eigenvector(v1, v2, ev)
                states.append(coeff2 * e_vector)
            self._state_vector = np.sum(states, axis=0)
        else:                                                                  # non mutual orthogonal situation (conjecture)
            # partition = eg.optimal_solution_smallerT_i(unitary_theta, partition_i)
            partition = eg.optimal_solution_smallerT()
            coeff = np.sqrt(1/len(partition))
            states = []
            for ev in partition:
                e_vector = Utility.eigenvector(v1, v2, ev)
                states.append(coeff * e_vector)
            self._state_vector = np.sum(states, axis=0)

        if self.check_state() is False:
            raise Exception(f'{self} is not a valid quantum state')
        self._optimze_method = 'Theorem'

    def evaluate(self, unitary_operator: Operator, priors: list, povm: Povm, eval_metric: str) -> float:
        '''evaluate the self.state_vector
        '''
        qstate = QuantumState(self.num_sensor, self.state_vector)
        return self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)

    def evaluate_orthogonal(self, unitary_operator: Operator):
        '''verify if all |psi> are mutually orthogonal
        '''
        init_state = QuantumState(self.num_sensor, self.state_vector)
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)

        for i in range(self.num_sensor):
            for j in range(i + 1, self.num_sensor):
                q1 = quantum_states[i].state_vector
                q2 = quantum_states[j].state_vector
                dot = np.dot(np.conj(q1), q2)
                if abs(dot) > Default.EPSILON:
                    return 0
        return 1

    def get_innerproducts(self, unitary_operator: Operator):
        '''get the innerproduct between every pair of <\phi_i|\phi_j>
        '''
        init_state = QuantumState(self.num_sensor, self.state_vector)
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)

        innerprods = []
        for i in range(self.num_sensor):
            for j in range(i + 1, self.num_sensor):
                q1 = quantum_states[i].state_vector
                q2 = quantum_states[j].state_vector
                dot = np.dot(np.conj(q1), q2)
                innerprods.append((i, j, round(abs(dot), 6)))
        dot = innerprods[0][2]
        conjecture = (self.num_sensor-1)/self.num_sensor * (1 - math.sqrt(1-dot**2))
        print(f'conjuecture = {conjecture}')
        return innerprods

    def _evaluate(self, init_state: QuantumState, unitary_operator: Operator, priors: list, povm: Povm, eval_metric: str) -> float:
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
            try:
                povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
            except Exception as e:
                raise e
        elif eval_metric == 'unambiguous':
            try:
                povm.semidefinite_programming_unambiguous(quantum_states, priors, debug=False)
            except Exception as e:
                raise e
        elif eval_metric == 'computational':
            povm.computational_basis(self.num_sensor, quantum_states, priors)
        else:
            raise Exception(f'unknown eval_metric: {eval_metric}!')
        return povm.theoretical_success


    def find_neighbors(self, qstate: QuantumState, i: int, step_size: float):
        '''find four random neighbors of qstate
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
            array.append(QuantumState(self.num_sensor, self.normalize_state(state_vector)))
        return array


    def hill_climbing(self, startState: QuantumState, seed: int, unitary_operator: Operator, priors: list, epsilon: float, \
                      step_size: list, decrease_rate: float, min_iteration: int, eval_metric: str):
        '''use the good old hill climbing method to optimize the initial state
        Args:
            startState       -- the start point of hill climbing
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors    -- prior probabilities
            epsilon   -- for termination
            step_size -- step sizes
            decrease_rate -- decrease rate
            min_iteration -- minimal number of iteration
            eval_metric -- 'min error' or 'unambiguous'
        Return:
            list, list -- a list of scores at each iteration, and a list of symmetry index at each iteration
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
        try:
            best_score = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        except Exception as e:
            raise e
        scores = [round(best_score, 7)]
        symmetry = self.get_symmetry_index(qstate.state_vector, unitary_operator)
        symmetries = [round(symmetry, 7)]
        terminate = False
        iteration = 0
        while terminate is False or iteration < min_iteration:
            iteration += 1
            before_score = best_score
            for i in range(N):
                neighbors = self.find_neighbors(qstate, i, step_size[i])
                best_step = -1
                for j in range(len(neighbors)):
                    try:
                        score = self._evaluate(neighbors[j], unitary_operator, priors, povm, eval_metric)
                    except Exception as e:
                        score = 0
                        print(f'solver issue at iteration={iteration}, dimension={i}, neighbor={j}, error={e}')
                    if score > best_score:
                        best_score = score
                        best_step = j
                if best_step == -1:
                    step_size[i] *= decrease_rate
                else:
                    qstate = neighbors[best_step]
                    step_size[i] *= decrease_rate
            scores.append(round(best_score, 7))
            symmetries.append(round(self.get_symmetry_index(qstate.state_vector, unitary_operator), 7))
            if best_score - before_score < epsilon:
                terminate = True
            else:
                terminate = False
        # print(f'POVM stats: {povm.sdp_info}')
        self._state_vector = qstate.state_vector
        self._optimze_method = 'Hill climbing'
        return scores, symmetries


    def get_symmetry_index(self, state_vector: np.ndarray, unitary_operator: Operator) -> float:
        '''given a state_vector in the computational basis and the unitary operator from the problem's input
           return its symmetry index under the custom_basis (computed from the unitary operator)
        '''
        custom_basis = Utility.generate_custombasis(self.num_sensor, U=unitary_operator)
        qstate_custom = QuantumStateCustomBasis(self.num_sensor, custom_basis=custom_basis, state_vector_custom=None, state_vector=state_vector)
        symmetry_index = qstate_custom.get_symmetry_index()
        # print(f'\nsymmetry index = {symmetry_index}')
        # qstate_custom.visualize_computation_in_custombasis(matplotlib=False)
        return symmetry_index


    def generate_init_temperature(self, qstate, init_step, N: int, unitary_operator: Operator, priors: list, povm: Povm, eval_metric) -> float:
        scores = []
        for i in range(N):
            neighbor = self.find_SA_neighbor(qstate, i, init_step)
            score = self._evaluate(neighbor, unitary_operator, priors, povm, eval_metric)
            scores.append(score)
        return np.std(scores)

    def find_SA_neighbor(self, qstate: QuantumState, i: int, step_size: float) -> QuantumState:
        '''return a random neighbor by updating the ith element of the state vector
        '''
        real = 2 * np.random.random() - 1
        imag = 2 * np.random.random() - 1
        direction = real + 1j*imag
        direction /= abs(direction)  # normalize
        state_vector = qstate.state_vector.copy()
        state_vector[i] += direction * step_size
        normalized_vector = self.normalize_state(state_vector)
        return QuantumState(self.num_sensor, normalized_vector)
    
    def jump_around(self, qstate: QuantumState) -> QuantumState:
        '''when stuck try jumping around: update ALL of the elements of the state vector
        '''
        state_vector = qstate.state_vector.copy()
        n = qstate.num_sensor
        for i in range(2 ** n):
            real = 2 * np.random.random() - 1
            imag = 2 * np.random.random() - 1
            direction = real + 1j*imag
            direction /= abs(direction)                           # normalize 
            state_vector[i] += direction * random.random() * 0.2  # random step size between [0, 0.2)
        normalized_vector = self.normalize_state(state_vector)
        return QuantumState(self.num_sensor, normalized_vector)

    def simulated_annealing(self, seed: int, unitary_operator: Operator, priors: list, init_step: float, stepsize_decreasing_rate: float, \
                                  epsilon: float, max_stuck: int, cooling_rate: float, min_iteration: int, eval_metric: str):
        '''use the simulated annealing to optimize the initial state
        Args:
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors    -- prior probabilities
            init_step -- the initial step size
            stepsize_decreasing_rate -- the rate that the steps are decreasing at each iteration
            epsilon   -- for termination
            max_stuck -- frozen criteria
            cooling_rate   -- cooling rate, the rate of the std of the previous iteration scores
            min_iteration  -- minimal number of iterations
            eval_metric    -- 'min error' or 'unambiguous'
        Return:
            list, list -- a list of scores at each iteration, and a list of symmetry index at each iteration
        '''
        print('Start simulated annealing...')
        np.random.seed(seed)
        random.seed(seed)
        qstate = QuantumState(self.num_sensor, random_state(nqubits=self.num_sensor))
        symmetry = self.get_symmetry_index(qstate.state_vector, unitary_operator)
        symmetries = [round(symmetry, 7)]
        # print(f'Random start:\n{qstate}')
        povm = Povm()
        N = 2**self.num_sensor
        init_temperature = self.generate_init_temperature(qstate, init_step, N, unitary_operator, priors, povm, eval_metric)
        temperature = init_temperature
        score1  = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        scores = [round(score1, 7)]
        terminate  = False
        eval_count = 0
        stuck_count = 0
        std_ratio = 1
        stepsize = init_step
        min_evaluation = min_iteration * 4*N
        while terminate is False or eval_count < min_evaluation:
            previous_score = score1
            scores_iteration = []
            for i in range(N):
                for _ in range(4):
                    neighbor = self.find_SA_neighbor(qstate, i, stepsize)
                    try:
                        score2 = self._evaluate(neighbor, unitary_operator, priors, povm, eval_metric)
                        scores_iteration.append(score2)
                        eval_count += 1
                    except Exception as e:
                        score2 = -100
                        print(f'solver issue at eval_count={eval_count}, error={e}')
                    dS = score2 - score1 # score2 is the score of the neighbor state, score1 is for current state
                    if dS > 0:
                        qstate = neighbor           # qstate improves
                        score1 = score2
                    else:  # S <= 0
                        prob = np.exp(dS / temperature)
                        if np.random.uniform(0, 1) < prob:
                            qstate = neighbor       # qstate becomes worse
                            score1 = score2
                        else:                       # qstate no change
                            pass
            scores.append(round(score1, 7))
            symmetries.append(round(self.get_symmetry_index(qstate.state_vector, unitary_operator), 7))
            if previous_score >= score1 - epsilon:
                stuck_count += 1
            else:
                stuck_count = 0
                terminate = False
            if stuck_count == max_stuck:
                terminate = True
            
            std = np.std(scores_iteration[-10:])
            std_ratio *= cooling_rate
            temperature = min(temperature*cooling_rate, std*std_ratio)
            stepsize *= stepsize_decreasing_rate
        # print(f'POVM stats: {povm.sdp_info}')
        self._state_vector = qstate.state_vector
        self._optimze_method = 'Simulated annealing'
        return scores, symmetries

    def simulated_annealing_new(self, seed: int, unitary_operator: Operator, priors: list, init_step: float, stepsize_decreasing_rate: float, \
                                      epsilon: float, max_stuck: int, cooling_rate: float, min_iteration: int, eval_metric: str):
        '''use the simulated annealing to optimize the initial state
        Args:
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors    -- prior probabilities
            init_step -- the initial step size
            stepsize_decreasing_rate -- the rate that the steps are decreasing at each iteration
            epsilon   -- for termination
            max_stuck -- frozen criteria
            cooling_rate   -- cooling rate, the rate of the std of the previous iteration scores
            min_iteration  -- minimal number of iterations
            eval_metric    -- 'min error' or 'unambiguous'
        Return:
            list, list -- a list of scores at each iteration, and a list of symmetry index at each iteration
        '''
        print('Start simulated annealing...')
        np.random.seed(seed)
        random.seed(seed)
        qstate = QuantumState(self.num_sensor, random_state(nqubits=self.num_sensor))
        symmetry = self.get_symmetry_index(qstate.state_vector, unitary_operator)
        symmetries = [round(symmetry, 7)]
        # print(f'Random start:\n{qstate}')
        povm = Povm()
        N = 2**self.num_sensor
        init_temperature = self.generate_init_temperature(qstate, init_step, N, unitary_operator, priors, povm, eval_metric)
        temperature = init_temperature
        score1  = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        scores = [round(score1, 7)]
        eval_count = 0
        stuck_count = 0
        std_ratio = 1
        stepsize = init_step
        min_evaluation = min_iteration * 4*N
        while eval_count < min_evaluation:
            previous_score = score1
            scores_iteration = []
            for i in range(N):
                for _ in range(4):
                    neighbor = self.find_SA_neighbor(qstate, i, stepsize)
                    try:
                        score2 = self._evaluate(neighbor, unitary_operator, priors, povm, eval_metric)
                        scores_iteration.append(score2)
                        eval_count += 1
                    except Exception as e:
                        score2 = -100
                        print(f'solver issue at eval_count={eval_count}, error={e}')
                    dS = score2 - score1 # score2 is the score of the neighbor state, score1 is for current state
                    if dS > 0:
                        qstate = neighbor           # qstate improves
                        score1 = score2
                    else:  # S <= 0
                        prob = np.exp(dS / temperature)
                        if np.random.uniform(0, 1) < prob:
                            qstate = neighbor       # qstate becomes worse
                            score1 = score2
                        else:                       # qstate no change
                            pass
            scores.append(round(score1, 7))
            symmetries.append(round(self.get_symmetry_index(qstate.state_vector, unitary_operator), 7))
            if previous_score >= score1 - epsilon:
                stuck_count += 1
            else:
                stuck_count = 0
            if stuck_count == max_stuck:
                temperature *= 1.5
                std_ratio *= 1.5
                stepsize *= 1.5
                stuck_count = 0
                continue
            
                # print(f'NO! Got stuck after {eval_count} number of evals.', end='  ')
                # success, (new_qstate, score2) = self.get_out_of_stuck(qstate, score1, unitary_operator, priors, povm, eval_metric)
                # if success:
                #     qstate = new_qstate
                #     score1 = score2
                #     stuck_count = 0
                # else:
                #     break
            
            std = np.std(scores_iteration[-10:])
            std_ratio *= cooling_rate
            temperature = min(temperature*cooling_rate, std*std_ratio)
            stepsize *= stepsize_decreasing_rate
        # print(f'POVM stats: {povm.sdp_info}')
        self._state_vector = qstate.state_vector
        self._optimze_method = 'Simulated annealing'
        return scores, symmetries


    def get_out_of_stuck(self, qstate: QuantumState, score1: float, unitary_operator: Operator, priors: list, povm: Povm, eval_metric: str):
        '''if gets out of stuck, then return a tuple:        True,  (new state, score)
           if fails to get out of stuc, then return a tuple: False, (None, None)
        '''
        i = 0
        max_trial = 20_000
        while i < max_trial:
            new_state = self.jump_around(qstate)
            score2 = self._evaluate(new_state, unitary_operator, priors, povm, eval_metric)
            if score2 > score1:
                print(f'YES! Got out of stuck at the {i}th trail')
                return True, (new_state, score2)
            i += 1
        print(f'NO! Failed to get out of stuck after {max_trial} trails')
        return False, (None, None)


    def simulated_annealing_startstate(self, start_state: QuantumState, seed: int, unitary_operator: Operator, priors: list, init_step: float, stepsize_decreasing_rate: float, \
                                             epsilon: float, max_stuck: int, cooling_rate: float, min_iteration: int, eval_metric: str):
        '''use the simulated annealing to optimize the initial state
           Not random start, but at a given startstate
        Args:
            start_state      -- a non-random start state
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors    -- prior probabilities
            init_step -- the initial step size
            stepsize_decreasing_rate -- the rate that the steps are decreasing at each iteration
            epsilon   -- for termination
            max_stuck -- frozen criteria
            cooling_rate   -- cooling rate, the rate of the std of the previous iteration scores
            min_iteration  -- minimal number of iterations
            eval_metric    -- 'min error' or 'unambiguous'
        Return:
            list, list -- a list of scores at each iteration, and a list of symmetry index at each iteration
        '''
        print('Start simulated annealing...')
        np.random.seed(seed)
        random.seed(seed)
        qstate = start_state
        symmetry = self.get_symmetry_index(qstate.state_vector, unitary_operator)
        symmetries = [round(symmetry, 7)]
        # print(f'Random start:\n{qstate}')
        povm = Povm()
        N = 2**self.num_sensor
        init_temperature = self.generate_init_temperature(qstate, init_step, N, unitary_operator, priors, povm, eval_metric)
        temperature = init_temperature
        score1  = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        scores = [round(score1, 7)]
        terminate  = False
        eval_count = 0
        stuck_count = 0
        std_ratio = 1
        stepsize = init_step
        min_evaluation = min_iteration * 4*N
        while terminate is False or eval_count < min_evaluation:
            previous_score = score1
            scores_iteration = []
            for i in range(N):
                for _ in range(4):
                    neighbor = self.find_SA_neighbor(qstate, i, stepsize)
                    try:
                        score2 = self._evaluate(neighbor, unitary_operator, priors, povm, eval_metric)
                        scores_iteration.append(score2)
                        eval_count += 1
                    except Exception as e:
                        score2 = -100
                        print(f'solver issue at eval_count={eval_count}, error={e}')
                    dS = score2 - score1 # score2 is the score of the neighbor state, score1 is for current state
                    if dS > 0:
                        qstate = neighbor           # qstate improves
                        score1 = score2
                    else:  # S <= 0
                        prob = np.exp(dS / temperature)
                        if np.random.uniform(0, 1) < prob:
                            qstate = neighbor       # qstate becomes worse
                            score1 = score2
                        else:                       # qstate no change
                            pass
            scores.append(round(score1, 7))
            symmetries.append(round(self.get_symmetry_index(qstate.state_vector, unitary_operator), 7))
            if previous_score >= score1 - epsilon:
                stuck_count += 1
            else:
                stuck_count = 0
                terminate = False
            if stuck_count == max_stuck:
                terminate = True
            
            std = np.std(scores_iteration[-10:])
            std_ratio *= cooling_rate
            temperature = min(temperature*cooling_rate, std*std_ratio)
            stepsize *= stepsize_decreasing_rate
        # print(f'POVM stats: {povm.sdp_info}')
        self._state_vector = qstate.state_vector
        self._optimze_method = 'Simulated annealing'
        return scores, symmetries


    def _compute_rank(self, fitness: list) -> list:
        '''
        Args:
            fitness -- a list of fitness
        '''
        fitness_rank = {}
        for i, fit in enumerate(sorted(fitness)):
            fitness_rank[fit] = i + 1
        rank = []
        for fit in fitness:
            rank.append(fitness_rank[fit])
        return rank

    def _selection(self, population: list, rank: list) -> list:
        '''Rank selection
        Args:
            population -- a population of quantum states
            rank -- the worst quantum state has a rank of 1, the best has a rank of len(population)
        Return:
            a list of two quantum states that are selected
        '''
        prefix_sum = list(accumulate(rank))
        upperbound = prefix_sum[-1]
        parents = []
        while len(parents) < 2:
            num = random.randint(1, upperbound)
            parent = bisect_left(prefix_sum, num)
            if parent not in parents:
                parents.append(parent)
        return [population[i] for i in parents]

    def _crossover(self, parents: list) -> list:
        '''2 point crossover
        Args:
            parents -- two quantum states
        Return:
            a list of two new offspring quantum states
        '''
        child0_state_vector = np.copy(parents[0].state_vector)
        child1_state_vector = np.copy(parents[1].state_vector)
        size = len(child0_state_vector)
        point1, point2 = random.sample(range(size + 1), 2)
        if point1 > point2:
            point1, point2 = point2, point1
        for i in range(point1, point2):
            child0_state_vector[i], child1_state_vector[i] = child1_state_vector[i], child0_state_vector[i]
        child0 = QuantumState(parents[0].num_sensor, self.normalize_state(child0_state_vector))
        child1 = QuantumState(parents[1].num_sensor, self.normalize_state(child1_state_vector))
        return [child0, child1]

    def _mutation(self, childs: list, mutation_rate: float, stepsize: float):
        '''randomly select one point and do a random neighbor, modify inplace
        Args:
            childs -- two quantum states
            mutation_rate -- the probability of happening mutation for an individual
            stepsize -- the step size for a mutation
        '''
        for child in childs:
            if random.random() < mutation_rate:
                size = len(child.state_vector)
                point = random.randint(0, size - 1)
                child.state_vector[point] += self.generate_random_direction() * stepsize
                child.state_vector = self.normalize_state(child.state_vector)
            
    def genetic_algorithm(self, seed: int, unitary_operator: Operator, priors: list, epsilon: float, population_size: int, mutation_rate: float, \
                                crossover_rate: float, init_step: float, stepsize_decreasing_rate: float, min_iteration: int, eval_metric: str):
        '''use genetic algorithm to optimize the initial state
        Args:
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors           -- prior probabilities
            epsilon          -- for termination
            population_size  -- the size of the population, i.e. number of solutions
            mutation_rate    -- the probability of doing mutation once during a offspring production
            crossover_rate   -- the probability of doing crossover once during a offspring production 
            init_step -- the initial step size
            stepsize_decreasing_rate -- the rate that the steps are decreasing at each iteration
            min_iteration    -- minimal number of iterations
            eval_metric      -- 'min error' or 'unambiguous'
        Return:
            list -- a list of scores at each iteration
        '''
        print('Start Genetic algorithm...')
        # initialize a population
        np.random.seed(seed)
        population = []
        fitness = []
        povm = Povm()
        best_fitness = 0
        best_qstate = None
        for _ in range(population_size):
            qstate = QuantumState(self.num_sensor, random_state(nqubits=self.num_sensor))
            population.append(qstate)
            fit = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
            fitness.append(fit)
            if fit > best_fitness:
                best_fitness = fit
                best_qstate = qstate
        scores = [round(best_fitness, 7)]
        symmetry = self.get_symmetry_index(best_qstate.state_vector, unitary_operator)
        symmetries = [round(symmetry, 7)]
        iteration = 0
        stepsize = init_step
        terminate = False
        while terminate is False or iteration < min_iteration:
            before_fitness = best_fitness
            iteration += 1
            rank = self._compute_rank(fitness)
            child_fitness = []
            child_population = []
            while len(child_population) < population_size:
                # selection
                parents = self._selection(population, rank)
                # crossover
                if random.random() < crossover_rate:
                    childs = self._crossover(parents)
                else:
                    childs = [QuantumState(p.num_sensor, np.array(p.state_vector)) for p in parents]
                # mutation
                self._mutation(childs, mutation_rate, stepsize)
                child_population.extend(childs)

            for qstate in child_population:
                child_fitness.append(self._evaluate(qstate, unitary_operator, priors, povm, eval_metric))
            
            # mix parent and child together and select the top 50%
            fitness.extend(child_fitness)
            threshold = sorted(fitness)[population_size - 1]
            new_population = []
            new_fitness = []
            best_fitness = 0
            best_qstate = None
            for qstate, fit in zip(population + child_population, fitness):
                if fit > threshold:
                    new_population.append(qstate)
                    new_fitness.append(fit)
                if fit > best_fitness:
                    best_fitness = fit
                    best_qstate = qstate

            scores.append(round(best_fitness, 7))
            symmetries.append(round(self.get_symmetry_index(best_qstate.state_vector, unitary_operator), 7))
            stepsize *= stepsize_decreasing_rate
            population = new_population
            fitness = new_fitness
            if best_fitness - before_fitness < epsilon:
                terminate = True
            else:
                terminate = False

        best_qstate  = population[0]
        best_fitness = fitness[0]
        for i in range(1, population_size):
            if fitness[i] > best_fitness:
                best_fitness = fitness[i]
                best_qstate = population[i]
        self._state_vector = best_qstate.state_vector
        self._optimze_method = 'Genetic algorithm'
        return scores, symmetries


    def particle_swarm_optimization(self, seed: int, unitary_opeartor: Operator, priors: list, epsilon: float, population_size: int, \
                                    w: float, eta1: float, eta2: float, init_step_size: float, min_iteration, eval_metric: str):
        '''use particle swarm optimization to optimize the initial state
        Args:
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors           -- prior probabilities
            epsilon          -- for termination
            population_size  -- the size of the population, i.e. number of solutions
            w                -- inertia weight
            eta1             -- cognitive constant
            eta2             -- social constant
            init_step        -- the initial step size
            min_iteration    -- minimal number of iterations
            eval_metric      -- 'min error' or 'unambiguous'
        Return:
            list -- a list of scores at each iteration
        '''
        class Particle:
            def __init__(self, qstate: QuantumState, fitness: float, velocity: complex):
                self.qstate = qstate       # current state
                self.fitness = fitness
                self.velocity = velocity
                self.pbest = qstate        # personal best state
                self.pbest_fitness = fitness
                self.dimension = len(qstate.state_vector)

            def update_velocity(self, gbest: QuantumState):
                '''update the velocity of the particle
                '''
                for i in range(self.dimension):
                    r1 = random.random()
                    r2 = random.random()
                    v_cognitive = eta1 * r1 * (self.pbest.state_vector[i] - self.qstate.state_vector[i])
                    v_social    = eta2 * r2 * (gbest.state_vector[i] - self.qstate.state_vector[i])
                    self.velocity[i] = w * self.velocity[i] + v_cognitive + v_social

            def update_position(self):
                '''update the position (qstate) of the particle, also the fitness value
                '''
                for i in range(self.dimension):
                    self.qstate.state_vector[i] += self.velocity[i]
                self.qstate.state_vector = self.qstate.normalize_state(self.qstate.state_vector)
            
            def update_fitness(self, fitness):
                self.fitness = fitness

        print('Start particle swarm optimization...')
        np.random.seed(seed)
        random.seed(seed)
        povm = Povm()
        swarm = []
        gbest = None        # globel best qstate
        gbest_fitness = 0   # the fitness of the globel best qstate
        for _ in range(population_size):
            qstate = QuantumState(self.num_sensor, random_state(nqubits=self.num_sensor))   # random initialization
            fitness = self._evaluate(qstate, unitary_opeartor, priors, povm, eval_metric)
            if fitness > gbest_fitness:
                gbest = qstate
                gbest_fitness = fitness
            velocity = [init_step_size * self.generate_random_direction() for _ in range(2**self.num_sensor)]
            particle = Particle(qstate, fitness, velocity)
            swarm.append(particle)

        scores = [round(gbest_fitness, 7)]
        iteration = 0
        terminate = False
        # while terminate is False or iteration < min_iteration:
        while iteration < min_iteration:
            iteration += 1
            for particle in swarm:
                particle.update_velocity(gbest)
                particle.update_position()
                particle.update_fitness(self._evaluate(particle.qstate, unitary_opeartor, priors, povm, eval_metric))
                if particle.fitness > particle.pbest_fitness:
                    particle.pbest = QuantumState(self.num_sensor, np.copy(particle.qstate.state_vector))
                    particle.pbest_fitness = particle.fitness
                    if particle.fitness > gbest_fitness:
                        gbest = QuantumState(self.num_sensor, np.copy(particle.qstate.state_vector))
                        gbest_fitness = particle.fitness
            scores.append(round(gbest_fitness, 7))
            if scores[-1] - scores[-2] < epsilon:
                terminate = True
            else:
                terminate = False

        self._optimze_method = 'Particle swarm'
        self._state_vector = gbest.state_vector
        return scores

