'''
Optimizing initial state
'''

import copy
from matplotlib.pyplot import cool
import numpy as np
import math
from qiskit_textbook.tools import random_state
from qiskit.quantum_info.operators.operator import Operator
from quantum_state import QuantumState
from utility import Utility
from povm import Povm
from input_output import Default


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

    def set_statevector_from_str(self, s: str):
        '''set the self._state_vector from the __str__() string
        Args:
            s -- the __str__() string
            Example:

            Guess
            Initial state:
            |000>:  0.31088  0.3194i
            |001>:  0.42490 -0.0000i
            |010>:  0.42490 -0.0000i
            |011>: -0.19850  0.2039i
            |100>:  0.42490 -0.0000i
            |101>: -0.19850  0.2039i
            |110>: -0.19850  0.2039i
            |111>: -0.00349 -0.1295i
        '''
        statevector = []
        s = s.split('\n')
        for line in s:
            if '>:' in line:
                line = line.split()
                real = line[1]
                imag = line[2]
                real = float(real.strip())
                imag = float(imag[:-1].strip())
                statevector.append(complex(real, imag))
        self._state_vector = np.array(statevector)

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

    def upperbound(self, unitary_operator: Operator, priors: list):
        '''an upper bound from equation (10) of this paper: https://arxiv.org/pdf/1509.04592.pdf
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
            try:
                povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
            except Exception as e:
                raise e
        elif eval_metric == 'unambiguous':
            try:
                povm.semidefinite_programming_unambiguous(quantum_states, priors, debug=True)
            except Exception as e:
                raise e
        else:
            raise Exception(f'unknown eval_metric: {eval_metric}!')
        return povm.therotical_success

    def find_neighbors(self, init_state: QuantumState, i: int, mod_step: list, amp_step: float):
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

    def find_neighbors_realimag(self, init_state: QuantumState, i: int, real_step: float, imag_step: float):
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
        array = []
        init_state_vector[i] = (z.real + real_step) + 1j*z.imag
        array.append(QuantumState(self.num_sensor, self.normalize_state(init_state_vector)))
        init_state_vector[i] = (z.real - real_step) + 1j*z.imag
        array.append(QuantumState(self.num_sensor, self.normalize_state(init_state_vector)))
        init_state_vector[i] = z.real + 1j*(z.imag + imag_step)
        array.append(QuantumState(self.num_sensor, self.normalize_state(init_state_vector)))
        init_state_vector[i] = z.real + 1j*(z.imag - imag_step)
        array.append(QuantumState(self.num_sensor, self.normalize_state(init_state_vector)))
        return array

    def _generate_random_direction(self):
        '''generate a random direction
        '''
        real = 2 * np.random.random() - 1
        imag = 2 * np.random.random() - 1
        direction = real + 1j*imag
        direction /= abs(direction)     # normalize
        return direction

    def find_neighbors_random(self, qstate: QuantumState, i: int, step_size: list):
        '''find four random neighbors of
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
            direction = self._generate_random_direction()
            state_vector[i] += direction * step_size
            array.append(QuantumState(self.num_sensor, self.normalize_state(state_vector)))
        return array


    def hill_climbing(self, startState: QuantumState, seed: int, unitary_operator: Operator, priors: list, epsilon: float, mod_step: list, \
                      amp_step: list, decrease_rate: float, min_iteration: int, eval_metric: str, random_neighbor: bool, realimag_neighbor: bool):
        '''use the good old hill climbing method to optimize the initial state
        Args:
            startState       -- the start point of hill climbing
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors   -- prior probabilities
            epsilon  -- for termination
            mod_step -- step size for modulus
            amp_step -- step size for amplitude
            decrease_rate -- decrease rate
            min_iteration -- minimal number of iteration
            eval_metric -- 'min error' or 'unambiguous'
            random_neighbor -- random neighbor or predefined directions
            realimag_neighbor -- change real and imaginary parts
        Return:
            list -- a list of scores at each iteration
        '''
        if random_neighbor:
            return self._hill_climbing_randomneighbor(startState, seed, unitary_operator, priors, epsilon, \
                                                      mod_step, decrease_rate, min_iteration, eval_metric)
        if realimag_neighbor:
            return self._hill_climbing_realimagneighbor(startState, seed, unitary_operator, priors, epsilon, \
                                                        mod_step, amp_step, decrease_rate, min_iteration, eval_metric)

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
        terminate = False
        iteration = 0
        while terminate is False or iteration < min_iteration:
            iteration += 1
            before_score = best_score
            for i in range(N):
                neighbors = self.find_neighbors(qstate, i, mod_step[i], amp_step[i])
                best_step = -1
                for j in range(len(neighbors)):
                    try:
                        score = self._evaluate(neighbors[j], unitary_operator, priors, povm, eval_metric)
                    except Exception as e:
                        # print(e)
                        score = 0
                        print(f'solver issue at iteration={iteration}, dimension={i}, neighbor={j}, error={e}')
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
            scores.append(round(best_score, 7))
            if best_score - before_score < epsilon:
                terminate = True
            else:
                terminate = False

        self._state_vector = qstate.state_vector
        self._optimze_method = 'Hill climbing'
        return scores


    def _hill_climbing_randomneighbor(self, startState: QuantumState, seed: int, unitary_operator: Operator, priors: list, epsilon: float, \
                                      step_size: list, decrease_rate: float, min_iteration: int, eval_metric: str):
        '''use the good old hill climbing method to optimize the initial state
        Args:
            startState       -- the start point of hill climbing
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors   -- prior probabilities
            epsilon  -- for termination
            init_step -- step size for modulus
            decrease_rate -- decrease rate
            min_iteration -- minimal number of iteration
            eval_metric -- 'min error' or 'unambiguous'
            random_neighbor -- random neighbor or predefined directions
        Return:
            list -- a list of scores at each iteration
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
        terminate = False
        iteration = 0
        while terminate is False or iteration < min_iteration:
            iteration += 1
            before_score = best_score
            for i in range(N):
                neighbors = self.find_neighbors_random(qstate, i, step_size[i])
                best_step = -1
                for j in range(len(neighbors)):
                    try:
                        score = self._evaluate(neighbors[j], unitary_operator, priors, povm, eval_metric)
                    except Exception as e:
                        # print(e)
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
            if best_score - before_score < epsilon:
                terminate = True
            else:
                terminate = False

        self._state_vector = qstate.state_vector
        self._optimze_method = 'Hill climbing'
        return scores


    def _hill_climbing_realimagneighbor(self, startState: QuantumState, seed: int, unitary_operator: Operator, priors: list, epsilon: float, \
                                      real_step: list, imag_step: list, decrease_rate: float, min_iteration: int, eval_metric: str):
        '''use the good old hill climbing method to optimize the initial state
        Args:
            startState       -- the start point of hill climbing
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors   -- prior probabilities
            epsilon  -- for termination
            real_step -- step size for real part
            imag_step -- step size for imaginary part
            decrease_rate -- decrease rate
            min_iteration -- minimal number of iteration
            eval_metric -- 'min error' or 'unambiguous'
        Return:
            list -- a list of scores at each iteration
        '''
        print('\nStart hill climbing (real, imaginary version)...')
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
        terminate = False
        iteration = 0
        while terminate is False or iteration < min_iteration:
            iteration += 1
            before_score = best_score
            for i in range(N):
                neighbors = self.find_neighbors_realimag(qstate, i, real_step[i], imag_step[i])
                best_step = -1
                for j in range(len(neighbors)):
                    try:
                        score = self._evaluate(neighbors[j], unitary_operator, priors, povm, eval_metric)
                    except Exception as e:
                        # print(e)
                        score = 0
                        print(f'solver issue at iteration={iteration}, dimension={i}, neighbor={j}, error={e}')
                    if score > best_score:
                        best_score = score
                        best_step = j
                if best_step == -1:
                    real_step[i] *= decrease_rate
                    imag_step[i] *= decrease_rate
                elif best_step in [0, 1]:
                    qstate = neighbors[best_step]
                    real_step[i] *= decrease_rate
                else: # best_step in [2, 3]:
                    qstate = neighbors[best_step]
                    imag_step[i] *= decrease_rate
            scores.append(round(best_score, 7))
            if best_score - before_score < epsilon:
                terminate = True
            else:
                terminate = False

        self._state_vector = qstate.state_vector
        self._optimze_method = 'Hill climbing'
        return scores

    def generate_init_temperature(self, qstate, init_step, N: int, unitary_operator: Operator, priors: list, povm: Povm, eval_metric):
        scores = []
        for i in range(N):
            neighbor = self.find_SA_neighbor(qstate, i, init_step)
            score = self._evaluate(neighbor, unitary_operator, priors, povm, eval_metric)
            scores.append(score)
        return np.std(scores)

    def find_SA_neighbor(self, qstate: QuantumState, i: int, step_size: float):
        real = 2 * np.random.random() - 1
        imag = 2 * np.random.random() - 1
        direction = real + 1j*imag
        direction /= abs(direction)  # normalize
        state_vector = qstate.state_vector.copy()
        state_vector[i] += direction * step_size
        normalized_vector = self.normalize_state(state_vector)
        return QuantumState(self.num_sensor, normalized_vector)

    def simulated_annealing(self, seed: int, unitary_operator: Operator, priors: list, init_step: float, epsilon: float, \
                                  max_stuck: int, cooling_rate: float, min_iteration: int, eval_metric: str):
        '''use the simulated annealing to optimize the initial state
        Args:
            seed             -- random seed
            unitary_operator -- describe the interaction with the environment
            priors    -- prior probabilities
            init_step -- the initial step size
            epsilon   -- for termination
            max_stuck -- frozen criteria
            cooling_rate   -- cooling rate
            min_iteration -- minimal number of iterations
            eval_metric    -- 'min error' or 'unambiguous'
        Return:
            list -- a list of scores at each iteration
        '''
        print('\nStart simulated annealing...')
        np.random.seed(seed)
        qstate = QuantumState(self.num_sensor, random_state(nqubits=self.num_sensor))
        print(f'Random start:\n{qstate}')
        povm = Povm()
        N = 2**self.num_sensor
        init_temperature = self.generate_init_temperature(qstate, init_step, N, unitary_operator, priors, povm, eval_metric)
        temperature = init_temperature
        score1  = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        scores = [round(score1, 7)]
        terminate  = False
        eval_count = 0
        stuck_count = 0
        min_evaluation = min_iteration * 4*N
        while terminate is False or eval_count < min_evaluation:
            previous_score = score1
            stepsize = init_step * temperature / init_temperature
            for i in range(N):
                for _ in range(4):
                    neighbor = self.find_SA_neighbor(qstate, i, stepsize)
                    try:
                        score2 = self._evaluate(neighbor, unitary_operator, priors, povm, eval_metric)
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
            scores.append(round(score2, 7))
            if previous_score >= score2 - epsilon:
                stuck_count += 1
            else:
                stuck_count = 0
                terminate = False
            if stuck_count == max_stuck:
                terminate = True
            
            # check optimal

            temperature *= cooling_rate

        self._state_vector = qstate.state_vector
        self._optimze_method = 'Simulated annealing'
        return scores
