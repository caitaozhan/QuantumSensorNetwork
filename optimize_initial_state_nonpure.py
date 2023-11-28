import math
import numpy as np
import copy
from qiskit.quantum_info.operators.operator import Operator
from typing import List
from quantum_state_nonpure import QuantumStateNonPure
from povm import Povm
from depolarising_noise import DepolarisingNoise
from utility import Utility
from input_output import Default
from equation_generator import EquationGenerator


class OptimizeInitialStateNonpure(QuantumStateNonPure):
    '''A non pure quantum state that has optimization capabilities
       cannot do heuristic search for non pure quantum states
       only make guess on initial state
    '''
    def __init__(self, num_sensor: int):
        super().__init__(num_sensor=num_sensor, density_matrix=None)
        self._optimze_method = ''

    def __str__(self):
        s = f'\n{self.optimize_method}\nInitial state:\n'
        parent = super().__str__()
        return s + parent

    @property
    def optimize_method(self):
        return self._optimze_method

    def theorem(self, unitary_operator: Operator, unitary_theta: float) -> Povm:
        '''implementing the theorem (corollary + conjecture)
           return the POVM using the optimal initial state
        '''
        self._optimze_method = 'Theorem'
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
            state_vector = np.sum(states, axis=0)
            self.density_matrix = np.outer(state_vector, np.conj(state_vector))
        else:                                                                  # non mutual orthogonal situation (conjecture)
            # partition = eg.optimal_solution_smallerT_i(unitary_theta, partition_i)
            partition = eg.optimal_solution_smallerT()
            coeff = np.sqrt(1/len(partition))
            states = []
            for ev in partition:
                e_vector = Utility.eigenvector(v1, v2, ev)
                states.append(coeff * e_vector)
            state_vector = np.sum(states, axis=0)
            self.density_matrix = np.outer(state_vector, np.conj(state_vector))

        if self.check_matrix() is False:
            raise Exception('Oops! Not a valid quantum state')

    def check_matrix(self) -> bool:
        '''check if the trace of the density matrix equals 1
        '''
        return abs(np.trace(self.density_matrix) - 1) < Default.EPSILON

    def get_povm_nonoise(self, unitary_operator: Operator, priors: List[float], eval_metric: str) -> Povm:
        '''return the povm given the initial state and without noise
        Args:
            unitary_operator -- unitary operator that describes the evolution
            priors           -- prior probabilities
            eval_metrix      -- 'min error'
        '''
        init_state = QuantumStateNonPure(self.num_sensor, self.density_matrix)
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)
        povm = Povm()
        if eval_metric == 'min error':
            povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
        else:
            raise Exception(f'unknown eval_metric: {eval_metric}')
        return povm

    def get_povm_noise(self, unitary_operator: Operator, priors: List[float], eval_metric: str, depolarising_noise: DepolarisingNoise) -> Povm:
        '''return the povm computed on the final states evolved from the noisy initial state
        Args:
            unitary_operator -- unitary operator that describes the evolution
            priors           -- prior probabilities
            eval_metrix      -- 'min error'
            depolarising_noise -- the noise
        '''
        init_state = QuantumStateNonPure(self.num_sensor, self.density_matrix)
        init_state.evolve(Operator(depolarising_noise.get_matrix(init_state.num_sensor)))  # first apply depolarsing noise
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)
        povm = Povm()
        if eval_metric == 'min error':
            povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
        else:
            raise Exception(f'unknown eval_metric: {eval_metric}')
        return povm

    def ghz(self, unitary_operator: Operator):
        '''GHZ state in the basis composed of U's eigen vectors
        '''
        self._optimized_method = 'GHZ'
        e_vals, e_vectors = np.linalg.eig(unitary_operator._data)
        theta1 = Utility.get_theta(e_vals[0].real, e_vals[0].imag)
        theta2 = Utility.get_theta(e_vals[1].real, e_vals[1].imag)
        v1 = e_vectors[:, 0]    # v1 is positive
        v2 = e_vectors[:, 1]    # v2 is negative
        if theta1 < theta2:
            v1, v2, = v2, v1

        coeff = np.sqrt(1/2)
        states = []
        for ev in ['0'*self.num_sensor, '1'*self.num_sensor]:
            e_vector = Utility.eigenvector(v1, v2, ev)
            states.append(coeff * e_vector)
        state_vector = np.sum(states, axis=0)
        self.density_matrix = np.outer(state_vector, np.conj(state_vector))
        
        if self.check_matrix() is False:
            raise Exception('Oops! Not a valid quantum state')

    def non_entangle(self, unitary_operator: Operator):
        '''Non entangled uniform superposition state in the basis composed of U's eigen vectors
        '''
        self._optimized_method = 'Non entangle'
        e_vals, e_vectors = np.linalg.eig(unitary_operator._data)
        theta1 = Utility.get_theta(e_vals[0].real, e_vals[0].imag)
        theta2 = Utility.get_theta(e_vals[1].real, e_vals[1].imag)
        v1 = e_vectors[:, 0]    # v1 is positive
        v2 = e_vectors[:, 1]    # v2 is negative
        if theta1 < theta2:
            v1, v2, = v2, v1

        N = 2 ** self.num_sensor
        coeff = np.sqrt(1/N)
        states = []
        for i in range(N):
            ev = bin(i)[2:]
            if len(ev) < self.num_sensor:
                ev = '0'*(self.num_sensor - len(ev)) + ev
            e_vector = Utility.eigenvector(v1, v2, ev)
            states.append(coeff * e_vector)
        state_vector = np.sum(states, axis=0)
        self.density_matrix = np.outer(state_vector, np.conj(state_vector))
        
        if self.check_matrix() is False:
            raise Exception('Oops! Not a valid quantum state')

    def evaluate_noise(self, unitary_operator: Operator, priors: List[float], povm: Povm, depolarising_noise: DepolarisingNoise, repeat: int) -> float:
        '''do simulation by appling the (not-considering noise) POVM on the set of final states that considered noise
        Args:
            unitary_operator   -- unitary operator that describes the evolution
            priors             -- prior probabilities
            eval_metrix        -- 'min error'
            povm               -- the povm
            depolarising_noise -- the depolarising noise
            repeat             -- # of repetation of single shot measurement
        Return:
            probability of error
        '''
        init_state = QuantumStateNonPure(self.num_sensor, self.density_matrix)
        init_state.evolve(Operator(depolarising_noise.get_matrix(init_state.num_sensor)))  # first apply depolarising noise
        quantum_states = []
        for i in range(self.num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, self.num_sensor, i)
            init_state_copy = copy.deepcopy(init_state)
            init_state_copy.evolve(evolve_operator)
            quantum_states.append(init_state_copy)
        return povm.simulate(quantum_states, priors, repeat=repeat)


