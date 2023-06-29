import copy
import numpy as np
from qiskit.quantum_info.operators.operator import Operator

from quantum_state_custombasis import QuantumStateCustomBasis
from utility import Utility
from povm import Povm


class OptimizeInitialStateCustom(QuantumStateCustomBasis):
    '''A quantum state on a Custom Basis that has optimization capabilities
       NOTE: the Inheritance design is ackward, should Refactor
    '''
    def __init__(self, num_sensor: int, custom_basis: list):
        super().__init__(num_sensor=num_sensor, custom_basis=custom_basis)
        self._optimize_method = ''

    @property
    def optimize_method(self):
        return self._optimize_method
    
    def __str__(self) -> str:
        s = f'\n{self.optimize_method}\nInitial state:\n'
        parent = super().__str__()
        return s + parent
    
    def _evaluate(self, init_state: QuantumStateCustomBasis, unitary_operator: Operator, priors: list, povm: Povm, eval_metric: str) -> float:
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


    def find_neighbors(self, qstate: QuantumStateCustomBasis, i: int, step_size: float):
        '''find 2 neighbors of qstate
        Args:
            qstate -- initial state
            i        -- ith element of the state vector to be modified
            step_size -- step size for modulus
        Return:
            list -- a list of QuantumState object
        '''
        array = []
        for j in range(2):
            state_vector_custom = qstate.state_vector_custom.copy()
            state_vector_custom[i] += (-1)**j * step_size
            array.append(QuantumStateCustomBasis(self.num_sensor, custom_basis=qstate.custom_basis, state_vector_custom=self.normalize_state(state_vector_custom)))
        return array

    def get_symmetry_index(self, qstate: QuantumStateCustomBasis) -> float:
        return qstate.get_symmetry_index()

    def hill_climbing(self, seed: int, unitary_operator: Operator, priors: list, epsilon: float, \
                      step_size: list, decrease_rate: float, min_iteration: int, eval_metric: str):
        '''use the good old hill climbing method to optimize the initial state
        Args:
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
        np.random.seed(seed)
        qstate = QuantumStateCustomBasis(self.num_sensor, custom_basis=self.custom_basis)
        qstate.init_random_state_realnumber()
        N = 2**self.num_sensor
        povm = Povm()
        best_score = self._evaluate(qstate, unitary_operator, priors, povm, eval_metric)
        scores = [round(best_score, 7)]
        symmetry = self.get_symmetry_index(qstate)
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
            symmetries.append(round(self.get_symmetry_index(qstate), 7))
            terminate = True if best_score - before_score < epsilon else False
        self._state_vector_custom = qstate.state_vector_custom
        self._state_vector = qstate.state_vector
        self._optimize_method = 'Hill climbing C'
        return scores, symmetries
        
