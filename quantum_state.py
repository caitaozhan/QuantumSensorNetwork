import numpy as np
import math
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info import random_unitary, random_state
from qiskit.visualization import plot_bloch_multivector
from qiskit_textbook.tools import random_state, array_to_latex
from utility import Utility


class QuantumState:
    '''Encapsulate a quantum state, i.e. a complex vector in the Hilbert space.
       Currently using two-level quantum state, i.e., qubits
       One quantum detector is represented by a single qubit quantum state
       N quantum detector are represented by a N qubit quantum state
    '''
    def __init__(self, num_detector: int, psi: np.array = None):
        '''
        Args:
            num_detector: number of detector
            psi:          greek letter psi, a state vector of dimension 2**num_detector
        '''
        self._num_detector = num_detector
        self._psi = psi

    @property
    def num_detector(self):
        return self._num_detector

    @property
    def psi(self):
        return self._psi

    def init_random_state(self, seed: int = None):
        '''init a random quantum state'''
        if seed is not None:
            np.random.seed(seed)
        self._psi = random_state(self.num_detector)

    def evolve(self, operator: Operator):
        '''the evolution of a quantum state
        Args:
            operator: describe the interaction of the environment, essentily a matrix
        '''
        psi_dim = self._psi.shape[0]  # for N qubits, the dimension is 2**N
        operator_dim = np.product(operator.input_dims()) # for N qubits, the input_dims() return (2, 2, ..., 2), N twos.
        if psi_dim == operator_dim:
            self._psi = np.dot(operator._data, self._psi)
        else:
            raise Exception('psi and operator dimension not equal')

    def __str__(self):
        string = 'Quantum state is:\n'
        index = 0
        num_of_bit = math.ceil(math.log2(len(self.psi)))
        for index, cmpl in enumerate(self.psi):
            state = Utility.integer2bit(index, num_of_bit)
            cmpl = str(cmpl)[1:-1]     # (-0.14139694215565082+0.3754394106901288j)
            string += f'|{state}>: {cmpl}\n'
        return string


def test1():
    '''test random_state, Utility.norm_square'''
    np.random.seed(1)
    psi = random_state(1)
    print('psi', psi)
    summ = 0
    for cmpl in psi:
        summ += Utility.norm_squared(cmpl)
    print('sum of probability', summ)
    array_to_latex(psi, pretext='|\\psi\\rangle =')
    fig = plot_bloch_multivector(psi)
    fig.savefig('tmp.png')

def test2():
    '''test qs.init_random_state'''
    qs = QuantumState(num_detector=2)
    qs.init_random_state(seed=0)
    print(qs)
    print('shape', qs.psi.shape[0])

def test3():
    '''random unitary operator, identity operator, tensor product'''
    operator = random_unitary(2)
    print(operator, operator.is_unitary())

    # operator = random_unitary(4)
    # print(operator, operator.is_unitary())

    # operator = random_unitary(8)
    # print(operator, operator.is_unitary())

    identity_oper = Operator(np.eye(2**1))
    print(identity_oper, identity_oper.is_unitary())
    
    tensor_product = operator.tensor(identity_oper)
    print(tensor_product, tensor_product.is_unitary())

def test4():
    '''evolution'''
    seed = 0
    qs = QuantumState(2)
    qs.init_random_state(seed=seed)
    print(qs)
    nqubit = 1
    unitary_operator = random_unitary(2**nqubit, seed=seed)
    identity_operator = Operator(np.eye(2**nqubit))
    tensor_product = unitary_operator.tensor(identity_operator)
    print('Evolution...\n')
    qs.evolve(tensor_product)
    print(qs)

    qs = QuantumState(2)
    qs.init_random_state(seed=seed)
    nqubit = 1
    unitary_operator = random_unitary(2**nqubit, seed=seed)
    identity_operator = Operator(np.eye(2**nqubit))
    tensor_product = identity_operator.tensor(unitary_operator)
    print('Evolution...\n')
    qs.evolve(tensor_product)
    print(qs)

if __name__ == '__main__':
    test4()

