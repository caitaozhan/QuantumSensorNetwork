from numpy import np
from utility import Utility
from quantum_state import QuantumState
from qiskit.quantum_info import random_unitary
from qiskit.visualization import plot_bloch_multivector
from qiskit_textbook.tools import random_state, array_to_latex
from qiskit.quantum_info.operators.operator import Operator


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
    print('shape', qs.state_vector.shape[0])

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
    test1()
    # test4()

