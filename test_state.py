import numpy as np
import math
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


def test5():
    '''A random U with many different random initial state |a>, compute the minimum error and unambiguous discrimination error
    '''
    print('test_state.test5()')
    print('A random U with many different random initial state |a>, compute the minimum error and unambiguous discrimination error.\n')
    U = random_unitary(dims=2, seed=0)
    U_ct = U.conjugate().transpose()
    print('U')
    Utility.print_matrix(U.data)
    print('\nU_ct')
    Utility.print_matrix(U_ct.data)
    UU_ct = U.tensor(U_ct)
    print('\nU tensor U_ct')
    Utility.print_matrix(UU_ct.data)
    print()

    for i in range(20):
        qs = QuantumState(num_detector=2)
        qs.init_random_state(seed=i)
        vec = qs.state_vector
        print(f'seed = {i}, |a> = ', end='')
        Utility.print_matrix([vec])
        val = np.dot(np.dot(np.conj(vec), UU_ct.data), vec)
        error = 0.5 * (1 - math.sqrt(1 - abs(val)**2))
        print(f'P(min error) = {error:.4f}; P(unambiguous) = {abs(val):.4f}')
        print()

        # U_ctU = U_ct.tensor(U)
        # val = np.dot(np.dot(np.conj(vec), U_ctU.data), vec)
        # print(f'|<a|U_ct(tensor)U|a>| = {abs(val)}')
        # print()


if __name__ == '__main__':
    # test1()
    # test4()
    test5()

