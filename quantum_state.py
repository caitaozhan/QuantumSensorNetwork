import numpy as np
import math
from qiskit.visualization import plot_bloch_multivector
from qiskit_textbook.tools import random_state, array_to_latex
from utility import Utility


class QuantumState:
    '''Encapsulate a quantum state, i.e. a complex vector in the Hilbert space.
       Currently using two-level quantum state, i.e., qubits
       One quantum detector is represented by a single qubit quantum state
       N quantum detector are represented by a N qubit quantum state
    '''
    def __init__(self, number_detector):
        self.number = number_detector   # number of detectors, i.e. number of qubits
        self.psi = None                 # state vector

    def init_random_state(self, seed=None):
        '''init a random quantum state'''
        if seed is not None:
            np.random.seed(seed)
        self.psi = random_state(self.number)

    def init_state(self, psi):
        self.psi = psi

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
    qs = QuantumState(number=2)
    qs.init_random_state()
    print(qs)

if __name__ == '__main__':
    test2()
