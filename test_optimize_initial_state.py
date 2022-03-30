'''Optimize the initial state
'''

from qiskit.quantum_info import random_unitary
from optimize_initial_state import OptimizeInitialState


def test1():
    seed = 2
    num_sensor = 2 
    U = random_unitary(2, seed = seed)
    optis = OptimizeInitialState(num_sensor=num_sensor)
    optis.guess(U)
    print(optis)



if __name__ == '__main__':
    test1()
