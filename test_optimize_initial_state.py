'''
Optimize the initial state
'''

from qiskit.quantum_info import random_unitary
from optimize_initial_state import OptimizeInitialState
from optimize_initial_state_nonentangled import OptimizeInitialStateNonentangled


def test1():
    seed = 2
    num_sensor = 2 
    U = random_unitary(2, seed = seed)
    optis = OptimizeInitialState(num_sensor=num_sensor)
    optis.guess(U)
    print(optis)


def test2():
    num_sensor = 3
    seed = 0
    optis_ne = OptimizeInitialStateNonentangled(num_sensor=num_sensor)
    optis_ne.init_random_state(seed)
    print(optis_ne)



if __name__ == '__main__':
    test2()
