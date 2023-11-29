'''Modeling quantum noise
https://pennylane.ai/qml/demos/tutorial_noisy_circuits/
'''
import numpy as np
from typing import List


class QuantumNoise:
    '''base class of quantum noise for n qubits/sensors
    '''
    def __init__(self, n: int):
        '''
        Args:
            n -- number of qubits/sensors
        '''
        self.n = n
        self.kraus = []

    def __str__(self):
        s = ''
        for i, kraus in enumerate(self.kraus):
            s += f'{i}:\n {kraus}\n'
        return s
    
    def check_kraus(self) -> bool:
        '''check if the kruas is valid by definition
        '''
        N = 2 ** self.n
        summ = np.zeros((N, N), dtype=np.complex128)
        for k in self.kraus:
            k_dagger = np.transpose(np.conj(k))
            summ += k_dagger @ k
        identity = np.eye(N, dtype=np.complex128)
        if np.allclose(summ, identity):
            return True
        else:
            return False


class DepolarisingChannel(QuantumNoise):
    '''Incoherent noise: time independent depolarising channel for n qubits
    '''
    def __init__(self, n: int, p: float):
        super().__init__(n)
        assert 0 <= p <= 1/3
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -complex(0, 1)], [complex(0, 1), 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        single_qubit = [(1-3*p, I), (p, X), (p, Y), (p, Z)]   # kraus for a single qubit
        self._combinations = []
        self._generate_combinations(0, [])
        self.kraus = []                                       # kraus for n qubits
        for combination in self._combinations:
            tensor = 1
            prob   = 1
            for i in combination:
                prob *= single_qubit[i][0]
                tensor = np.kron(tensor, single_qubit[i][1])
            self.kraus.append(np.sqrt(prob) * tensor)
        assert self.check_kraus() is True

    def _generate_combinations(self, i: int, stack: list):
        if len(stack) == self.n:
            self._combinations.append(stack.copy())
            return
        for j in range(4):   # {I, X, Y, Z}
            self._generate_combinations(i + 1, stack + [j])


class PhaseShiftNoise(QuantumNoise):
    '''Coherent noise: applying a phase shift noise on each of the n qubits
    '''
    def __init__(self, n: int, theta: float):
        super().__init__(n)
        assert 0 <= theta < 2*np.pi
        phaseshift = np.array([[1, 0], [0, np.exp(complex(0, theta))]])
        self.kraus = []
        tensor = 1
        for _ in range(self.n):
            tensor = np.kron(tensor, phaseshift)  # phase shift on all qubit/sensors
        self.kraus.append(tensor)
        assert self.check_kraus() is True



if __name__ == '__main__':
    num_sensor = 2
    depolar = DepolarisingChannel(num_sensor, 0.01)
    print(depolar)

    phaseshift = PhaseShiftNoise(num_sensor, np.pi/8)
    print(phaseshift)
