'''Modeling quantum noise
https://pennylane.ai/qml/demos/tutorial_noisy_circuits/
'''
import numpy as np


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
        self._single_qubit = []
        self._combinations = []

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

    def _generate_combinations(self, stack: list):
        '''length = len(single_qubit)
        '''
        if len(stack) == self.n:
            self._combinations.append(stack.copy())
            return
        for j in range(len(self._single_qubit)):
            self._generate_combinations(stack + [j])


class DepolarisingChannel(QuantumNoise):
    '''Incoherent noise: time independent depolarising channel for n qubits
    '''
    def __init__(self, n: int, p: float):
        super().__init__(n)
        assert 0 <= p <= 1
        self.p = p
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -complex(0, 1)], [complex(0, 1), 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self._single_qubit = [(1-p, I), (p/3, X), (p/3, Y), (p/3, Z)]   # kraus for a single qubit
        self._generate_combinations([])
        self.kraus = []                                           # kraus for n qubits
        for combination in self._combinations:
            tensor = 1
            prob   = 1
            for i in combination:
                prob *= self._single_qubit[i][0]
                tensor = np.kron(tensor, self._single_qubit[i][1])
            self.kraus.append(np.sqrt(prob) * tensor)
        assert self.check_kraus() is True


class AmplitudeDamping(QuantumNoise):
    '''Amplitude damping noise
    '''
    def __init__(self, n: int, gamma: float):
        super().__init__(n)
        assert 0 <= gamma <= 1
        self.gamma = gamma
        E0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]])
        E1 = np.array([[0, np.sqrt(self.gamma)], [0, 0]])
        self._single_qubit = [E0, E1]
        self._generate_combinations([])
        for combination in self._combinations:
            tensor = 1
            for i in combination:
                tensor = np.kron(tensor, self._single_qubit[i])
            self.kraus.append(tensor)
        assert self.check_kraus() is True


class PhaseDamping(QuantumNoise):
    '''Phase damping noise
    '''
    def __init__(self, n: int, gamma: float):
        super().__init__(n)
        assert 0 <= gamma <= 1
        self.gamma = gamma
        E0 = np.array([[1, 0], [0, np.sqrt(1 - self.gamma)]])
        E1 = np.array([[0, 0], [0, np.sqrt(self.gamma)]])
        self._single_qubit = [E0, E1]
        self._generate_combinations([])
        for combination in self._combinations:
            tensor = 1
            for i in combination:
                tensor = np.kron(tensor, self._single_qubit[i])
            self.kraus.append(tensor)
        assert self.check_kraus() is True


class RZNoise(QuantumNoise):
    '''Coherent noise: applying a phase shift noise on each of the n qubits
    '''
    def __init__(self, n: int, epsilon: float, std: float):
        '''
        Args:
            n -- number of sensors/qubits
            epsilon -- mean value of phase shift error epsilon
            std     -- the standard deviation of epsilon
        '''
        super().__init__(n)
        assert 0 <= epsilon < 2*np.pi
        self.epsilon = epsilon
        self.std = std
        self.kraus_mean()

    def kraus_mean(self):
        '''compute the kraus operators using only the mean value of epsilon
        '''
        rz = np.array([[np.exp(-complex(0, self.epsilon/2)), 0], [0, np.exp(complex(0, self.epsilon/2))]])
        self.kraus = []
        tensor = 1
        for _ in range(self.n):
            tensor = np.kron(tensor, rz)  # phase shift on all qubit/sensors
        self.kraus.append(tensor)
        assert self.check_kraus() is True

    def kruas_mean_std(self):
        '''compute the kruas operators using both the mean value and standard deviation of epsilon
        '''
        epsilon = np.random.normal(self.epsilon, self.std, 1)[0]
        rz = np.array([[np.exp(-complex(0, epsilon/2)), 0], [0, np.exp(complex(0, epsilon/2))]])
        self.kraus = []
        tensor = 1
        for _ in range(self.n):
            tensor = np.kron(tensor, rz)  # phase shift on all qubit/sensors
        self.kraus.append(tensor)
        assert self.check_kraus() is True



if __name__ == '__main__':
    num_sensor = 2
    # depolar = DepolarisingChannel(num_sensor, p=0.01)
    # print(depolar)

    amplitude_damp = AmplitudeDamping(num_sensor, gamma=0.01)
    print(amplitude_damp)

    phase_damp = PhaseDamping(num_sensor, gamma=0.01)
    print(phase_damp)

    # rz = RZNoise(num_sensor, np.pi/8, 0)
    # print(rz)
