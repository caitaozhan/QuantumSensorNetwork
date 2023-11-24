import numpy as np


class DepolarisingNoise:
    '''Time independent depolarising noise
    '''
    def __init__(self, p: float):
        self.p = p            # the probability of happening X, Y, or Z
        self.I = np.eye(2)    # the probability of I is 1 - 3*p
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -complex(0, 1)], [complex(0, 1), 0]])
        self.Z = np.array([[1, 0], [0, -1]])

    def get_matrix(self, num_sen: int) -> np.array:
        '''get the matrix representation of the depolarsing noise
        Args:
            num_sen: the number of sensors/qubits
        '''
        rho = (1 - 3*self.p) * self.I + self.p * self.X + self.p * self.Y + self.p * self.Z
        tensor = 1
        for _ in range(num_sen):
            tensor = np.kron(tensor, rho)
        return tensor


if __name__ == '__main__':
    depolar = DepolarisingNoise(0.01)
    print(depolar.get_matrix(1))
    print(depolar.get_matrix(2))
