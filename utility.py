'''Some utility tools
'''

import numpy as np
from qiskit.quantum_info.operators.operator import Operator
from input_output import Default

class Utility:

    @staticmethod
    def norm_squared(alpha):
        '''
        Args:
            alpha -- complex number -- the amplitude for an element in the state vector
        Return:
            float -- the norm squared of a complex number, i.e. alpha * alpha.complex_conjugate
        '''
        return abs(alpha)**2

    @staticmethod
    def integer2bit(integer: int , num_of_bit: int):
        '''transform an integer to its binary form
           if integer = 3 and num_of_bit = 3, return 011
        '''
        bit = bin(integer)    # '0b11'
        bit = bit[2:]
        zeros = '0' * (num_of_bit - len(bit))
        return f'{zeros}{bit}'

    @staticmethod
    def print_matrix(describe: str, matrix):
        '''print a matrix with complex values elegantly
        '''
        print(describe)
        for row in matrix:
            for item in row:
                real = f'{item.real:.4f}'
                imag = f'{item.imag:.4f}'
                if imag[0] != '-':
                    imag = '+' + imag
                print(f'({real:>7}{imag:>7}i)', end=' ')
            print()

    @staticmethod
    def check_zero(matrix):
        '''check if a matrix contains all zero entries
        Args:
            matrix -- np.array -- the matrix to be checked
        Return:
            bool -- True if all elements in the matrix are zero, False otherwise
        '''
        matrix = np.abs(matrix)
        maxx = np.max(matrix)
        if maxx < Default.EPSILON_SEMIDEFINITE:
            return True
        else:
            return False

    @staticmethod
    def check_optimal(quantum_states: list, priors: list, povms: list):
        '''check the optimality for minimum error povm
        Args:
            quantum_states -- a list of QuantumState objects
            priors         -- a list of prior probabilities
            povms          -- a list of Operator objects
        Return:
            bool -- True if povms are, False otherwise
        '''
        if not (len(quantum_states) == len(priors) == len(povms)):
            raise Exception('error in input, the input parameters do not have equal length')
        
        length = len(quantum_states)
        for i in range(length):
            for j in range(i+1, length):
                Pii  = povms[i].data
                Pij  = povms[j].data
                pi   = priors[i]
                pj   = priors[j]
                rhoi = quantum_states[i].density_matrix
                rhoj = quantum_states[j].density_matrix
                product = np.dot(Pii, np.dot(pi*rhoi - pj*rhoj, Pij))
                if Utility.check_zero(product) == False:
                    return False
        return True

    @staticmethod
    def get_theta(real: float, imag: float):
        '''return the theta in radian, want it between [0, 2*pi]
        '''
        theta = np.arctan(imag / real)  # theta between [-pi/2, pi/2]
        if real > 0 and imag > 0:       # first quadrant
            return theta
        elif real < 0 and imag > 0:     # second quadrant
            return np.pi + theta
        elif real < 0 and imag < 0:     # third quadrant
            return np.pi + theta
        else: # real > 0 and imag < 0:  # fourth quadrant
            return 2*np.pi + theta

    @staticmethod
    def evolve_operator(unitary_operator: Operator, num_sensor: int, i: int):
        '''Generate I \otimes U \otimes I
           In the above example, num_sensor = 3 and i = 1
        Return:
            Operator
        '''
        identity = np.eye(2)
        tensor = 1
        for j in range(num_sensor):
            if j == i:
                tensor = np.kron(tensor, unitary_operator._data)
            else:
                tensor = np.kron(tensor, identity)
        return Operator(tensor)

    @staticmethod
    def generate_priors(num_sensor: int, equal: bool):
        '''if equal, then generate equal priors for num_sensor number of sensors
        Return:
            array-like object
        '''
        if equal:
            return [1./num_sensor]*num_sensor
        else:
            np.random.seed(num_sensor)
            p = []
            for _ in range(num_sensor):
                p.append(np.random.rand())
            p = np.array(p)
            return p / np.sum(p)

    @staticmethod
    def generate_unitary_operator(theta: float, seed: int):
        '''
        Args:
            theta -- the angle (in degree) for the symmetric eigen values (from eigen value decomposition)
            unitary_seed -- control generating the random matrix
        '''
        from qiskit.quantum_info import random_unitary
        RAD = 180 / np.pi
        theta = theta / RAD
        e_val1 = complex(np.cos(theta), np.sin(theta))
        e_val2 = complex(np.cos(theta), -np.sin(theta))
        Lambda = np.array([[e_val1, 0], [0, e_val2]])
        Q = random_unitary(2, seed=seed)._data
        Q_inv = np.linalg.inv(Q)                # a random matrix is invertable almost for sure
        U = Operator(np.dot(Q, np.dot(Lambda, Q_inv)))
        if U.is_unitary():
            return U
        else:
            raise Exception('Failed to generate an unitary matrix')

