'''Some utility tools
'''
import numpy as np

class Utility:

    EPSILON = 1e-8
    EPSILON_SEMIDEFINITE = 8e-4 # relaxed for semidefinate programming optimal condition checking......

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
    def integer2bit(integer, num_of_bit):
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
        if maxx < Utility.EPSILON_SEMIDEFINITE:
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

