'''Some utility tools
'''

class Utility:

    EPSILON = 1e-8

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
    def print_matrix(matrix):
        '''print a matrix with complex values elegantly
        '''
        for row in matrix:
            for item in row:
                real = f'{item.real:.4f}'
                imag = f'{item.imag:.4f}'
                if imag[0] != '-':
                    imag = '+' + imag
                print(f'({real:>7}{imag:>7}i)', end=' ')
            print()
