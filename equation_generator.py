'''
Automatically generate euqations
'''

import sys

class EquationGenerator:
    def __init__(self, num_sensor):
        self.num_sensor = num_sensor
        self.z = {}    # raw z
        self.z_e = {}  # z with eigenvalues

    def set_z_helper(self, i: int, symbol_i: str, j: int, symbol_j: int) -> str:
        '''generate strings such as "U* (x) U (x) I"
        '''
        lst = ['I']*self.num_sensor
        lst[i] = symbol_i
        lst[j] = symbol_j
        return " x ".join(lst)

    def set_z(self):
        '''set all the z, which is <phi_1|phi_2>
        '''
        for i in range(self.num_sensor):
            for j in range(i + 1, self.num_sensor):
                tmp = self.set_z_helper(i, 'U*', j, 'U')
                tmp = f'<psi| {tmp} |psi>'
                tmp = f'<phi_{i}|phi_{j}> = ' + tmp
                self.z[(i, j)] = tmp

    def set_equations(self):
        '''set all the equations
        '''
        for key in self.z:
            e_plus = []
            e_minus = []
            one = []
            ustar_index = key[0]
            u_index     = key[1]
            for i in range(2**self.num_sensor):
                binary = bin(i)[2:]
                if len(binary) < self.num_sensor:
                    binary = '0'*(self.num_sensor - len(binary)) + binary
                ustar  = binary[ustar_index]
                u      = binary[u_index]
                if ustar == u:
                    one.append(i)
                if ustar == '0' and u == '1':
                    e_plus.append(i)
                if ustar == '1' and u == '0':
                    e_minus.append(i)
            self.z_e[key] = {'e_plus': e_plus, 'e_minus': e_minus, 'one': one}


if __name__ == '__main__':
    num_sensor = int(sys.argv[1])
    eg = EquationGenerator(num_sensor)
    eg.set_z()
    eg.set_equations()
    for key in eg.z:
        print(eg.z[key])
        print(eg.z_e[key])
        print()
