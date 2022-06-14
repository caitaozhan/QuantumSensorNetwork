'''
Automatically generate euqations
'''

import sys
from collections import defaultdict, Counter

class EquationGenerator:
    def __init__(self, num_sensor):
        self.num_sensor = num_sensor
        self.z = {}      # raw z               -- key: (i, j), val: str
        self.z_e = {}    # z with eigenvalues  -- key: (i, j), val: dict, has the information of e_plus, e_minus, and one
        self.coeff_partition = defaultdict(list)  # partition the coefficients
        self.coeff_variable = {}   # coefficient to variable (reverse of self.coeff_partition)
        self.equations = {}

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

    def analyze_equations(self):
        '''count the left hand sides and right hand sides
        '''
        rhs_dict = Counter()
        for key in self.z_e:
            val = self.z_e[key]
            one = val['one']
            for coeff in one:
                rhs_dict[coeff] += 1
        for i in range(2**self.num_sensor):
            rhs = rhs_dict[i]
            lhs = self.num_sensor * (self.num_sensor - 1) // 2 - rhs
            self.coeff_partition[(lhs, rhs)].append(i)

    def rewrite_equations(self):
        '''using the coefficient partition. variables in a same partition assume equal
        '''
        for key in self.coeff_partition:
            coeffs = self.coeff_partition[key]
            for coeff in coeffs:
                self.coeff_variable[coeff] = key

        for key in self.z_e:
            e_plus = self.z_e[key]['e_plus']
            e_minus = self.z_e[key]['e_minus']
            one = self.z_e[key]['one']
            left = Counter()
            right = Counter()
            for coeff in e_plus:
                variable = self.coeff_variable[coeff]
                left[variable] += 1
            for coeff in e_minus:
                variable = self.coeff_variable[coeff]
                left[variable] += 1
            for coeff in one:
                variable = self.coeff_variable[coeff]
                right[variable] += 1
            self.equations[key] = {'LHS': left, 'RHS': right}


def theorem(n):
    from math import comb
    lst = []
    for k in range(2, n-1):
        a = comb(n-2, k-2) + comb(n-2, k)
        b = 2 * comb(n-2, k-1)
        lst.append(a/b)
    print(n, f'{min(lst):.3f}', end=' ')
    print('[', end=' ')
    for e in lst:
        print(f'{e:.3f}', end=',')
    print(']')


def main1(num_sensor):
    eg = EquationGenerator(num_sensor)
    eg.set_z()
    eg.set_equations()
    eg.analyze_equations()
    eg.rewrite_equations()

    # for key in eg.z:
    #     print(eg.z[key])
    #     print(eg.z_e[key])
    #     print(eg.equations[key])
    #     print()
    print(f'n={num_sensor}. The partitions:', end=' ')
    for key in eg.coeff_partition:
        print(key, end='  ')
    print()
    # print('The partitioned coefficients:')
    # for key in eg.coeff_partition:
    #     print(key, ':', eg.coeff_partition[key], f', length = {len(eg.coeff_partition[key])}')
    # print()


if __name__ == '__main__':
    # num_sensor = int(sys.argv[1])
    for n in range(4, 21):
        main1(n)

    # for n in range(4, 21):
    #     theorem(n)
