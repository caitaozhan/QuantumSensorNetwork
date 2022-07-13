'''
Automatically generate euqations
'''

from math import comb, ceil, floor
import numpy as np
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

    def optimal_solution(self):
        '''return the coefficients for orthogonal situation
           This version do the mergings of two partition that has the same minimal RHS/LHS
        Return:
            (float, float, float, list)
        '''
        n = self.num_sensor
        if n % 2 == 0:
            c = comb(n, n//2)          # c is total number of x (x is on the partition with the smallest RHS/LHS)
            partition = self.get_partition(n//2)
        else:
            c = comb(n, ceil(n/2)) + comb(n, floor(n/2))
            partition = self.get_partition(ceil(n/2)) + self.get_partition(floor(n/2))
        partition.sort()
        RHS = ceil(n/2) - 1
        LHS = ceil(n/2)
        a = c * (LHS/(LHS + RHS))      # a is the number of x on the left hand side
        b = c * (RHS/(LHS + RHS))      # b is the number of x on the right hand side
        
        print(f'n = {self.num_sensor}, total x = {c}, LHS = {int(a)}, RHS = {int(b)}, parition = {1}')
        return a, b, c, partition


    def optimal_solution_nomerge(self):
        '''return the coefficients for orthogonal situation
           This version do not do the merging comparing
        Return:
            (float, float, float, list)
        '''
        n = self.num_sensor
        c = comb(n, floor(n/2))        # c is total number of x (x is on the partition with the smallest RHS/LHS)
        partition = self.get_partition(floor(n/2)) # the middle partition
        partition.sort()
        RHS = ceil(n/2) - 1
        LHS = ceil(n/2)
        a = c * (LHS/(LHS + RHS))      # a is the number of x on the left hand side
        b = c * (RHS/(LHS + RHS))      # b is the number of x on the right hand side
        
        # print(f'n = {self.num_sensor}, total x = {c}, LHS = {int(a)}, RHS = {int(b)}, parition = {1}')
        return a, b, c, partition

    def optimal_solution_smallerT_i(self, theta: float, i: int):
        '''Gives the |Sl|, LHS, RHS, and partition for the i_th partition
        Args:
            theta -- the theta (in degrees) of the unitary operator
            i     -- the i_th partition
        Return:
            list
        '''
        n = self.num_sensor
        c = comb(n, i)
        partition = self.get_partition(i)
        partition.sort()
        RHS = comb(i, 2) + comb(n-i, 2)
        LHS = comb(i, 1) * comb(n-i, 1)
        a = c * (LHS/(LHS + RHS))   # L_l
        b = c * (RHS/(LHS + RHS))   # R_l
        RAD = 180 / np.pi
        value = (b + np.cos(2*theta/RAD)*a)/c
        print(f'theta={theta}, i={i}, (cos(2theta)*L_l+R_l)/size={value:0.3f} -- L_l={int(a)}, R_l={int(b)}, cos(2theta)={np.cos(2*theta/RAD):.3f}')
        return partition

    def optimal_solution_smallerT(self):
        '''Always pick the middle partition
        Return:
            list
        '''
        n = self.num_sensor
        i = floor(n/2)
        partition = self.get_partition(i)
        partition.sort()
        return partition


    def get_partition(self, ones: int) -> list:
        '''get the coefficients (binary format) that has ones number of 1
        Args:
            ones -- number of 1
        Return:
            list of coefficients in the paritition that has ones number of 1
            eg: ['001', '010', '100']
        '''
        def dfs(stack, i, counter):
            if i == self.num_sensor:
                if counter == ones:
                    ans.append(''.join(stack))
                return
            
            dfs(stack + ['0'], i+1, counter)
            dfs(stack + ['1'], i+1, counter+1)

        ans = []
        dfs(stack=[], i=0, counter=0)
        return ans



def theorem(n):
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


def theorem2(n):
    lst = []
    for i in range(1, n):
        a = comb(i, 2) + comb(n-i, 2)
        b = i * (n-i)
        lst.append(a/b)
    print(n, f'{min(lst):.3f}', end=' ')
    print('[', end=' ')
    for e in lst:
        print(f'{e:.3f}', end=',')
    print(']')

def main1(num_sensor):
    '''list all the equations
    '''
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

    for key in eg.z:
        print(eg.equations[key])
        print()
        break

def main2(n):
    '''return the optimal solution for [T, 180-T]
    '''
    eg = EquationGenerator(num_sensor=n)
    # eg.optimal_solution()
    eg.optimal_solution_nomerge()

def main3(n):
    '''return the optimal solution for [0, T]
    '''
    eg = EquationGenerator(n)
    for theta in range(1, 71, 2):
        for i in range(n+1):
            eg.optimal_solution_smallerT_i(theta, i)
        print('---')


if __name__ == '__main__':
    # num_sensor = int(sys.argv[1])
    for n in range(10,11):
        main3(n)
        # print('--------')

    # for n in range(3, 30):
    #     theorem2(n)
