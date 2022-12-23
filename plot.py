from distutils.log import Log
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from logger import Logger
from optimize_initial_state import OptimizeInitialState
from utility import Utility

class Plot:

    plt.rcParams['font.size'] = 65
    plt.rcParams['lines.linewidth'] = 5
    
    @staticmethod
    def hillclimbing(scores: list, constant: float = None):
        '''
        Args:
            scores -- a list of scores, each score is one evaluation value during an iteration
            constant -- value from a guess
        '''
        fig, ax = plt.subplots(1, 1, figsize=(35, 20))
        ax.plot(scores, label='Hill Climbing')
        if constant:
            constant = [constant for _ in range(len(scores))]
            ax.plot(constant, label='Guess')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discrimination Success Probability')
        # ax.set_ylim([0.7, 0.9])
        fig.legend(loc='upper center', ncol=2)
        fig.savefig('tmp.png')

    @staticmethod
    def hillclimbing(scores1: list, scores2: list):
        '''
        Args:
            scores1 -- a list of scores, each score is one evaluation value during an iteration
            scores2 -- a list of scores, each score is one evaluation value during an iteration
        '''
        fig, ax = plt.subplots(1, 1, figsize=(35, 20))
        ax.plot(scores1, label='Hill Climbing')
        ax.plot(scores2, label='Particle Swarm Optimization')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discrimination Success Probability')
        # ax.set_ylim([0.7, 0.9])
        fig.legend(loc='upper center', ncol=2)
        fig.savefig('tmp.png')

    @staticmethod
    def vary_priors(data, filename):
        table = []
        for myinput, output_by_method in data:
            tmp = [myinput.priors]
            for method, output in output_by_method.items():
                if method == 'Guess':
                    tmp.append(output.success)
                if method == 'Hill climbing':
                    tmp.append(output.scores)
            table.append(tmp)

        iteration = 50
        prior0  = table[0][0]
        guess0  = [table[0][1] for _ in range(iteration)]
        scores0 = table[0][2][:iteration]
        prior1  = table[1][0]
        guess1  = [table[1][1] for _ in range(iteration)]
        scores1 = table[1][2][:iteration]

        fig, ax = plt.subplots(1, 1, figsize=(35, 25))
        fig.subplots_adjust(left=0.1, right=0.96, top=0.85, bottom=0.1)
        ax.plot(guess0,  color='r', label=str(prior0))
        ax.plot(scores0, color='r')
        ax.plot(guess1,  color='deepskyblue', label=str(prior1))
        ax.plot(scores1, color='deepskyblue')
        ax.legend(ncol=1, bbox_to_anchor=(0.2, 0.99))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Discrimination Success Probability')
        fig.savefig(filename)


    @staticmethod
    def vary_numsensor(data, filename):
        table = []
        for myinput, output_by_method in data:
            tmp = [myinput.num_sensor]
            for method, output in output_by_method.items():
                if method == 'Guess':
                    tmp.append(output.success)
                if method == 'Hill climbing':
                    tmp.append(output.scores)
            table.append(tmp)

        iteration = 50
        numsen0 = table[0][0]
        guess0  = [table[0][1] for _ in range(iteration)]
        scores0 = table[0][2][:iteration]
        numsen1 = table[1][0]
        guess1  = [table[1][1] for _ in range(iteration)]
        scores1 = table[1][2][:iteration]
        numsen2 = table[2][0]
        guess2  = [table[2][1] for _ in range(iteration)]
        scores2 = table[2][2][:iteration]
        numsen3 = table[3][0]
        guess3  = [table[3][1] for _ in range(iteration)]
        scores3 = table[3][2][:iteration]

        fig, ax = plt.subplots(1, 1, figsize=(35, 25))
        fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
        ax.plot(guess0,  color='r', label=str(numsen0))
        ax.plot(scores0, color='r')
        ax.plot(guess1,  color='deepskyblue', label=str(numsen1))
        ax.plot(scores1, color='deepskyblue')
        ax.plot(guess2,  color='g', label=str(numsen2))
        ax.plot(scores2, color='g')
        ax.plot(guess3,  color='black', label=str(numsen3))
        ax.plot(scores3, color='black')

        ax.legend(ncol=4, bbox_to_anchor=(0.2, 0.99))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Discrimination Success Probability')
        fig.savefig(filename)


    @staticmethod
    def vary_startseed(data, filename):
        table = []
        for _, output_by_method in data:
            tmp = []
            for method, output in output_by_method.items():
                if method == 'Hill climbing':
                    tmp.append(output.start_seed)
                    tmp.append(output.scores)
            table.append(tmp)

        iteration = 50
        stseed0 = table[0][0]
        scores0 = table[0][1][:iteration]
        stseed1 = table[1][0]
        scores1 = table[1][1][:iteration]
        stseed2 = table[2][0]
        scores2 = table[2][1][:iteration]
        stseed3 = table[3][0]
        scores3 = table[3][1][:iteration]

        fig, ax = plt.subplots(1, 1, figsize=(35, 25))
        fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
        ax.plot(scores0,  color='r', label=str(stseed0))
        ax.plot(scores1,  color='deepskyblue', label=str(stseed1))
        ax.plot(scores2,  color='g', label=str(stseed2))
        ax.plot(scores3,  color='black', label=str(stseed3))

        ax.legend(ncol=4, bbox_to_anchor=(0.2, 0.99))
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Discrimination Success Probability')
        fig.savefig(filename)

def vary_priors():
    logs = ['result/4.2.2022/vary-prior']
    figname = 'result-tmp/vary-prior'
    data = Logger.read_log(logs)
    Plot.vary_priors(data, figname)

def vary_numsensors():
    logs = ['result/4.2.2022/vary-numsensor']
    filename = 'result-tmp/vary-numsensor'
    data = Logger.read_log(logs)
    Plot.vary_numsensor(data, filename)

def vary_startseed():
    logs = ['result/4.2.2022/vary-startseed']
    filename = 'result-tmp/vary-startseed'
    data = Logger.read_log(logs)
    Plot.vary_startseed(data, filename)


def n_sensor_analytical(N: int, theta: float):
    '''Equation 26 in https://arxiv.org/pdf/2210.17254.pdf
    '''
    RAD = 180 / np.pi
    term2 = 0.5 * (N - 2) * (1 - np.cos(2*theta/RAD))
    term3 = np.sqrt(N-1) * np.abs(np.sin(2*theta/RAD))
    return 1/N * (1 + term2 + term3)


# min error discrimination, 2 sensors, varying the theta of the unitary operator
def special_u_2sensor(draw: bool):
    '''
    Args:
        draw: if False, then don't plot, only return the plotting data
    '''

    logs = ['result/6.3.2022/varying_theta_2sensor_minerror']
    X = []
    y_guess = []
    y_hillclimb = []
    
    unsorted = []
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        unsorted.append([myinput.unitary_theta, output_by_method['Guess'].success, output_by_method['Hill climbing'].success])

    aftersort = sorted(unsorted)
    for x, guess, hill in aftersort:
        X.append(x)
        y_guess.append(guess)
        y_hillclimb.append(hill)

    if draw is False:
        return y_guess, y_hillclimb

    RAD = 180 / np.pi
    y_mark = []
    for x in X:
        y_mark.append(0.5 + 0.5 * np.sin(2*x/RAD))

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, y_guess, label='Guess, Evaluate by SDP')
    ax.plot(X, y_hillclimb, label='Hill climbing')
    ax.plot(X, y_mark, label='Mark Equation', linestyle=':')
    ax.legend(loc='lower center')
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.set_title('2 Sensors Initial State Opimization Problem')
    ax.set_ylim([0.3, 1.05])
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/6.3.2022/varying_theta_2sensors'
    fig.savefig(filename)
    return -1, -1


# min error discrimination, 3 sensors, varying the theta of the unitary operator
def special_u_3sensor(draw: bool):
    '''
    Args:
        draw: if False, then don't plot, only return the plotting data
    '''

    RAD = 180 / np.pi
    def theory(theta):
        alpha = np.sqrt(5 + 4*np.cos(2*theta))
        return 1./27 * (8*alpha*abs(np.sin(theta)) - 4*np.cos(2*theta) + 13)

    logs = ['result/4.6.2022/varying_theta']
    X = []
    y_guess = []
    y_hillclimb0 = []
    y_hillclimb1 = []
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        if output_by_method['Hill climbing'].start_seed == 0:
            X.append(myinput.unitary_theta)
            y_guess.append(output_by_method['Guess'].success)
            y_hillclimb0.append(output_by_method['Hill climbing'].success)
        if output_by_method['Hill climbing'].start_seed == 1:
            y_hillclimb1.append(output_by_method['Hill climbing'].success)
        
        if myinput.unitary_theta in [40, 50, 60, 70, 80, 90] and output_by_method['Hill climbing'].start_seed == 0:
        # if myinput.unitary_theta in [40, 50, 60, 70, 80, 90]:
            print('\ntheta =', myinput.unitary_theta)
            print(output_by_method['Guess'].init_state)
            print('guess success probability =', output_by_method['Guess'].success)
            print(output_by_method['Hill climbing'].init_state)
            print('hill climbing probability =', output_by_method['Hill climbing'].success)
            print('---')

    y_theory = []
    for x in X:
        y_theory.append(theory(x/RAD))

    y_hillclimb = [max(y0, y1) for y0, y1 in zip(y_hillclimb0, y_hillclimb1)]

    if draw is False:
        return y_guess, y_hillclimb

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, y_theory, label='Guess, Theoritical equation')
    ax.plot(X, y_guess, label='Guess, Evaluate by SDP')
    ax.plot(X, y_hillclimb, label='Hill climbing')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.set_title('3 Sensors Initial State Opimization Problem')
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/4.6.2022/varying_theta_3sensors'
    fig.savefig(filename)
    return -1, -1


# min error discrimination, 4 sensors, varying the theta of the unitary operator
def special_u_4sensor(draw: bool):
    '''
    Args:
        draw: if False, then don't plot, only return the plotting data
    '''

    logs = ['result/4.30.2022/varying_theta_4sensor_minerror.guess']
    y_guess = defaultdict(list)
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        y_guess[myinput.unitary_theta] = output_by_method['Guess'].success
    tmp = []
    X = []
    for theta, success in sorted(y_guess.items()):
        X.append(theta)
        tmp.append(success)
    y_guess = tmp

    logs = ['result/5.1.2022/varying_theta_4sensor_minerror.randomneighbor', \
            'result/4.30.2022/varying_theta_4sensor_minerror.randomneighbor']

    y_hillclimb0 = defaultdict(list)
    y_hillclimb1 = defaultdict(list)
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        
        if output_by_method['Hill climbing'].start_seed == 0:
            y_hillclimb0[(myinput.unitary_theta, 0)] = output_by_method['Hill climbing'].success
        if output_by_method['Hill climbing'].start_seed == 1:
            y_hillclimb1[(myinput.unitary_theta, 1)] = output_by_method['Hill climbing'].success

    y_hillclimb = []
    for theta in range(1, 180):
        y_hillclimb.append(max(y_hillclimb0[(theta, 0)], y_hillclimb1[(theta, 1)]))
    
    y_analitical = []
    for theta in range(1, 180):
        y_analitical.append(n_sensor_analytical(4, theta))

    if draw is False:
        return y_guess, y_hillclimb

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, y_guess, label='Guess, Evaluate by SDP')
    ax.plot(X, y_hillclimb, label='Hill climbing')
    ax.plot(X, y_analitical, label='Analytical', linestyle=':')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.set_title('4 Sensors Initial State Opimization Problem')
    # ax.set_ylim([0.3, 1.05])
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/5.1.2022/varying_theta_4sensors'
    fig.savefig(filename)
    return -1, -1


# min error discrimination, 5 sensors, varying the theta of the unitary operator
def special_u_5sensor(draw: bool):
    '''
    Args:
        draw: if False, then don't plot, only return the plotting data
    '''

    logs = ['result/5.3.2022/varying_theta_5sensor_minerror']
    y_guess = defaultdict(list)
    y_hillclimb = defaultdict(list)
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        y_guess[myinput.unitary_theta] = output_by_method['Guess'].success
        y_hillclimb[myinput.unitary_theta] = output_by_method['Hill climbing'].success
    guess = []
    hillclimb = []
    for theta in range(1, 91):
        guess.append(y_guess[theta])
        hillclimb.append(y_hillclimb[theta])
    y_guess = guess + guess[-2::-1]
    y_hillclimb = hillclimb + hillclimb[-2::-1]
    X = [i for i in range(1, 180)]

    if draw is False:
        return y_guess, y_hillclimb

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, y_guess, label='Guess, Evaluate by SDP')
    ax.plot(X, y_hillclimb, label='Hill climbing')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.set_title('5 Sensors Initial State Opimization Problem')
    # ax.set_ylim([0.3, 1.05])
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/5.3.2022/varying_theta_5sensors'
    fig.savefig(filename)


def special_u_allsensors(guess_2s, hillclimb_2s, guess_3s, hillclimb_3s, guess_4s, hillclimb_4s, guess_5s, hillclimb_5s):
    X = [i for i in range(1, 180)]
    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, guess_2s, label='2 Sensors')
    ax.plot(X, guess_3s, label='3 Sensors')
    ax.plot(X, guess_4s, label='4 Sensors')
    ax.plot(X, guess_5s, label='5 Sensors')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.set_title('Initial State Opimization Problem -- Guessing')
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/5.5.2022/varying_theta_allsensors_guess'
    fig.savefig(filename)

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, hillclimb_2s, label='2 Sensors')
    ax.plot(X, hillclimb_3s, label='3 Sensors')
    ax.plot(X, hillclimb_4s, label='4 Sensors')
    ax.plot(X, hillclimb_5s, label='5 Sensors')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.set_title('Initial State Opimization Problem -- Hill climbing')
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/5.5.2022/varying_theta_allsensors_hillclimb'
    fig.savefig(filename)

# unambiguous discrimination, 3 sensors, varying the theta of the unitary operator
def special_u_2():
    RAD = 180 / np.pi
    def theory_minerror(theta):
        alpha = np.sqrt(5 + 4*np.cos(2*theta))
        return 1./27 * (8*alpha*abs(np.sin(theta)) - 4*np.cos(2*theta) + 13)

    def theory_unambiguous(theta):
        return 1./3 * (4 * np.sin(theta)**2) / (8 * np.cos(theta)**2 + 1)

    logs = ['result/4.10.2022/varying_theta_unambiguous', 'result/4.11.2022/varying_theta_unambiguous',\
            'result/4.13.2022/varying_theta_unambiguous']
    X_y_guess = []
    guessdict = defaultdict(list)
    hillclimbdict = defaultdict(list)
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        if 'Guess' in output_by_method:
            guessdict[myinput.unitary_theta].append(output_by_method['Guess'].success)
        if 'Hill climbing' in output_by_method:
            hillclimbdict[myinput.unitary_theta].append(output_by_method['Hill climbing'].success)

    y_hillclimb_unambiguous = []
    for theta, suc_list in sorted(hillclimbdict.items()):
        y_hillclimb_unambiguous.append((theta, max(suc_list)))
    y_guess_unambiguous = []
    for theta, suc_list in sorted(guessdict.items()):
        y_guess_unambiguous.append((theta, max(suc_list)))

    if len(y_guess_unambiguous) != len(y_hillclimb_unambiguous):
        raise Exception('Guess and Hill climbing data length are not matching')

    X = []
    y_guess = []
    y_hillclimb = []
    for (theta1, success1), (theta2, success2) in zip(y_guess_unambiguous, y_hillclimb_unambiguous):
        if theta1 != theta2:
            raise Exception('theta1 != theta2')
        X.append(theta1)
        y_guess.append(success1)
        y_hillclimb.append(success2)
    # y_theory_minerror = []
    # y_theory_unambiguous = []
    # for x in X:
        # y_theory_minerror.append(theory_minerror(x/RAD))
        # y_theory_unambiguous.append(theory_unambiguous(x/RAD))
    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    # ax.plot(X, y_theory_minerror, label='Min Error. Theoretical')
    # ax.plot(X, y_theory_unambiguous, label='Unambiguous. Theoretical')
    ax.plot(X, y_guess, label='Unambiguous. Guess, evaluate by SDP')
    ax.plot(X, y_hillclimb, label='Unambiguous. Hill climbing')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/4.13.2022/varying_theta_unambiguous'
    fig.savefig(filename)


# the upperbound of the initial state generated by Guess and Hill climbing (min error)
def upperbound():
    RAD = 180 / np.pi
    def theory(theta):
        alpha = np.sqrt(5 + 4*np.cos(2*theta))
        return 1./27 * (8*alpha*abs(np.sin(theta)) - 4*np.cos(2*theta) + 13)

    logs = ['result/4.6.2022/varying_theta']
    num_sensor = 3
    unitary_seed = 2
    priors = [1./3] * 3
    X = []
    y_guess = []
    y_hillclimb = []
    y_hillclimb0 = []
    y_hillclimb1 = []
    y_hillclimb_success = []
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        if output_by_method['Hill climbing'].start_seed == 0:
            X.append(myinput.unitary_theta)
            y_guess.append(output_by_method['Guess'].init_state)
            y_hillclimb0.append((output_by_method['Hill climbing'].success, output_by_method['Hill climbing'].init_state))
        if output_by_method['Hill climbing'].start_seed == 1:
            y_hillclimb1.append((output_by_method['Hill climbing'].success, output_by_method['Hill climbing'].init_state))
    for i in range(len(y_hillclimb0)):
        if y_hillclimb0[i][0] > y_hillclimb1[i][0]:
            y_hillclimb.append(y_hillclimb0[i][1])    
            y_hillclimb_success.append(y_hillclimb0[i][0])
        else:
            y_hillclimb.append(y_hillclimb1[i][1])
            y_hillclimb_success.append(y_hillclimb1[i][0])
    
    y_guess_ub = []
    y_guess_ubnew = []
    for unitary_theta, init_state_str in zip(X, y_guess):
        opt_initstate = OptimizeInitialState(num_sensor)
        opt_initstate.set_statevector_from_str(init_state_str)
        unitary_operator = Utility.generate_unitary_operator(theta=unitary_theta, seed=unitary_seed)
        y_guess_ub.append(opt_initstate.upperbound(unitary_operator, priors))
        y_guess_ubnew.append(opt_initstate.upperbound_new(unitary_operator, priors))
    y_hillclimb_ub = []
    y_hillclimb_ubnew = []
    for unitary_theta, init_state in zip(X, y_hillclimb):
        opt_initstate = OptimizeInitialState(num_sensor)
        opt_initstate.set_statevector_from_str(init_state)
        unitary_operator = Utility.generate_unitary_operator(theta=unitary_theta, seed=unitary_seed)
        y_hillclimb_ub.append(opt_initstate.upperbound(unitary_operator, priors))
        y_hillclimb_ubnew.append(opt_initstate.upperbound_new(unitary_operator, priors))
    
    y_theory = []
    for x in X:
        y_theory.append(theory(x/RAD))

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, y_theory, label='Guess')
    ax.plot(X, y_guess_ub, label='Guess, Upper bound')
    ax.plot(X, y_guess_ubnew, label='Guess, Upper bound new')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/7.1.2022/varying_theta_guess_upperbound'
    fig.savefig(filename)

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, y_hillclimb_success, label='Hill climbing')
    ax.plot(X, y_hillclimb_ub, label='Hill climbing, Upper bound')
    ax.plot(X, y_hillclimb_ubnew, label='Hill climbing, Upper bound new')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/7.1.2022/varying_theta_hillclimbing_upperbound'
    fig.savefig(filename)


def print_results():
    # logs = ['result/4.10.2022/varying_theta_unambiguous']
    logs = ['result/4.29.2022/varying_theta_3sensor_minerror.randomneighbor']
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        if myinput.unitary_theta != 46:
            continue
        output_by_method = experiment[1]
        if 'Hill climbing' in output_by_method:
            print(output_by_method['Hill climbing'].init_state)
            print('Hill climbing success probability =', output_by_method['Hill climbing'].success)

    logs = ['result/4.29.2022/varying_theta_3sensor_minerror.realimag.100iter']
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        if myinput.unitary_theta != 46:
            continue
        output_by_method = experiment[1]
        if 'Hill climbing' in output_by_method:
            print('\ntheta =', myinput.unitary_theta)
            print(output_by_method['Hill climbing'].init_state)
            eval_metric = output_by_method['Hill climbing'].eval_metric
            print(f'Hill climbing ({eval_metric}) probability =', output_by_method['Hill climbing'].success)
            print('---')


# comparing simulated annealing and hill climbing
def simulatedanneal_hillclimb_compare():
    # logs = ['result/4.27.2022/varying_theta_3sensor_minerror', 'result/4.28.2022/varying_theta_3sensor_minerror']
    logs = ['result/5.4.2022/simulatedanneal_hillclimb']
    data = Logger.read_log(logs)
    simulated = {}   # (theta, start_seed) --> success
    hillclimb = {}
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Hill climbing' in output_by_method:
            start_seed = output_by_method['Hill climbing'].start_seed
            success = output_by_method['Hill climbing'].success
            hillclimb[(theta, start_seed)] = success
        if 'Simulated annealing' in output_by_method:
            start_seed = output_by_method['Simulated annealing'].start_seed
            success = output_by_method['Simulated annealing'].success
            simulated[(theta, start_seed)] = success

    X = []
    sa_minus_hc = []
    sa_minus_hc0 = []
    sa_minus_hc1 = []
    zeros = []
    for theta in range(1, 90, 5):
        X.append(theta)
        zeros.append(0)
        for start_seed in [0, 1]:
            simulated_success = simulated[(theta, start_seed)]
            hillclimb_success = hillclimb[(theta, start_seed)]
            sa_minus_hc.append(simulated_success - hillclimb_success)
            if start_seed == 0:
                sa_minus_hc0.append(simulated_success - hillclimb_success)
            if start_seed == 1:
                sa_minus_hc1.append(simulated_success - hillclimb_success)

    print('Start seed both 0 and 1')
    print(f'avg. = {np.average(sa_minus_hc)}')
    print(f'std. = {np.std(sa_minus_hc)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(sa_minus_hc)
    ax.plot(zeros*2)
    ax.set_ylim([-0.0001, 0.0001])
    ax.set_ylabel('Success probability')
    ax.set_title('Simulated Anneal - Hill Climb')
    fig.savefig('result/5.4.2022/sa-hc-seed0-1')

    print('Start seed 0')
    print(f'avg. = {np.average(sa_minus_hc0)}')
    print(f'std. = {np.std(sa_minus_hc0)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, sa_minus_hc0)
    ax.plot(X, zeros)
    ax.set_ylim([-0.0001, 0.0001])
    ax.set_ylabel('Success probability')
    ax.set_title('Simulated Anneal - Hill Climb')
    fig.savefig('result/5.4.2022/sa-hc-seed0.png')

    print('Start seed 1')
    print(f'avg. = {np.average(sa_minus_hc1)}')
    print(f'std. = {np.std(sa_minus_hc1)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, sa_minus_hc1)
    ax.plot(X, zeros)
    ax.set_ylim([-0.0001, 0.0001])
    ax.set_ylabel('Success probability')
    ax.set_title('Simulated Anneal - Hill Climb')
    fig.savefig('result/5.4.2022/sa-hc-seed1.png')


# 
def hillclimb_bugfix():
    logs = ['result/4.6.2022/varying_theta']
    data = Logger.read_log(logs)
    hillclimb = {}       # (theta, start_seed) --> success
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Hill climbing' in output_by_method:
            start_seed = output_by_method['Hill climbing'].start_seed
            success = output_by_method['Hill climbing'].success
            hillclimb[(theta, start_seed)] = success

    logs = ['result/4.28.2022/varying_theta_3sensor_minerror.bugfix']
    data = Logger.read_log(logs)
    hillclimb_bugfix = {}
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Hill climbing' in output_by_method:
            start_seed = output_by_method['Hill climbing'].start_seed
            success = output_by_method['Hill climbing'].success
            hillclimb_bugfix[(theta, start_seed)] = success

    hillclimbbugfix = []
    hillclimbbugfix0 = []
    hillclimbbugfix1 = []
    zeros = []
    for theta in range(1, 90):
        zeros.append(0)
        for start_seed in [0, 1]:
            hillclimbbugfix_success = hillclimb_bugfix[(theta, start_seed)]
            hillclimb_success = hillclimb[(theta, start_seed)]
            hillclimbbugfix.append(hillclimbbugfix_success - hillclimb_success)
            if start_seed == 0:
                hillclimbbugfix0.append(hillclimbbugfix_success - hillclimb_success)
            if start_seed == 1:
                hillclimbbugfix1.append(hillclimbbugfix_success - hillclimb_success)

    print('Start seed both 0 and 1')
    print(f'avg. = {np.average(hillclimbbugfix)}')
    print(f'std. = {np.std(hillclimbbugfix)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.1)
    ax.plot(hillclimbbugfix)
    ax.plot(zeros*2)
    ax.set_ylim([-0.005, 0.025])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climbing Bug Fix')
    fig.savefig('result/4.28.2022/hillclimbbugfix-seed0-1.bugfix.png')

    print('Start seed 0')
    print(f'avg. = {np.average(hillclimbbugfix0)}')
    print(f'std. = {np.std(hillclimbbugfix0)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.9, bottom=0.1)
    ax.plot(hillclimbbugfix0)
    ax.plot(zeros)
    ax.set_ylim([-0.005, 0.01])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climbing Bug Fix')
    fig.savefig('result/4.28.2022/hillclimbbugfix-seed0.bugfix.png')

    print('Start seed 1')
    print(f'avg. = {np.average(hillclimbbugfix1)}')
    print(f'std. = {np.std(hillclimbbugfix1)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.9, bottom=0.1)
    ax.plot(hillclimbbugfix1)
    ax.plot(zeros)
    ax.set_ylim([-0.005, 0.01])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climbing Bug Fix')
    fig.savefig('result/4.28.2022/hillclimbbugfix-seed1.bugfix.png')


# comparing hill climbing and guess
def hillclimb_guess_compare():
    logs = ['result/4.6.2022/varying_theta']
    data = Logger.read_log(logs)
    guess = {}       # (theta, start_seed) --> success
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Guess' in output_by_method:
            success = output_by_method['Guess'].success
            guess[(theta, 0)] = success
            guess[(theta, 1)] = success

    logs = ['result/4.28.2022/varying_theta_3sensor_minerror.bugfix']
    data = Logger.read_log(logs)
    hillclimb_bugfix = {}
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Hill climbing' in output_by_method:
            start_seed = output_by_method['Hill climbing'].start_seed
            success = output_by_method['Hill climbing'].success
            hillclimb_bugfix[(theta, start_seed)] = success

    hillclimb_minus_guess = []
    hillclimb_minus_guess0 = []
    hillclimb_minus_guess1 = []
    zeros = []
    for theta in range(1, 61):
        zeros.append(0)
        for start_seed in [0, 1]:
            hillclimbbugfix_success = hillclimb_bugfix[(theta, start_seed)]
            hillclimb_success = guess[(theta, start_seed)]
            hillclimb_minus_guess.append(hillclimbbugfix_success - hillclimb_success)
            if start_seed == 0:
                hillclimb_minus_guess0.append(hillclimbbugfix_success - hillclimb_success)
            if start_seed == 1:
                hillclimb_minus_guess1.append(hillclimbbugfix_success - hillclimb_success)

    print('Start seed both 0 and 1')
    print(f'avg. = {np.average(hillclimb_minus_guess)}')
    print(f'std. = {np.std(hillclimb_minus_guess)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(hillclimb_minus_guess)
    ax.plot(zeros*2)
    ax.set_ylim([-0.0002, 0.0002])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climbing Minus Guess')
    fig.savefig('result/4.28.2022/hillclimb-guess-seed0-1.bugfix.png')

    print('Start seed 0')
    print(f'avg. = {np.average(hillclimb_minus_guess0)}')
    print(f'std. = {np.std(hillclimb_minus_guess0)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(hillclimb_minus_guess0)
    ax.plot(zeros)
    ax.set_ylim([-0.0002, 0.0002])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climbing Minus Guess')
    fig.savefig('result/4.28.2022/hillclimb-guess-seed0.bugfix.png')

    print('Start seed 1')
    print(f'avg. = {np.average(hillclimb_minus_guess1)}')
    print(f'std. = {np.std(hillclimb_minus_guess1)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(hillclimb_minus_guess1)
    ax.plot(zeros)
    ax.set_ylim([-0.0002, 0.0002])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climbing Minus Guess')
    fig.savefig('result/4.28.2022/hillclimb-guess-seed1.bugfix.png')


# shows the iterations of simulated annealing and hill climbing
def simulatedanneal_hillclimb_iterations():
    # logs = ['result/4.28.2022/varying_theta_3sensor_minerror.bugfix']
    logs = ['result/5.4.2022/simulatedanneal_hillclimb']
    data = Logger.read_log(logs)
    simulated = {}   # (theta, start_seed) --> scores
    hillclimb = {}
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Hill climbing' in output_by_method:
            start_seed = output_by_method['Hill climbing'].start_seed
            scores = output_by_method['Hill climbing'].scores
            hillclimb[(theta, start_seed)] = scores
        if 'Simulated annealing' in output_by_method:
            start_seed = output_by_method['Simulated annealing'].start_seed
            scores = output_by_method['Simulated annealing'].scores
            simulated[(theta, start_seed)] = scores

    theta = 37
    start_seed = 0
    simulated0_score = simulated[(theta, start_seed)]
    hillclimb0_score = hillclimb[(theta, start_seed)]
    start_seed = 1
    # simulated1_score = simulated[(theta, start_seed)]
    # hillclimb1_score = hillclimb[(theta, start_seed)]

    print(f'simulated seed 0 max = {max(simulated0_score)} at {simulated0_score.index(max(simulated0_score))}')
    # print(f'simulated seed 1 max = {max(simulated1_score)} at {simulated1_score.index(max(simulated1_score))}')

    # print('Start seed both 0 and 1')
    # fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    # fig.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.1)
    # ax.plot(simulated0_score, label='Seed 0')
    # ax.plot(simulated1_score, label='Seed 1')
    # ax.set_ylabel('Success probability')
    # ax.set_title('Simulated Seed 0 and 1')
    # ax.legend()
    # fig.savefig('result/4.28.2022/simulated-iterations-seed0-1.png')

    print('Start seed 0')
    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.1)
    ax.plot(simulated0_score[:51], label='Simulated Annealing')
    ax.plot(hillclimb0_score[:51], label='Hill Climbing')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Success probability')
    ax.set_title('Simulated annealing VS. Hill Climbing')
    ax.legend()
    fig.savefig('result/5.4.2022/simulatedanneal_hillclimb')

    # print('Start seed 1')
    # fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    # fig.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.1)
    # ax.plot(hillclimb1_score, label='Hill Climb')
    # ax.plot(simulated1_score, label='Simulated Anneal')
    # ax.set_ylabel('Success probability')
    # ax.set_title('Simulated & Hill Climb Seed 1')
    # ax.legend()
    # fig.savefig('result/4.28.2022/simulated-vs-hillclimb-iterations-seed1.png')


# comparing different neighbor finding strategies in hill climbing
def hillclimb_neighbor_compare():
    logs = ['result/4.30.2022/varying_theta_4sensor_minerror.randomneighbor']
    data = Logger.read_log(logs)
    hillclimb_randomneighbor = {}       # (theta, start_seed) --> success
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Hill climbing' in output_by_method:
            start_seed = output_by_method['Hill climbing'].start_seed
            success = output_by_method['Hill climbing'].success
            hillclimb_randomneighbor[(theta, start_seed)] = success

    logs = ['result/4.30.2022/varying_theta_4sensor_minerror.realimag']
    data = Logger.read_log(logs)
    hillclimb_realimag = {}
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        theta = int(myinput.unitary_theta)
        if 'Hill climbing' in output_by_method:
            start_seed = output_by_method['Hill climbing'].start_seed
            success = output_by_method['Hill climbing'].success
            hillclimb_realimag[(theta, start_seed)] = success

    polar_minus_realimag_0 = []
    polar_minus_realimag_1 = []
    zeros = []
    for theta in range(1, 180, 3):
        zeros.append(0)
        for start_seed in [0, 1]:
            hillclimb_randomneighbor_success = hillclimb_randomneighbor[(theta, start_seed)]
            hillclimb_realimag_success = hillclimb_realimag[(theta, start_seed)]
            if start_seed == 0:
                polar_minus_realimag_0.append(hillclimb_randomneighbor_success - hillclimb_realimag_success)
            if start_seed == 1:
                polar_minus_realimag_1.append(hillclimb_randomneighbor_success - hillclimb_realimag_success)

    print('Start seed 0')
    print(f'avg. = {np.average(polar_minus_realimag_0)}')
    print(f'std. = {np.std(polar_minus_realimag_0)}')
    counter = 0
    for item in polar_minus_realimag_0:
        if item > 0:
            counter += 1
    print(f'# cases when randomneighbor is better than realimag = {counter}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(polar_minus_realimag_0)
    ax.plot(zeros)
    ax.set_ylim([-0.0003, 0.0003])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climb Randomneighbor Minus Realimag')
    fig.savefig('result/4.30.2022/hillclimb-randomneighbor-realimag-seed0.png')

    print('Start seed 1')
    print(f'avg. = {np.average(polar_minus_realimag_1)}')
    print(f'std. = {np.std(polar_minus_realimag_1)}')
    counter = 0
    for item in polar_minus_realimag_1:
        if item > 0:
            counter += 1
    print(f'# cases when randomneighbor is better than realimag = {counter}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.9, bottom=0.1)
    ax.plot(polar_minus_realimag_1)
    ax.plot(zeros)
    ax.set_ylim([-0.0003, 0.0003])
    ax.set_ylabel('Success probability')
    ax.set_title('Hill Climbing Randomneighbor Minus Realimag')
    fig.savefig('result/4.30.2022/hillclimb-randomneighbor-realimag-seed1.png')


# vary startseed: does the landspace has many hills with same height?
def varystartseed():
    logs = ['result/5.3.2022/varying_startseed_3sensor_minerror']
    figname = 'result/5.3.2022/3sensor_hillclimbing_iterations'
    data = Logger.read_log(logs)
    successes = []
    scores = []
    for experiment in data:
        # myinput = experiment[0]
        output_by_method = experiment[1]
        print(output_by_method['Hill climbing'].init_state)
        successes.append(output_by_method['Hill climbing'].success)
        scores.append(output_by_method['Hill climbing'].scores)
    print(f'mean = {np.mean(successes)}')
    print(f'std. = {np.std(successes)}')
    fig, ax = plt.subplots(1, 1, figsize=(30, 20))
    for i in range(10):
        ax.plot(scores[i], label=f'start seed={i}')
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Success Probability')
    ax.set_title('The Landscape has Many Hills with the SAME Height')
    fig.savefig(figname)

def nonentangled_3sensors(guess_3s, hillclimb_3s):
    logs = ['result/5.16.2022/varying_theta_3sensor_ne']
    figname = 'result/5.16.2022/varying_theta_3sensor_ne'
    data = Logger.read_log(logs)
    success_dict = {}
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        if output_by_method['Hill climbing'].method == 'Hill climbing (NE)' and output_by_method['Hill climbing'].start_seed == 0:
            success_dict[myinput.unitary_theta] = output_by_method['Hill climbing'].success
    
    successes = [success for _, success in sorted(success_dict.items())]
    X = [i for i in range(1, 180)]

    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, guess_3s, label='Guess')
    ax.plot(X, hillclimb_3s, label='Hill climbing')
    ax.plot(X, successes, label='Hill climbing (Non-entangled)')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.set_title('3 Sensors Initial State Opimization Problem')
    # ax.set_ylim([0.3, 1.05])
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/5.16.2022/varying_theta_3sensors_nonentangled'
    fig.savefig(filename)


if __name__ == '__main__':
    # scores = [0.789991, 0.832989, 0.845341, 0.852194, 0.857062, 0.859864, 0.860891, 0.860928, 0.861471, 0.861524, 0.861533, 0.861536, 0.861537, 0.861541, 0.861543, 0.861763, 0.861993, 0.862333, 0.862427, 0.86244, 0.862667, 0.862849, 0.862851, 0.862853, 0.862854, 0.862854, 0.862855, 0.862856, 0.862856, 0.862857, 0.862857, 0.862857, 0.862858, 0.862858, 0.862858, 0.862858, 0.862858, 0.862858, 0.86286, 0.862944, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862955, 0.862989, 0.863003, 0.863005]
    # constant = 0.794494
    # Plot.hillclimbing(scores, constant)

    # vary_priors()
    # vary_numsensors()
    # vary_startseed()

    draw = True
    # guess_2s, hillclimb_2s = special_u_2sensor(draw)
    # guess_3s, hillclimb_3s = special_u_3sensor(draw)
    guess_4s, hillclimb_4s = special_u_4sensor(draw)
    # guess_5s, hillclimb_5s = special_u_5sensor(draw)
    # special_u_allsensors(guess_2s, hillclimb_2s, guess_3s, hillclimb_3s, guess_4s, hillclimb_4s, guess_5s, hillclimb_5s)
    # special_u_2()
    # nonentangled_3sensors(guess_3s, hillclimb_3s)

    # print_results()

    # upperbound()


    # hillclimb_bugfix()

    # hillclimb_guess_compare()

    # simulatedanneal_hillclimb_compare()
    # simulatedanneal_hillclimb_iterations()

    # hillclimb_neighbor_compare()

    # varystartseed()

    
