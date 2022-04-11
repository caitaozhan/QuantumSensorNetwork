import matplotlib.pyplot as plt
from logger import Logger
import numpy as np


class Plot:

    plt.rcParams['font.size'] = 65
    plt.rcParams['lines.linewidth'] = 8
    
    @staticmethod
    def hillclimbing(scores: list, constant: float = None):
        '''
        Args:
            scores -- a list of scores, each score is one evaluation value during an iteration
            constant -- value from a guess
        '''
        constant = [constant for _ in range(len(scores))]
        fig, ax = plt.subplots(1, 1, figsize=(35, 20))
        ax.plot(scores, label='Hill Climbing')
        ax.plot(constant, label='Guess')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discrimination Success Probability')
        ax.set_ylim([0.7, 0.9])
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

def special_u():
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
    fig, ax = plt.subplots(1, 1, figsize=(35, 25))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.1)
    ax.plot(X, y_theory, label='Theoritical equation')
    ax.plot(X, y_guess, label='Guess, evaluate by SDP')
    ax.plot(X, y_hillclimb, label='Hill climbing')
    ax.legend()
    ax.set_xlabel('Theta (in degrees)')
    ax.set_ylabel('Success Probability')
    ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
    filename = 'result/4.6.2022/varying_theta'
    fig.savefig(filename)

    # print(data[0][1]['Guess'].init_state)


def print_results():
    logs = ['result/4.10.2022/varying_theta_unambiguous']
    data = Logger.read_log(logs)
    for experiment in data:
        myinput = experiment[0]
        output_by_method = experiment[1]
        
        # if myinput.unitary_theta in [40, 50, 60, 70, 80, 90] and output_by_method['Hill climbing'].start_seed == 0:
        if output_by_method['Hill climbing'].start_seed == 0:
            print('\ntheta =', myinput.unitary_theta)
            print(output_by_method['Guess'].init_state)
            print('Guess success probability =', output_by_method['Guess'].success)
            print(output_by_method['Hill climbing'].init_state)
            eval_metric = output_by_method['Hill climbing'].eval_metric
            print(f'Hill climbing ({eval_metric}) probability =', output_by_method['Hill climbing'].success)
            print('---')


if __name__ == '__main__':
    # scores = [0.789991, 0.832989, 0.845341, 0.852194, 0.857062, 0.859864, 0.860891, 0.860928, 0.861471, 0.861524, 0.861533, 0.861536, 0.861537, 0.861541, 0.861543, 0.861763, 0.861993, 0.862333, 0.862427, 0.86244, 0.862667, 0.862849, 0.862851, 0.862853, 0.862854, 0.862854, 0.862855, 0.862856, 0.862856, 0.862857, 0.862857, 0.862857, 0.862858, 0.862858, 0.862858, 0.862858, 0.862858, 0.862858, 0.86286, 0.862944, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862949, 0.862955, 0.862989, 0.863003, 0.863005]
    # constant = 0.794494
    # Plot.hillclimbing(scores, constant)

    # vary_priors()
    # vary_numsensors()
    # vary_startseed()
    # special_u()

    print_results()
