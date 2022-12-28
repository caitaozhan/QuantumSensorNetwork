import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from logger import Logger
from optimize_initial_state import OptimizeInitialState
from utility import Utility
from collections import defaultdict

class Plot:

    plt.rcParams['font.size'] = 60
    plt.rcParams['lines.linewidth'] = 6

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
    _LABEL   = ['Hill Climbing', 'Simulated Annealing', 'Genetic Algorithm']
    METHOD  = dict(zip(_METHOD, _LABEL))

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
    _STYLE  = ['solid',         'dashed',              'dotted']
    LINE_STYLE = dict(zip(_METHOD, _STYLE))

    @staticmethod
    def vary_theta(data, filename):
        '''vary theta, hill climbing, n-sensors (n=2,3,4,5)
        '''
        method = 'Hill climbing'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            table[myinput.num_sensor].append({myinput.unitary_theta: output_by_method[method].success})
        Y = defaultdict(list)
        X = [i for i in range(1, 180)]
        for num_sensor, mylist in table.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, success in key_val.items():
                    y[theta] = success
            y2 = []
            for theta in X:
                if theta in y:
                    y2.append(y[theta])
                else:
                    y2.append(y[180-theta])  # 5 sensors doesn't has data for theta > 91 degrees
            Y[num_sensor] = y2
        
        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15)
        ax.plot(X, Y[2], label='2 Sensors')
        ax.plot(X, Y[3], label='3 Sensors')
        ax.plot(X, Y[4], label='4 Sensors')
        ax.plot(X, Y[5], label='5 Sensors')
        ax.vlines(x=45,   ymin=0.2, ymax=1, linestyles='dotted', colors='black')
        ax.vlines(x=60,   ymin=0.2, ymax=1, linestyles='dotted', colors='black')
        ax.vlines(x=65.9, ymin=0.2, ymax=1, linestyles='dotted', colors='black')
        ax.legend(fontsize='55', loc='center', bbox_to_anchor=(0.6, 0.25))
        xticks = [i for i in range(0, 181, 15)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax.set_xlim([0, 180])
        ax.set_ylim([0.2, 1.02])
        ax.set_xlabel('Theta (degrees)', labelpad=15)
        ax.set_ylabel('Success Probability (%)')
        ax.set_title('Empirical Validation of Theorem 1 & 2 By Hill Climbing', pad=30, fontsize=65)
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        fig.savefig(filename)


    @staticmethod
    def methods_similar(data1: list, data2: list, filename: str):
        # data1: varying theta, 4 sensors
        table1 = defaultdict(list)
        for myinput, output_by_methods in data1:
            for method, output in output_by_methods.items():
                table1[method].append({myinput.unitary_theta : output.success})
        Y = defaultdict(list)
        X = [i for i in range(1, 90)]
        for method, mylist in table1.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, success in key_val.items():
                    y[theta] = success
            y2 = []
            for theta in X:
                if theta in y:
                    y2.append(y[theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y[method] = y2

        # data2: varying iteration, 4 sensors, theta = 40 case
        table2 = {}
        for myinput, output_by_methods in data2:
            for method, output in output_by_methods.items():
                table2[method] = output.scores[:50]
        Y2 = table2
        
        # plotting
        methods = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(30, 17))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.2)
        # ax0
        ax0.plot(X, Y[methods[0]], label=Plot.METHOD[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]])
        ax0.plot(X, Y[methods[1]], label=Plot.METHOD[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]])
        ax0.plot(X, Y[methods[2]], label=Plot.METHOD[methods[2]], linestyle=Plot.LINE_STYLE[methods[2]])
        ax0.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0.2, 1.2, 0.05, 0.05), fontsize=55, columnspacing=0.8)
        xticks = [i for i in range(0, 91, 15)]
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.2, 0.4, 0.6, 0.8, 1]
        ax0.set_yticks(yticks)
        ax0.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax0.set_ylim([0.25, 1.02])
        ax0.vlines(x=40, ymin=0.2, ymax=0.883, linestyles='dotted', colors='black')
        ax0.hlines(y=0.883, xmin=0,   xmax=40, linestyles='dotted', colors='black')
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_title('HC, SA, GA Performs Similary', fontsize=55)
        ax0.set_xlabel('Theta (degrees)', labelpad=15)
        ax0.set_xlim([0, 90])
        ax0.set_ylim([0.2, 1.02])
        ax0.set_ylabel('Success Probability (%)')
        ax0.text(42, 0.865, '(40, 0.883)', fontsize=50)
        # ax1
        ax1.plot(Y2[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]])
        ax1.plot(Y2[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]])
        ax1.plot(Y2[methods[2]], linestyle=Plot.LINE_STYLE[methods[2]])
        xticks = [i for i in range(0, 51, 10)]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.8, 0.82, 0.84, 0.86, 0.88]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_title('Searching Process (Theta = 40)', fontsize=55)
        ax1.set_xlabel('Iteration Number', labelpad=15)
        ax1.set_xlim([-1, 50])
        ax1.set_ylim([0.805, 0.89])
        plt.figtext(0.28, 0.01, '(a)')
        plt.figtext(0.75, 0.01, '(b)')
        fig.savefig(filename)


def vary_theta():
    logs = ['result2/12.22.2022/varying_theta_2sensors', 'result2/12.22.2022/varying_theta_3sensors', \
            'result2/12.22.2022/varying_theta_4sensors', 'result2/12.22.2022/varying_theta_5sensors']
    data = Logger.read_log(logs)
    filename = 'result2/12.22.2022/varying_theta_hillclimbing_nsensors.png'
    Plot.vary_theta(data, filename)


def methods_similar():
    logs1 = ['result2/12.22.2022/varying_theta_4sensors', 'result2/12.26.2022/compare_methods_4sensors']
    logs2 = ['result2/12.23.2022/compare_methods_4sensors']
    data1 = Logger.read_log(logs1)
    data2 = Logger.read_log(logs2)
    filename = 'result2/12.26.2022/compare_methods_similar.png'
    Plot.methods_similar(data1, data2, filename)


if __name__ == '__main__':
    vary_theta()
    methods_similar()

    
