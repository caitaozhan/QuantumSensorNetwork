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

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm', 'Theorem']
    _LABEL  = ['Hill Climbing', 'Simulated Annealing', 'Genetic Algorithm', '$Conjecture$ 1 + $Corollary$ 1']
    METHOD  = dict(zip(_METHOD, _LABEL))

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm', 'Theorem']
    _STYLE  = ['solid',         'dashed',              'dotted',            'dashdot']
    LINE_STYLE = dict(zip(_METHOD, _STYLE))

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm', 'Theorem']
    _COLOR  = ['blue',          'orange',              'green',             'pink']
    COLOR   = dict(zip(_METHOD, _COLOR))


    @staticmethod
    def vary_theta(data, filename):
        '''vary theta, hill climbing, n-sensors (n=2,3,4,5)
        '''
        method = 'Hill climbing'
        table = defaultdict(list)
        for myinput, output_by_method in data:
            table[myinput.num_sensor].append({myinput.unitary_theta: output_by_method[method].error})
        Y = defaultdict(list)
        X = [i for i in range(1, 180)]
        for num_sensor, mylist in table.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X:
                if theta in y:
                    y2.append(y[theta])
                else:
                    y2.append(y[180-theta])  # 5 sensors doesn't has data for theta > 91 degrees
            Y[num_sensor] = y2
        
        fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.15)
        ax.plot(X, Y[5], label='5 Sensors', linewidth=10)
        ax.plot(X, Y[4], label='4 Sensors', linewidth=10)
        ax.plot(X, Y[3], label='3 Sensors', linewidth=10)
        ax.plot(X, Y[2], label='2 Sensors', linewidth=10)
        ax.vlines(x=45,   ymin=0, ymax=1, linestyles='dotted', colors='black')
        ax.vlines(x=60,   ymin=0, ymax=1, linestyles='dotted', colors='black')
        ax.vlines(x=65.9, ymin=0, ymax=1, linestyles='dotted', colors='black')
        ax.legend(fontsize=55, loc='center', bbox_to_anchor=(0.6, 0.5), edgecolor='black')
        xticks = [i for i in range(0, 181, 15)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax.set_xlim([0, 180])
        ax.set_ylim([-0.002, 1])
        ax.set_xlabel('Theta (degree)', labelpad=15)
        ax.set_ylabel('Probability of Error (%)')
        ax.set_title('Empirical Validation of Hill Climbing', pad=30, fontsize=65)
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        fig.savefig(filename)


    @staticmethod
    def methods_similar_old(data1: list, data2: list, filename: str):
        # data1: varying theta, 4 sensors
        table1 = defaultdict(list)
        for myinput, output_by_methods in data1:
            for method, output in output_by_methods.items():
                table1[method].append({myinput.unitary_theta: output.success})
        Y = defaultdict(list)
        X = [i for i in range(1, 91)]
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
        arrowprops = dict(facecolor='black', width=5, headwidth=25)
        methods = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(30, 17))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.83, bottom=0.19)
        # ax0
        ax0.plot(X, Y[methods[0]], label=Plot.METHOD[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=9)
        ax0.plot(X, Y[methods[1]], label=Plot.METHOD[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=7)
        ax0.plot(X, Y[methods[2]], label=Plot.METHOD[methods[2]], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=5)
        ax0.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0.2, 1.23, 0.05, 0.05), fontsize=55, columnspacing=0.8, edgecolor='black')
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
        ax0.set_title('HC, SA, GA Performs Similary', fontsize=55, pad=20)
        ax0.set_xlabel('Theta (degree)', labelpad=15)
        ax0.set_xlim([0, 90])
        ax0.set_ylim([0.2, 1.02])
        ax0.set_ylabel('Probability of Success (%)')
        ax0.annotate('(40, 0.883)', xy=(42, 0.883), xytext=(51, 0.883), arrowprops=arrowprops, fontsize=50, va='center')
        # ax1
        ax1.plot(Y2[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=9)
        ax1.plot(Y2[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=7)
        ax1.plot(Y2[methods[2]], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=5)
        xticks = [i for i in range(0, 51, 10)]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.8, 0.82, 0.84, 0.86, 0.88]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_title('Searching Process (Theta = 40)', fontsize=55, pad=20)
        ax1.set_xlabel('Iteration Number', labelpad=15)
        ax1.set_xlim([-1, 50])
        ax1.set_ylim([0.805, 0.89])
        ax1.annotate('Random Initial State', xy=(0.7, 0.811), xytext=(6, 0.811), arrowprops=arrowprops, fontsize=50, va='center')
        plt.figtext(0.28, 0.01, '(a)')
        plt.figtext(0.75, 0.01, '(b)')
        fig.savefig(filename)


    @staticmethod
    def methods_similar(data1: list, data2: list, filename: str):
        # data1: varying theta, 4 sensors
        table1 = defaultdict(list)
        for myinput, output_by_methods in data1:
            for method, output in output_by_methods.items():
                table1[method].append({myinput.unitary_theta: output.success})
        Y = defaultdict(list)
        X = [i for i in range(1, 91)]
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
        arrowprops = dict(facecolor='black', width=5, headwidth=25)
        methods = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(30, 17))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.83, bottom=0.19)
        # ax0
        ax0.plot(X, Y[methods[0]], label=Plot.METHOD[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=9)
        ax0.plot(X, Y[methods[1]], label=Plot.METHOD[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=7)
        ax0.plot(X, Y[methods[2]], label=Plot.METHOD[methods[2]], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=5)
        ax0.legend(ncol=3, loc='upper left', bbox_to_anchor=(-0.2, 1.23, 0.05, 0.05), fontsize=55, columnspacing=0.8, edgecolor='black')
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
        ax0.set_title('HC, SA, GA Performs Similary', fontsize=55, pad=20)
        ax0.set_xlabel('Theta (degree)', labelpad=15)
        ax0.set_xlim([0, 90])
        ax0.set_ylim([0.2, 1.02])
        ax0.set_ylabel('Probability of Success (%)')
        ax0.annotate('(40, 0.883)', xy=(42, 0.883), xytext=(51, 0.883), arrowprops=arrowprops, fontsize=50, va='center')
        # ax1
        ax1.plot(Y2[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=9)
        ax1.plot(Y2[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=7)
        ax1.plot(Y2[methods[2]], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=5)
        xticks = [i for i in range(0, 51, 10)]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.8, 0.82, 0.84, 0.86, 0.88]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_title('Searching Process (Theta = 40)', fontsize=55, pad=20)
        ax1.set_xlabel('Iteration Number', labelpad=15)
        ax1.set_xlim([-1, 50])
        ax1.set_ylim([0.805, 0.89])
        ax1.annotate('Random Initial State', xy=(0.7, 0.811), xytext=(6, 0.811), arrowprops=arrowprops, fontsize=50, va='center')
        plt.figtext(0.28, 0.01, '(a)')
        plt.figtext(0.75, 0.01, '(b)')
        fig.savefig(filename)



    @staticmethod
    def lemma2(data, filename):
        plt.rcParams['font.size'] = 45
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(22, 15))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.12, wspace=0.4, hspace=0.3)
        
        boxprops = dict(linewidth=5)
        medianprops = dict(linewidth=5)
        whiskerprops = dict(linewidth=5)
        capprops = dict(linewidth=5)

        n = 2
        print(data[f'n{n}.perm'][0], min(data[f'n{n}.avg']), max(data[f'n{n}.avg']))
        ax0.hlines(y=data[f'n{n}.perm'][0], xmin=0.85, xmax=1.15, linewidth=5)
        ax0.boxplot(data[f'n{n}.avg'], whis=(0, 100), widths=0.2, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
        xticks = [1]
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f'{x+1}' for x in xticks])
        yticks = [0.069, 0.070, 0.071, 0.072, 0.073]
        ax0.set_yticks(yticks)
        ax0.set_yticklabels([f'{round(y*100, 1)}' for y in yticks])
        ax0.tick_params(axis='x', direction='in', length=8, width=2, pad=15)
        ax0.tick_params(axis='y', direction='in', length=8, width=2, pad=15)
        n = 3
        print(data[f'n{n}.perm'][0], min(data[f'n{n}.avg']), max(data[f'n{n}.avg']))
        ax1.hlines(y=data[f'n{n}.perm'][0], xmin=0.85, xmax=1.15, linewidth=5)
        ax1.boxplot(data[f'n{n}.avg'], whis=(0, 100), widths=0.2, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x+2}' for x in xticks])
        yticks = [0.189, 0.191, 0.193, 0.195, 0.197]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f'{round(y*100, 1)}' for y in yticks])
        ax1.tick_params(axis='x', direction='in', length=8, width=2, pad=15)
        ax1.tick_params(axis='y', direction='in', length=8, width=2, pad=15)
        n = 4
        print(data[f'n{n}.perm'][0], min(data[f'n{n}.avg']), max(data[f'n{n}.avg']))
        ax2.hlines(y=data[f'n{n}.perm'][0], xmin=0.85, xmax=1.15, linewidth=5)
        ax2.boxplot(data[f'n{n}.avg'], whis=(0, 100), widths=0.2, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f'{x+3}' for x in xticks])
        yticks = [0.182, 0.186, 0.190, 0.194, 0.198]
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([f'{round(y*100, 1)}' for y in yticks])
        ax2.tick_params(axis='x', direction='in', length=8, width=2, pad=15)
        ax2.tick_params(axis='y', direction='in', length=8, width=2, pad=15)
        n = 5
        print(data[f'n{n}.perm'][0], min(data[f'n{n}.avg']), max(data[f'n{n}.avg']))
        ax3.hlines(y=data[f'n{n}.perm'][0], xmin=0.85, xmax=1.15, linewidth=5)
        ax3.boxplot(data[f'n{n}.avg'], whis=(0, 100), widths=0.2, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, capprops=capprops)
        ax3.set_xticks(xticks)
        ax3.set_xticklabels([f'{x+4}' for x in xticks])
        yticks = [0.224, 0.227, 0.23, 0.233, 0.236]
        ax3.set_yticks(yticks)
        ax3.set_yticklabels([f'{round(y*100, 1)}' for y in yticks])
        ax3.tick_params(axis='x', direction='in', length=8, width=2, pad=15)
        ax3.tick_params(axis='y', direction='in', length=8, width=2, pad=15)
        # fig
        fig.supylabel('Probability of Error (%)')
        fig.supxlabel('Number of Sensors')
        fig.suptitle('$Lemma\ 2$: The Averaged Initial States Have Lower $PoE$')
        fig.savefig(filename)


    @staticmethod
    def lemma3(data, filename):
        fig, ax = plt.subplots(1, 1, figsize=(24, 14))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.16)

        X = np.linspace(0, 0.99, 100)
        X = np.append(X, [0.995, 0.998, 0.999, 0.99999])
        n=2
        ax.plot(X, data[f'n{n}'], label='2 Sensors')
        n=3
        ax.plot(X, data[f'n{n}'], label='3 Sensors')
        n=4
        ax.plot(X, data[f'n{n}'], label='4 Sensors')
        n=5
        ax.plot(X, data[f'n{n}'], label='5 Sensors')
        ax.legend(fontsize=50, edgecolor='black')
        yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        xticks = xticks[::-1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        ax.set_xlim([0, 1.001])
        ax.set_ylim([-0.001, 0.801])
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_xlabel('$x$ (such that $\\forall i \\neq j$: $\langle \phi_i \| \phi_j \\rangle = x$)', labelpad=22)
        ax.set_ylabel('Probability of Error (%)', labelpad=30, fontsize=58)
        ax.set_title('$Lemma\ 3$: $PoE$ Decrease with the Decrease in $x$', pad=40, fontsize=58)
        ax.invert_xaxis()
        fig.savefig(filename)


    @staticmethod
    def conjecture_1(data, filename):
        table_3sensor = defaultdict(list)
        table_5sensor = defaultdict(list)
        for myinput, output_by_methods in data:
            if myinput.num_sensor == 3:
                for method, output in output_by_methods.items():
                    table_3sensor[method].append({myinput.unitary_theta: output.success})
            if myinput.num_sensor == 5:
                for method, output in output_by_methods.items():
                    table_5sensor[method].append({myinput.unitary_theta: output.success})

        X = [i for i in range(1, 90)]
        Y3 = defaultdict(list)
        for method, mylist in table_3sensor.items():
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
            Y3[method] = y2
        Y5 = defaultdict(list)
        for method, mylist in table_5sensor.items():
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
            Y5[method] = y2
        
        methods = ['Hill climbing', 'Theorem']
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(30, 17))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.83, bottom=0.19)
        # ax0
        ax0.plot(X, Y3[methods[0]], label=Plot.METHOD[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=9)
        ax0.plot(X, Y3[methods[1]], label=Plot.METHOD[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=7)
        xticks = [i for i in range(0, 91, 15)]
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.2, 0.4, 0.6, 0.8, 1]
        ax0.set_yticks(yticks)
        ax0.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax0.set_xlim([0, 90])
        ax0.set_ylim([0.2, 1.02])
        ax0.vlines(x=60, ymin=0.2, ymax=1, linestyles='dotted', colors='black', )
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax0.set_title('3 Sensors', fontsize=60, pad=20)
        ax0.legend(ncol=2, loc='upper left', bbox_to_anchor=(0.08, 1.23, 0.05, 0.05), fontsize=55, handlelength=3.5, edgecolor='black')
        ax0.set_xlabel('Theta (degree)', labelpad=15)
        ax0.set_ylabel('Probability of Success (%)', fontsize=60)
        # ax1
        ax1.plot(X, Y5[methods[0]], label=Plot.METHOD[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=9)
        ax1.plot(X, Y5[methods[1]], label=Plot.METHOD[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=7)
        xticks = [i for i in range(0, 91, 15)]
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.2, 0.4, 0.6, 0.8, 1]
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax1.set_xlim([0, 90])
        ax1.set_ylim([0.2, 1.02])
        ax1.vlines(x=65.9, ymin=0.2, ymax=1, linestyles='dotted', colors='black', )
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_title('5 Sensors', fontsize=60, pad=20)
        ax1.set_xlabel('Theta (degree)', labelpad=15)
        plt.figtext(0.28, 0.01, '(a)')
        plt.figtext(0.75, 0.01, '(b)')
        fig.savefig(filename)


    @staticmethod
    def conjecture_2(data, filename):
        # data
        table_6sensor = defaultdict(list)
        table_7sensor = defaultdict(list)
        table_8sensor = defaultdict(list)
        table_9sensor = defaultdict(list)
        for myinput, output_by_methods in data:
            if myinput.num_sensor == 6:
                for method, output in output_by_methods.items():
                    table_6sensor[method].append({myinput.unitary_theta: output.success})
            if myinput.num_sensor == 7:
                for method, output in output_by_methods.items():
                    table_7sensor[method].append({myinput.unitary_theta: output.success})
            if myinput.num_sensor == 8:
                for method, output in output_by_methods.items():
                    table_8sensor[method].append({myinput.unitary_theta: output.success})
            if myinput.num_sensor == 9:
                for method, output in output_by_methods.items():
                    table_9sensor[method].append({myinput.unitary_theta: output.success})

        X = [i for i in range(1, 91)]
        Y6 = defaultdict(list)
        for method, mylist in table_6sensor.items():
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
            Y6[method] = y2
        Y7 = defaultdict(list)
        for method, mylist in table_7sensor.items():
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
            Y7[method] = y2
        Y8 = defaultdict(list)
        for method, mylist in table_8sensor.items():
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
            Y8[method] = y2
        Y9 = defaultdict(list)
        for method, mylist in table_9sensor.items():
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
            Y9[method] = y2
        
        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(24, 14))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.16)
        method = 'Theorem'
        ax.plot(X, Y6[method], label='6 Sensors')
        ax.plot(X, Y7[method], label='7 Sensors')
        ax.plot(X, Y8[method], label='8 Sensors')
        ax.plot(X, Y9[method], label='9 Sensors')
        ax.legend(edgecolor='black', fontsize=55, loc='lower center')
        ax.vlines(x=65.9,   ymin=0, ymax=1, linestyles='dotted', colors='black')
        ax.vlines(x=69.3,   ymin=0, ymax=1, linestyles='dotted', colors='black')
        ax.vlines(x=71.6,   ymin=0, ymax=1, linestyles='dotted', colors='black')
        xticks = [i for i in range(0, 91, 15)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax.set_ylim([0.1, 1.02])
        ax.set_xlim([0, 90])
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_xlabel('Theta (degree)', labelpad=20)
        ax.set_ylabel('Probability of Success (%)')
        ax.set_title('The Performance of $Conjecture\ 1$ and $Corollary\ 1$', fontsize=60, pad=30)
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


def lemma2():
    file_perm = 'result2/12.28.2022/lemma2.n{}.perm.npy'
    file_avg = 'result2/12.28.2022/lemma2.n{}.avg.npy'
    data = {}
    for n in range(2, 6):
        perm = np.load(file_perm.format(n))
        avg  = np.load(file_avg.format(n))
        data[f'n{n}.perm'] = perm
        data[f'n{n}.avg'] = avg
    filename = 'result2/12.28.2022/lemma2.png'
    Plot.lemma2(data, filename)


def lemma3():
    file = 'result2/1.6.2023/lemma3.n{}.npy'
    data = {}
    for n in range(2, 6):
        y = np.load(file.format(n))
        data[f'n{n}'] = y
    filename = 'result2/1.6.2023/lemma3.png'
    Plot.lemma3(data, filename)


def conjecture():
    logs = ['result2/12.31.2022/conjecture_3sensor', 'result2/12.31.2022/conjecture_5sensor',
            'result2/12.22.2022/varying_theta_3sensors', 'result2/12.22.2022/varying_theta_5sensors']
    data = Logger.read_log(logs)
    filename = 'result2/12.31.2022/conjecture_1.png'
    Plot.conjecture_1(data, filename)

    logs = ['result2/12.31.2022/conjecture_6sensor', 'result2/12.31.2022/conjecture_7sensor',
            'result2/12.31.2022/conjecture_8sensor', 'result2/12.31.2022/conjecture_9sensor']
    data = Logger.read_log(logs)
    filename = 'result2/12.31.2022/conjecture_2.png'
    Plot.conjecture_2(data, filename)


if __name__ == '__main__':
    # vary_theta()
    methods_similar()
    # lemma2()
    # lemma3()
    # conjecture()
    
