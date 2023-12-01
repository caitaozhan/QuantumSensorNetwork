import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from logger import Logger
from collections import defaultdict


class Plot:

    plt.rcParams['font.size'] = 60
    plt.rcParams['lines.linewidth'] = 7

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm', 'Theorem',        'GHZ',    'Non entangle', 'Theorem povm-noise']
    _LABEL  = ['Hill Climbing', 'Simulated Annealing', 'Genetic Algorithm', '$Conjecture$ 1', 'GHZ',    'Uniform-all', '$Conjecture$ 1 (povm-noise)']
    METHOD  = dict(zip(_METHOD, _LABEL))

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm', 'Theorem',        'GHZ',    'Non entangle']
    _STYLE  = ['solid',         'dashed',              'dotted',            'dashed',         'dotted', 'solid']
    LINE_STYLE = dict(zip(_METHOD, _STYLE))

    _METHOD = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm', 'Theorem',        'GHZ',    'Non entangle', 'Theorem povm-noise']
    _COLOR  = ['blue',          'orange',              'green',             'lime',           'gray',   'black',        'green']
    COLOR   = dict(zip(_METHOD, _COLOR))


    @staticmethod
    def vary_theta(data, filename):
        '''vary theta, hill climbing, n-sensors (n=2,3,4,5)
        '''
        methods = ['Hill climbing', 'Simulated annealing']
        table_2sensor = defaultdict(list)
        table_3sensor = defaultdict(list)
        table_4sensor = defaultdict(list)
        table_5sensor = defaultdict(list)

        for myinput, output_by_methods in data:
            if myinput.num_sensor == 2:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_2sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 3:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_3sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 4:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_4sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 5:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_5sensor[method].append({myinput.unitary_theta: output.error})

        Y2 = defaultdict(list)
        X2 = [i for i in range(1, 180)]
        for method, mylist in table_2sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X2:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y2[method] = y2

        Y3 = defaultdict(list)
        X3 = [i for i in range(1, 180)]
        for method, mylist in table_3sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X3:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y3[method] = y2

        Y4 = defaultdict(list)
        X4 = [i for i in range(1, 180)]
        for method, mylist in table_4sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X4:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y4[method] = y2

        Y5 = defaultdict(list)
        X5 = [i for i in range(1, 180)]
        for method, mylist in table_5sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X5:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y5[method] = y2
        
        # step 2: plotting

        fig, ax = plt.subplots(figsize=(30, 18))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.12)

        ax.plot(X2[:45], Y2[methods[0]][:45], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11, label=Plot.METHOD[methods[0]])
        ax.plot(X2[45:], Y2[methods[0]][45:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X2[:45], Y2[methods[1]][:45], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9, label=Plot.METHOD[methods[1]])
        ax.plot(X2[45:], Y2[methods[1]][45:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X3[:60], Y3[methods[0]][:60], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X3[60:], Y3[methods[0]][60:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X3[:60], Y3[methods[1]][:60], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X3[60:], Y3[methods[1]][60:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X4[:60], Y4[methods[0]][:60], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X4[60:], Y4[methods[0]][60:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X4[:60], Y4[methods[1]][:60], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X4[60:], Y4[methods[1]][60:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X5[:65], Y5[methods[0]][:65], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X5[65:], Y5[methods[0]][65:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X5[:65], Y5[methods[1]][:65], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X5[65:], Y5[methods[1]][65:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        arrowprops = dict(facecolor='black', width=5, headwidth=20)
        ax.annotate('2 Sensor', xy=(154.5, 0.1), xytext=(158, 0.05), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('3 Sensor', xy=(162.5, 0.4),   xytext=(128, 0.4), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('4 Sensor', xy=(164, 0.5),  xytext=(128, 0.5), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('5 Sensor', xy=(166, 0.6), xytext=(128, 0.6), arrowprops=arrowprops, fontsize=50, va='center')
        ax.vlines(x=45,   ymin=-0.002, ymax=0.5, linestyles='solid', colors='grey', zorder=10)
        ax.vlines(x=60,   ymin=-0.002, ymax=0.5, linestyles='solid', colors='grey', zorder=10)
        ax.vlines(x=65.9, ymin=-0.002, ymax=0.43, linestyles='solid', colors='grey', zorder=10)
        ax.text(34, 0.51, '$T$: 45    60')
        ax.text(62, 0.44, '65.9')

        xticks = [i for i in range(0, 181, 15)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax.set_xlim([0, 180])
        ax.set_ylim([-0.002, 1])
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_title('Empirical Validation of Search Heuristics', fontsize=65, pad=40)
        ax.legend(ncol=2, loc='upper right',  fontsize=55, handlelength=3.5, edgecolor='black')
        ax.set_xlabel('$\\theta$ (degree)', labelpad=15)
        ax.set_ylabel('Optimal Objective Value $P()$ (%)', fontsize=60, labelpad=10)
        fig.savefig(filename)


    @staticmethod
    def methods_similar(data: list, filename: str):
        # data2: varying iteration, 4 sensors, theta = 40 case
        theta = 46
        table = {}
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.unitary_theta == theta:
                    table[method] = 1 - np.array(output.scores[:100])  # success --> error
        Y = table
        
        # plotting
        arrowprops = dict(facecolor='black', width=5, headwidth=25)
        methods = ['Genetic algorithm', 'Simulated annealing', 'Hill climbing']
        fig, ax = plt.subplots(figsize=(26, 16))
        fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.14)
        
        ax.plot(Y[methods[0]], label=Plot.METHOD[methods[0]], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=10)
        ax.plot(Y[methods[2]], label=Plot.METHOD[methods[2]], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=10)
        ax.plot(Y[methods[1]], label=Plot.METHOD[methods[1]], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=8)
        ax.legend(fontsize=50, bbox_to_anchor=(0.45, 0.8), handlelength=3)
        xticks = [i for i in range(0, 101, 10)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks], fontsize=50)
        yticks = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks], fontsize=50)
        ax.tick_params(axis='x', direction='in',  length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='out', length=10, width=3, pad=15)
        ax.set_title('Heuristic Algo. Searching Process when $\\theta$ = 46', fontsize=55, pad=50)
        ax.set_xlabel('Iteration Number', labelpad=30, fontsize=50)
        ax.set_xlim([-0.1, 100])
        ax.set_ylabel('Optimal Objective Value $P()$ (%)', fontsize=50, labelpad=30)
        ax.set_ylim([0.05, 0.133])
        ax.annotate('Random Initial State', xy=(0.5, 0.189), xytext=(5, 0.189), arrowprops=arrowprops, fontsize=50, va='center')
        fig.savefig(filename)


    @staticmethod
    def lemma2(data, filename):
        fig, axes = plt.subplots(1, 4, figsize=(40, 21))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.87, bottom=0.09, wspace=0.4, hspace=0.3)
        
        for i, t in enumerate([26, 46, 66, 86]):
            y_min, y_max, y = [], [], []
            for n in [3, 4, 5]:
                y_min.append(min(data[f'n{n}-t{t}.avg']))
                y_max.append(max(data[f'n{n}-t{t}.avg']))
                y.append(data[f'n{n}-t{t}.perm'][0])
            y_min = np.array(y_min) * 100
            y_max = np.array(y_max) * 100
            y = np.array(y) * 100
            X = [3,4,5]
            yerr = np.stack([y - y_min, y_max - y])
            axes[i].errorbar(X, y, yerr=yerr, linewidth=5, capsize=20, capthick=5, fmt=' ', marker='.', markersize=45, color='r', ecolor='b')
            axes[i].yaxis.grid()
            axes[i].set_xlim([2.7, 5.3])
            axes[i].set_title(f'$\\theta$={t}', fontsize=60, pad=20)
            axes[i].set_xlabel('Number of Sensors', fontsize=58, labelpad=15)
            axes[i].tick_params(axis='y', direction='in', length=10, width=4)
            axes[i].tick_params(pad=10)
        
        fig.supylabel('Optimal Objective Value $P()$ (%)')
        # fig.supxlabel('Theta (degree)')
        fig.suptitle('$Conjecture\ 2$: The Averaged Initial States Have Lower $PoE$', fontsize=75)
        fig.savefig(filename)


    @staticmethod
    def lemma3(data, filename):
        fig, ax = plt.subplots(1, 1, figsize=(26, 16))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.14)

        X = np.linspace(0, 0.99, 100)
        X = np.append(X, [0.995, 0.998, 0.999, 0.99999])
        n=5
        ax.plot(X, data[f'n{n}'], label='5 States')
        n=4
        ax.plot(X, data[f'n{n}'], label='4 States')
        n=3
        ax.plot(X, data[f'n{n}'], label='3 States')
        n=2
        ax.plot(X, data[f'n{n}'], label='2 States')
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
        ax.set_xlabel('$x$ (such that $\\forall i \\neq j$: $\langle \phi_i \| \phi_j \\rangle = x$)', labelpad=20)
        ax.set_ylabel('Probability of Error (%)', labelpad=40, fontsize=58)
        ax.set_title('$Conjecture\ 3$: $PoE$ Increase with the Increase in $x$', pad=50, fontsize=58)
        ax.grid()
        fig.savefig(filename)


    @staticmethod
    def lemma3_tmp(data, filename):
        import math
        def p(r1: float, n: int) -> float:
            '''probability of success as a function of r1
               n number of states
            '''
            a = math.sqrt(1 - (n-1)*r1)
            b = (n-1)*math.sqrt(r1)
            return 1/n * (a + b) ** 2

        def poe_list(X: list, n: int) -> list:
            '''X: a list of inner-product values'''
            y = []
            for x in X:
                r1 = (1-x) / n
                p_success = p(r1, n)
                y.append(1 - p_success)
            return y

        fig, ax = plt.subplots(1, 1, figsize=(26, 16))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.91, bottom=0.14)

        X = np.linspace(0, 0.99, 100)
        X = np.append(X, [0.995, 0.998, 0.999, 0.99999])
        n=5
        ax.plot(X, data[f'n{n}'], label='5 States')
        ax.plot(X, poe_list(X, n), label='5 States, PGM', linestyle='dotted')
        n=4
        ax.plot(X, data[f'n{n}'], label='4 States')
        ax.plot(X, poe_list(X, n), label='4 States, PGM', linestyle='dotted')
        n=3
        ax.plot(X, data[f'n{n}'], label='3 States')
        ax.plot(X, poe_list(X, n), label='3 States, PGM', linestyle='dotted')
        n=2
        ax.plot(X, data[f'n{n}'], label='2 States')
        ax.plot(X, poe_list(X, n), label='2 States, PGM', linestyle='dotted')
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
        ax.set_xlabel('$x$ (such that $\\forall i \\neq j$: $\langle \phi_i \| \phi_j \\rangle = x$)', labelpad=20)
        ax.set_ylabel('Probability of Error (%)', labelpad=40, fontsize=58)
        ax.set_title('$Conjecture\ 3$: $PoE$ Increase with the Increase in $x$', pad=50, fontsize=58)
        ax.grid()
        fig.savefig(filename)


    @staticmethod
    def conjecture(data, filename):
        # step 1: prepare data

        table_2sensor = defaultdict(list)
        table_3sensor = defaultdict(list)
        table_4sensor = defaultdict(list)
        table_5sensor = defaultdict(list)

        for myinput, output_by_methods in data:
            if myinput.num_sensor == 2:
                for method, output in output_by_methods.items():
                    table_2sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 3:
                for method, output in output_by_methods.items():
                    table_3sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 4:
                for method, output in output_by_methods.items():
                    table_4sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 5:
                for method, output in output_by_methods.items():
                    table_5sensor[method].append({myinput.unitary_theta: output.error})

        Y2 = defaultdict(list)
        X2 = [i for i in range(1, 46)] + [i for i in range(135, 180)]
        for method, mylist in table_2sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X2:
                if theta in y:
                    y2.append(y[theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y2[method] = y2

        Y3 = defaultdict(list)
        X3 = [i for i in range(1, 61)] + [i for i in range(120, 180)]
        for method, mylist in table_3sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X3:
                if theta in y:
                    y2.append(y[theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y3[method] = y2

        Y4 = defaultdict(list)
        X4 = [i for i in range(1, 61)] + [i for i in range(120, 180)]
        for method, mylist in table_4sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X4:
                if theta in y:
                    y2.append(y[theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y4[method] = y2

        Y5 = defaultdict(list)
        X5 = [i for i in range(1, 66)] + [i for i in range(115, 180)]
        for method, mylist in table_5sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X5:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y5[method] = y2
        
        # step 2: plotting

        methods = ['Hill climbing', 'Theorem']
        fig, ax = plt.subplots(figsize=(30, 18))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.12)

        ax.plot(X2[:45], Y2[methods[0]][:45], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11, label=Plot.METHOD[methods[0]])
        ax.plot(X2[45:], Y2[methods[0]][45:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X2[:45], Y2[methods[1]][:45], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9, label=Plot.METHOD[methods[1]])
        ax.plot(X2[45:], Y2[methods[1]][45:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X3[:60], Y3[methods[0]][:60], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X3[60:], Y3[methods[0]][60:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X3[:60], Y3[methods[1]][:60], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X3[60:], Y3[methods[1]][60:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X4[:60], Y4[methods[0]][:60], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X4[60:], Y4[methods[0]][60:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X4[:60], Y4[methods[1]][:60], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X4[60:], Y4[methods[1]][60:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X5[:65], Y5[methods[0]][:65], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X5[65:], Y5[methods[0]][65:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        ax.plot(X5[:65], Y5[methods[1]][:65], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X5[65:], Y5[methods[1]][65:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        arrowprops = dict(facecolor='black', width=5, headwidth=20)
        ax.annotate('2 Sensor', xy=(154.5, 0.1), xytext=(158, 0.05), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('3 Sensor', xy=(162.5, 0.4),   xytext=(128, 0.4), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('4 Sensor', xy=(164, 0.5),  xytext=(128, 0.5), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('5 Sensor', xy=(166, 0.6), xytext=(128, 0.6), arrowprops=arrowprops, fontsize=50, va='center')
        ax.vlines(x=45,   ymin=-0.002, ymax=0.5, linestyles='solid', colors='grey', zorder=10)
        ax.vlines(x=60,   ymin=-0.002, ymax=0.5, linestyles='solid', colors='grey', zorder=10)
        ax.vlines(x=65.9, ymin=-0.002, ymax=0.43, linestyles='solid', colors='grey', zorder=10)
        ax.text(34, 0.51, '$T$: 45    60')
        ax.text(62, 0.44, '65.9')

        xticks = [i for i in range(0, 181, 15)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax.set_xlim([0, 180])
        ax.set_ylim([-0.002, 1])
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_title('Empirical Validation of $Conjecture$ 1', fontsize=65, pad=40)
        ax.legend(ncol=2, loc='upper right',  fontsize=55, handlelength=3.5, edgecolor='black')
        ax.set_xlabel('$\\theta$ (degree)', labelpad=15)
        ax.set_ylabel('Optimal Objective Value $P()$ (%)', fontsize=60)
        fig.savefig(filename)


    @staticmethod
    def symmetry_varyseed(data: dict, filename: str):
        theta = 46
        table = defaultdict(list)
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if method == 'Hill climbing' and myinput.unitary_theta == theta:
                    table[output.start_seed] = output.symmetries
        print(table)
        fig, ax = plt.subplots(figsize=(22, 16))
        fig.subplots_adjust(left=0.13, right=0.96, top=0.9, bottom=0.15)
        seeds = [0, 1, 2, 3, 4]
        for seed in seeds:
            ax.plot(table[seed][:100], label=str(seed))
        ax.legend()
        ax.set_title(f'Hill Climbing at $theta={theta}$, Varying Seed', pad=40)
        ax.set_xlabel('Iteration', labelpad=20)
        ax.set_ylabel('Symmetry Index', labelpad=20)
        ax.set_ylim([-0.001, 1])
        ax.set_xlim([-0.1, 100])
        ax.tick_params(axis='x', direction='out', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='out', length=10, width=3, pad=15)
        fig.savefig(filename)


    @staticmethod
    def symmetry_varymethod(data: dict, filename: str):
        theta = 46
        seed = 0
        table = defaultdict(list)
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.unitary_theta == theta and output.start_seed == seed:
                    table[method] = output.symmetries
        print(table)
        fig, ax = plt.subplots(figsize=(22, 16))
        fig.subplots_adjust(left=0.13, right=0.96, top=0.9, bottom=0.15)
        methods = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
        for method in methods:
            ax.plot(table[method][:100], label=method, linestyle=Plot.LINE_STYLE[method], color=Plot.COLOR[method])
      
        ax.legend(fontsize=45, handlelength=3)
        ax.set_title(f'Various Methods at $\\theta={theta}$', pad=40, fontsize=60)
        ax.set_xlabel('Iteration', labelpad=20)
        ax.set_ylabel('Symmetry Index', labelpad=20)
        ax.set_ylim([-0.001, 1])
        ax.set_xlim([-0.1, 100])
        ax.tick_params(axis='x', direction='out', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='out', length=10, width=3, pad=15)
        fig.savefig(filename)


    @staticmethod
    def symmetry_varytheta(data: dict, filename: str):
        table = defaultdict(list)
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if method == 'Hill climbing' and output.start_seed == 0:
                    table[myinput.unitary_theta] = output.symmetries
        print(table)
        fig, ax = plt.subplots(figsize=(22, 16))
        fig.subplots_adjust(left=0.11, right=0.96, top=0.9, bottom=0.15)
        thetas = [20, 40, 60, 80]
        for theta in thetas:
            ax.plot(table[theta][:100], label=f'$\\theta$={theta}')
        ax.legend(fontsize=45, handlelength=3)
        ax.set_title('Hill Climbing with varying $\\theta$', pad=40, fontsize=60)
        ax.set_xlabel('Iteration', labelpad=20)
        # ax.set_ylabel('Symmetry Index', labelpad=20)
        ax.set_ylim([-0.001, 1])
        ax.set_xlim([-0.1, 100])
        ax.tick_params(axis='x', direction='out', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='out', length=10, width=3, pad=15)
        fig.savefig(filename)


    @staticmethod
    def symmetry_poe_varymethod(data: dict, filename: str):
        theta = 46
        table = defaultdict(list)
        key = '{}-{}'  # {method}-{seed}
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.unitary_theta == theta:
                    table[key.format(method, output.start_seed)].append(output.symmetries)                     # [0] is symmetry
                    table[key.format(method, output.start_seed)].append((1 - np.array(output.scores)) * 100)   # [1] is poe
        fig, ax = plt.subplots(figsize=(22, 16))
        fig.subplots_adjust(left=0.115, right=0.98, top=0.9, bottom=0.14)
        methods = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
        seeds = [0, 1, 2, 3, 4]
        for seed in seeds:
            for method in methods:
                if seed == 0:
                    ax.scatter(table[key.format(method, seed)][0], table[key.format(method, seed)][1], color=Plot.COLOR[method], s=120, label=Plot.METHOD[method])
                else:
                    ax.scatter(table[key.format(method, seed)][0], table[key.format(method, seed)][1], color=Plot.COLOR[method], s=120)
        ax.legend(fontsize=40, facecolor='lightgray', markerscale=2)
        ax.set_xlabel('Symmetry Index', labelpad=20)
        ax.set_ylabel('Optimal Objective Value $P()$ (%)', labelpad=20, fontsize=53)
        ax.set_xlim([0, 0.82])
        xticks = [0, 0.2, 0.4, 0.6, 0.8]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        ax.invert_xaxis()
        ax.set_ylim([4, 18])
        ax.set_title('$P()$ (%) and Symmetry Index', fontsize=60, pad=40)
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=25)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        fig.savefig(filename)


    @staticmethod
    def symmetry_poe_varymethod_zoomin(data: dict, filename: str):
        theta = 46
        table = defaultdict(list)
        key = '{}-{}'  # {method}-{seed}
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.unitary_theta == theta:
                    table[key.format(method, output.start_seed)].append(output.symmetries)                     # [0] is symmetry
                    table[key.format(method, output.start_seed)].append((1 - np.array(output.scores)) * 100)   # [1] is poe
        fig, ax = plt.subplots(figsize=(22, 16))
        fig.subplots_adjust(left=0.115, right=0.985, top=0.9, bottom=0.14)
        methods = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
        seeds = [0, 1, 2, 3, 4]
        for seed in seeds:
            for method in methods:
                if seed == 0:
                    ax.scatter(table[key.format(method, seed)][0], table[key.format(method, seed)][1], color=Plot.COLOR[method], s=120, label=Plot.METHOD[method])
                else:
                    ax.scatter(table[key.format(method, seed)][0], table[key.format(method, seed)][1], color=Plot.COLOR[method], s=120)
        ax.legend(fontsize=40, facecolor='lightgray', markerscale=2)
        ax.set_xlabel('Symmetry Index', labelpad=20)
        # ax.set_ylabel('$PoE$ (%)', labelpad=20)
        ax.set_xlim([0, 0.082])
        ax.set_ylim([4.34, 4.44])
        xticks = [0, 0.02, 0.04, 0.06, 0.08]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        ax.invert_xaxis()
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=25)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_title('$P()$ (in %) and Symmetry Index', fontsize=60, pad=40)
        fig.savefig(filename)


    @staticmethod
    def symmetry_varymethod_poe(data: dict, filename: str):
        # prepare data
        theta = 46
        seed = 0
        table = defaultdict(list)
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.unitary_theta == theta and output.start_seed == seed:
                    table[method] = (1 - np.array(output.scores)) * 100
        print(table)
        fig, ax = plt.subplots(figsize=(23, 17))
        fig.subplots_adjust(left=0.13, right=0.96, top=0.9, bottom=0.15)
        methods = ['Hill climbing', 'Simulated annealing', 'Genetic algorithm']
        for method in methods:
            ax.plot(table[method][:100], label=method)
        ax.legend()
        ax.set_title(f'Various Methods at $theta={theta}$, $Seed={seed}$', pad=40)
        ax.set_xlabel('Iteration', labelpad=20)
        ax.set_ylabel('PoE (%)', labelpad=20)
        ax.set_ylim([3, 12])
        ax.set_xlim([-0.1, 100])
        ax.tick_params(axis='x', direction='out', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='out', length=10, width=3, pad=15)
        fig.savefig(filename)


    @staticmethod
    def unambiguous_varytheta(data, filename):
        '''vary theta, hill climbing, n-sensors (n=2,3,4,5)
        '''
        methods = ['Hill climbing', 'Simulated annealing', 'Theorem']
        table_2sensor = defaultdict(list)
        table_3sensor = defaultdict(list)
        table_4sensor = defaultdict(list)
        table_5sensor = defaultdict(list)
        table_6sensor = defaultdict(list)
        table_7sensor = defaultdict(list)

        for myinput, output_by_methods in data:
            if myinput.num_sensor == 2:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_2sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 3:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_3sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 4:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_4sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 5:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_5sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 6:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_6sensor[method].append({myinput.unitary_theta: output.error})
            if myinput.num_sensor == 7:
                for method, output in output_by_methods.items():
                    if method in methods:
                        table_7sensor[method].append({myinput.unitary_theta: output.error})

        Y2 = defaultdict(list)
        X2 = [i for i in range(1, 46)] + [i for i in range(135, 180)]
        for method, mylist in table_2sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X2:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y2[method] = y2

        Y3 = defaultdict(list)
        X3 = [i for i in range(1, 61)] + [i for i in range(120, 180)]
        for method, mylist in table_3sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X3:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y3[method] = y2

        Y4 = defaultdict(list)
        X4 = [i for i in range(1, 61)] + [i for i in range(120, 180)]
        for method, mylist in table_4sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X4:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y4[method] = y2

        Y5 = defaultdict(list)
        X5 = [i for i in range(1, 66)] + [i for i in range(115, 180)]
        for method, mylist in table_5sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X5:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y5[method] = y2

        Y6 = defaultdict(list)
        X6 = [i for i in range(1, 66)] + [i for i in range(115, 180)]
        for method, mylist in table_6sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X6:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y6[method] = y2

        Y7 = defaultdict(list)
        X7 = [i for i in range(1, 70)] + [i for i in range(111, 180)]
        for method, mylist in table_7sensor.items():
            y = defaultdict(list)
            for key_val in mylist: # each theta only one experiment
                for theta, error in key_val.items():
                    y[theta] = error
            y2 = []
            for theta in X7:
                if theta in y:
                    y2.append(y[theta])
                elif 180 - theta in y:
                    y2.append(y[180-theta])
                else:
                    raise Exception(f'data missing: theta={theta}')
            Y7[method] = y2

        # step 2: plotting

        fig, ax = plt.subplots(figsize=(30, 18))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.91, bottom=0.12)

        # ax.plot(X2[:45], Y2[methods[0]][:45], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11, label=Plot.METHOD[methods[0]])
        # ax.plot(X2[45:], Y2[methods[0]][45:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        # ax.plot(X2[:45], Y2[methods[1]][:45], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9, label=Plot.METHOD[methods[1]])
        # ax.plot(X2[45:], Y2[methods[1]][45:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X2[:45], Y2[methods[1]][:45], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7, label=Plot.METHOD[methods[2]])
        ax.plot(X2[45:], Y2[methods[1]][45:], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)
        
        # ax.plot(X3[:60], Y3[methods[0]][:60], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        # ax.plot(X3[60:], Y3[methods[0]][60:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        # ax.plot(X3[:60], Y3[methods[1]][:60], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        # ax.plot(X3[60:], Y3[methods[1]][60:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X3[:60], Y3[methods[2]][:60], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)
        ax.plot(X3[60:], Y3[methods[2]][60:], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)

        # ax.plot(X4[:60], Y4[methods[0]][:60], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        # ax.plot(X4[60:], Y4[methods[0]][60:], linestyle=Plot.LINE_STYLE[methods[0]], color=Plot.COLOR[methods[0]], linewidth=11)
        # ax.plot(X4[:60], Y4[methods[1]][:60], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        # ax.plot(X4[60:], Y4[methods[1]][60:], linestyle=Plot.LINE_STYLE[methods[1]], color=Plot.COLOR[methods[1]], linewidth=9)
        ax.plot(X4[:60], Y4[methods[2]][:60], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)
        ax.plot(X4[60:], Y4[methods[2]][60:], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)

        ax.plot(X5[:65], Y5[methods[2]][:65], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)
        ax.plot(X5[65:], Y5[methods[2]][65:], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)

        ax.plot(X6[:65], Y6[methods[2]][:65], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)
        ax.plot(X6[65:], Y6[methods[2]][65:], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)

        ax.plot(X7[:69], Y7[methods[2]][:69], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)
        ax.plot(X7[69:], Y7[methods[2]][69:], linestyle=Plot.LINE_STYLE[methods[2]], color=Plot.COLOR[methods[2]], linewidth=7)

        arrowprops = dict(facecolor='black', width=5, headwidth=20)
        # ax.annotate('2 Sensor', xy=(140.5, 0.15), xytext=(150, 0.15), arrowprops=arrowprops, fontsize=50, va='center')
        # ax.annotate('3 Sensor', xy=(133, 0.25),   xytext=(150, 0.25), arrowprops=arrowprops, fontsize=50, va='center')
        # ax.annotate('4 Sensor', xy=(131, 0.35),  xytext=(100, 0.35), arrowprops=arrowprops, fontsize=50, va='center')

        ax.annotate('2 Sensor', xy=(140, 0.15), xytext=(150, 0.15), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('3/4 Sensor', xy=(132.5, 0.25), xytext=(150, 0.25), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('5/6 Sensor', xy=(133, 0.35), xytext=(150, 0.35), arrowprops=arrowprops, fontsize=50, va='center')
        ax.annotate('7 Sensor', xy=(135, 0.45), xytext=(105, 0.45), arrowprops=arrowprops, fontsize=50, va='center')

        xticks = [i for i in range(0, 181, 15)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{x}' for x in xticks])
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
        ax.set_xlim([0, 180])
        ax.set_ylim([-0.002, 1])
        ax.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax.set_title('Search Heuristics (Unambiguous)', fontsize=65, pad=40)
        ax.legend(ncol=1, loc='upper center',  fontsize=55, handlelength=3.5, edgecolor='black')
        ax.set_xlabel('Theta (degree)', labelpad=15)
        ax.set_ylabel('Probability of Inconclusion (%)', fontsize=60)
        fig.savefig(filename)


    @staticmethod
    def noise_affect_vary_depolar_noise(data, filename):
        # process data
        methods = ['Theorem', 'GHZ', 'Non entangle']
        theta0 = 20
        theta1 = 45
        theta2 = 70
        table0 = defaultdict(list)  # theta0 = 20
        table1 = defaultdict(list)  # theta1 = 45
        table2 = defaultdict(list)  # theta2 = 70
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.noise_type == 'depolar' and myinput.unitary_theta == theta0 and method in methods:
                    table0[method].append((round(myinput.noise_param, 2), output.error))
                if myinput.noise_type == 'depolar' and myinput.unitary_theta == theta1 and method in methods:
                    table1[method].append((round(myinput.noise_param, 2), output.error))
                if myinput.noise_type == 'depolar' and myinput.unitary_theta == theta2 and method in methods:
                    table2[method].append((round(myinput.noise_param, 2), output.error))
        Y0 = defaultdict(list)
        Y1 = defaultdict(list)
        Y2 = defaultdict(list)
        for method in methods:
            table0[method].sort()
            table1[method].sort()
            table2[method].sort()
            Y0[method] = [y * 100 for _, y in table0[method]]  # to percentage
            Y1[method] = [y * 100 for _, y in table1[method]]
            Y2[method] = [y * 100 for _, y in table2[method]]
        X = [x * 100 for x, _ in table0[methods[0]]]           # to percentage
        
        # plotting
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(56, 18))
        fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.19, wspace=0.15)
        for method in methods:
            ax0.plot(X, Y0[method], label=Plot.METHOD[method], color=Plot.COLOR[method])
            ax1.plot(X, Y1[method], label=Plot.METHOD[method], color=Plot.COLOR[method])
            label = '$Corollary$ 1' if method == 'Theorem' else Plot.METHOD[method]
            ax2.plot(X, Y2[method], label=label, color=Plot.COLOR[method])
        # ax0
        ax0.grid()
        ax0.set_xlim([-0.01, 33])
        ax0.set_ylim([0, 100])
        ax0.set_title('$\\theta$ = 20 (degree)', pad=20)
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        xticks = [0, 5, 10, 15, 20, 25, 30, 33]
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f'{x}' for x in xticks])
        ax0.set_xlabel('Probability $p$ (%)', labelpad=30)
        ax0.set_ylabel('Optimal Objective Value $P()$ (%)', fontsize=60, labelpad=20)
        ax0.legend(fontsize=50, loc='upper left')
        ax0.text(15, -24.5, '(a)')
        # ax1
        ax1.grid()
        ax1.set_xlim([-0.01, 33])
        ax1.set_ylim([0, 100])
        ax1.set_title('$\\theta$ = 45 (degree)', pad=20)
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x}' for x in xticks])
        ax1.set_xlabel('Probability $p$ (%)', labelpad=30)
        ax1.legend(fontsize=50, loc='upper left')
        ax1.text(15, -24.5, '(b)')
        # ax2
        ax2.grid()
        ax2.set_xlim([-0.01, 33])
        ax2.set_ylim([0, 100])
        ax2.set_title('$\\theta$ = 70 (degree)', pad=20)
        ax2.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax2.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f'{x}' for x in xticks])
        ax2.set_xlabel('Probability $p$ (%)', labelpad=30)
        ax2.legend(fontsize=50, loc='upper left')
        ax2.text(15, -24.5, '(c)')
        fig.savefig(filename)


    @staticmethod
    def noise_affect_vary_phaseshift_noise(data, filename):
        # process data
        methods = ['Theorem', 'GHZ', 'Non entangle']
        theta0 = 20
        theta1 = 45
        theta2 = 70
        table0 = defaultdict(list)  # theta0 = 20
        table1 = defaultdict(list)  # theta1 = 45
        table2 = defaultdict(list)  # theta2 = 70
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.noise_type == 'phaseshift' and myinput.unitary_theta == theta0 and method in methods:
                    table0[method].append((round(myinput.noise_param, 4), output.error))
                if myinput.noise_type == 'phaseshift' and myinput.unitary_theta == theta1 and method in methods:
                    table1[method].append((round(myinput.noise_param, 4), output.error))
                if myinput.noise_type == 'phaseshift' and myinput.unitary_theta == theta2 and method in methods:
                    table2[method].append((round(myinput.noise_param, 4), output.error))
        Y0 = defaultdict(list)
        Y1 = defaultdict(list)
        Y2 = defaultdict(list)
        for method in methods:
            table0[method].sort()
            table1[method].sort()
            table2[method].sort()
            Y0[method] = [y * 100 for _, y in table0[method]]  # to percentage
            Y1[method] = [y * 100 for _, y in table1[method]]
            Y2[method] = [y * 100 for _, y in table2[method]]
        X = [x * 180 / np.pi for x, _ in table0[methods[0]]]   # to degree
        
        # plotting
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(56, 18))
        fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.19, wspace=0.15)
        for method in methods:
            ax0.plot(X, Y0[method], label=Plot.METHOD[method], color=Plot.COLOR[method])
            ax1.plot(X, Y1[method], label=Plot.METHOD[method], color=Plot.COLOR[method])
            label = '$Corollary$ 1' if method == 'Theorem' else Plot.METHOD[method]
            ax2.plot(X, Y2[method], label=label, color=Plot.COLOR[method])
        # ax0
        ax0.grid()
        ax0.set_xlim([-0.01, 180])
        ax0.set_ylim([0, 100])
        ax0.set_title('$\\theta$ = 20 degree', pad=20)
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        xticks = [0, 30, 60, 90, 120, 150, 180]
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f'{x}' for x in xticks])
        ax0.set_xlabel('Phase Shift $\\epsilon$ (degree)', labelpad=30)
        ax0.set_ylabel('Optimal Objective Value $P()$ (%)', fontsize=60, labelpad=20)
        ax0.legend(fontsize=50, loc='upper left')
        ax0.text(85, -24.5, '(a)')
        # ax1
        ax1.grid()
        ax1.set_xlim([-0.01, 180])
        ax1.set_ylim([0, 100])
        ax1.set_title('$\\theta$ = 45 degree', pad=20)
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x}' for x in xticks])
        ax1.set_xlabel('Phase Shift $\\epsilon$ (degree)', labelpad=30)
        ax1.legend(fontsize=50, loc='upper left')
        ax1.text(85, -24.5, '(b)')
        # ax2
        ax2.grid()
        ax2.set_xlim([-0.01, 180])
        ax2.set_ylim([0, 100])
        ax2.set_title('$\\theta$ = 70 degree', pad=20)
        ax2.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax2.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f'{x}' for x in xticks])
        ax2.set_xlabel('Phase Shift $\\epsilon$ (degree)', labelpad=30)
        ax2.legend(fontsize=50, loc='upper left')
        ax2.text(85, -24.5, '(c)')
        fig.savefig(filename)


    @staticmethod
    def povm_noise_vary_noise(data, filename):
        # process data
        methods = ['Theorem', 'Theorem povm-noise']
        theta = 45
        table0 = defaultdict(list)  # depolar noise
        table1 = defaultdict(list)  # phase shift noise
        for myinput, output_by_methods in data:
            for method, output in output_by_methods.items():
                if myinput.noise_type == 'phaseshift' and myinput.unitary_theta == theta and method in methods:
                    table0[method].append((round(myinput.noise_param, 4), output.error))
                if myinput.noise_type == 'depolar' and myinput.unitary_theta == theta and method in methods:
                    table1[method].append((round(myinput.noise_param, 2), output.error))
        Y0 = defaultdict(list)
        Y1 = defaultdict(list)
        for method in methods:
            table0[method].sort()
            table1[method].sort()
            Y0[method] = [y * 100 for _, y in table0[method]]   # to percentage
            Y1[method] = [y * 100 for _, y in table1[method]]   # to percentage
        X0 = [x * 100 for x, _ in table0[methods[0]]]           # to percentage
        X1 = [x * 100 for x, _ in table1[methods[0]]]           # to percentage
        
        # plotting
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(36, 18))
        fig.subplots_adjust(left=0.1, right=0.975, top=0.93, bottom=0.19, wspace=0.15)
        for method in methods:
            ax0.plot(X0, Y0[method], label=Plot.METHOD[method], color=Plot.COLOR[method])
            ax1.plot(X1, Y1[method], label=Plot.METHOD[method], color=Plot.COLOR[method])
        # ax0
        ax0.grid()
        ax0.set_xlim([-0.01, 180])
        ax0.set_ylim([0, 100])
        ax0.set_title('$\\theta$ = 45 degree', pad=20)
        ax0.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax0.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        xticks = [0, 30, 60, 90, 120, 150, 180]
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f'{x}' for x in xticks])
        ax0.set_xlabel('Phase Shift $\\epsilon$ (degree)', labelpad=30)
        ax0.legend(fontsize=50)
        ax0.set_ylabel('Optimal Objective Value $P()$ (%)', fontsize=60, labelpad=20)
        ax0.text(85, -24, '(a)')
        # ax1
        ax1.grid()
        ax1.set_xlim([-0.01, 33])
        ax1.set_ylim([0, 100])
        ax1.set_title('$\\theta$ = 45 degree', pad=20)
        ax1.tick_params(axis='x', direction='in', length=10, width=3, pad=15)
        ax1.tick_params(axis='y', direction='in', length=10, width=3, pad=15)
        xticks = [0, 5, 10, 15, 20, 25, 30, 33]        
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'{x}' for x in xticks])
        ax1.set_xlabel('Probability $p$ (%)', labelpad=30)
        ax1.legend(fontsize=50)
        ax1.text(15, -24, '(b)')
        fig.savefig(filename)


def vary_theta():
    logs = ['result/12.22.2022/varying_theta_2sensors', 'result/12.22.2022/varying_theta_3sensors', \
            'result/12.22.2022/varying_theta_4sensors', 'result/12.22.2022/varying_theta_5sensors', \
            'result/5.15.2023/varying_theta_2sensors_SA', 'result/5.15.2023/varying_theta_3sensors_SA',\
            'result/5.15.2023/varying_theta_5sensors_SA', 'result/12.26.2022/compare_methods_4sensors']
    data = Logger.read_log(logs)
    filename = 'result/12.22.2022/varying_theta_nsensors.png'
    Plot.vary_theta(data, filename)


def methods_similar():
    # logs1 = ['result/12.22.2022/varying_theta_4sensors', 'result/12.26.2022/compare_methods_4sensors']
    logs2 = ['result/12.26.2022/compare_methods_4sensors']
    # data1 = Logger.read_log(logs1)
    data2 = Logger.read_log(logs2)
    filename = 'result/12.26.2022/compare_methods_similar.png'
    Plot.methods_similar(data2, filename)


def lemma2():
    file_perm = 'result/5.25.2023/lemma2.n{}-t{}.perm.npy'
    file_avg  = 'result/5.25.2023/lemma2.n{}-t{}.avg.npy'
    data = {}
    for n in [3, 4, 5]:                 # number of sensor
        for t in [6, 26, 46, 66, 86]:   # theta
            perm = np.load(file_perm.format(n, t))
            avg  = np.load(file_avg.format(n, t))
            data[f'n{n}-t{t}.perm'] = perm
            data[f'n{n}-t{t}.avg'] = avg
    filename = 'result/5.25.2023/lemma2.png'
    Plot.lemma2(data, filename)


def lemma3():
    file = 'result/1.6.2023/lemma3.n{}.npy'
    data = {}
    for n in range(2, 6):
        y = np.load(file.format(n))
        data[f'n{n}'] = y
    filename = 'result/1.6.2023/lemma3.png'
    Plot.lemma3(data, filename)
    # filename = 'result/1.6.2023/lemma3_tmp.png'
    # Plot.lemma3_tmp(data, filename)


def conjecture():
    logs = ['result/12.31.2022/conjecture_2sensor', 'result/12.31.2022/conjecture_3sensor', 'result/12.31.2022/conjecture_4sensor', 'result/12.31.2022/conjecture_5sensor',
            'result/12.22.2022/varying_theta_2sensors', 'result/12.22.2022/varying_theta_3sensors', 'result/12.22.2022/varying_theta_4sensors', 'result/12.22.2022/varying_theta_5sensors']
    data = Logger.read_log(logs)
    filename = 'result/12.31.2022/conjecture.png'
    Plot.conjecture(data, filename)


def symmetry():
    logs = ['result/5.22.2023/symmetry_theta46', 'result/5.22.2023/symmetry_theta66', 'result/5.22.2023/symmetry_thetas']
    data = Logger.read_log(logs)
    filename = 'result/5.22.2023/symmetry_vary{}.png'
    # Plot.symmetry_varyseed(data, filename.format('seed'))
    # Plot.symmetry_varymethod_poe(data, filename.format('method_poe'))
    
    Plot.symmetry_varymethod(data, filename.format('method'))
    Plot.symmetry_varytheta(data, filename.format('theta'))

    # filename = 'result/5.22.2023/poe_symmetry.png'
    # Plot.symmetry_poe_varymethod(data, filename)
    # filename = 'result/5.22.2023/poe_symmetry_zoomin.png'
    # Plot.symmetry_poe_varymethod_zoomin(data, filename)

def unambiguous_vary_theta():
    logs = ['result/6.16.2023/unambiguous_varytheta_2sen', 'result/6.16.2023/unambiguous_varytheta_3sen', 
            'result/6.16.2023/unambiguous_varytheta_4sen', 'result/6.16.2023/unambiguous_varytheta_5sen',
            'result/6.16.2023/unambiguous_varytheta_6sen', 'result/6.16.2023/unambiguous_varytheta_7sen']
    data = Logger.read_log(logs)
    filename = 'result/6.16.2023/unambiguous_varying_theta2.png'
    Plot.unambiguous_varytheta(data, filename)


def noise_affect():
    # depolar noise
    logs = ['result/11.28.2023/noise_affect_depolar']
    data = Logger.read_log(logs)
    filename = 'result/11.28.2023/noise_affect_depolar.png'
    Plot.noise_affect_vary_depolar_noise(data, filename)

    # phase shift noise
    logs = ['result/11.28.2023/noise_affect_phaseshift']
    data = Logger.read_log(logs)
    filename = 'result/11.28.2023/noise_affect_phaseshift.png'
    Plot.noise_affect_vary_phaseshift_noise(data, filename)


def povm_noise():
    logs = ['result/11.28.2023/povmnoise_depolar', 'result/11.28.2023/povmnoise_phaseshift',
            'result/11.28.2023/noise_affect_depolar', 'result/11.28.2023/noise_affect_phaseshift']
    data = Logger.read_log(logs)
    filename = 'result/11.28.2023/povmnoise.png'
    Plot.povm_noise_vary_noise(data, filename)



'''plotting Eqn. 26 and Eqn. 28 in PRA paper titled Discrete outcome quantum sensor networks
https://caitaozhan.github.io/file/PhysRevA.QuantumSensor.pdf
'''
def pra():
    import math
    def dicke_init_state(N, theta):
        '''Eqn. 26'''
        a = 0.5 * (N - 2)*(1 - math.cos(2*theta))
        b = math.sqrt(N - 1) * abs(math.sin(2*theta))
        return 1/N * (1 + a + b)

    def seperate_init_state(N, theta):
        '''Eqn. 28'''
        a = (N-1)*(N-2)/(2*N) * (1 - math.cos(2*theta))
        b = (N-1) / math.sqrt(N)
        c = math.sin(2*theta) ** 2
        d = 1/N * (1 - math.cos(2*theta)) ** 2
        return 1/N * (1 + a + b * math.sqrt(c + d))

    def n_infinity(theta):
        return 0.5 * (1 - math.cos(2*theta))

    filename = 'result/9.12.2023/pra.png'

    Theta = [i * math.pi / 180 for i in range(0, 46)]
    X = [i for i in range(0, 46)]
    N = 2
    N_2_dicke = [dicke_init_state(N, theta) for theta in Theta]
    N_2_sep   = [seperate_init_state(N, theta) for theta in Theta]
    N = 5
    N_5_dicke = [dicke_init_state(N, theta) for theta in Theta]
    N_5_sep   = [seperate_init_state(N, theta) for theta in Theta]
    N = 10
    N_10_dicke = [dicke_init_state(N, theta) for theta in Theta]
    N_10_sep   = [seperate_init_state(N, theta) for theta in Theta]
    N = 1000000
    N_1000000_dicke = [dicke_init_state(N, theta) for theta in Theta]
    N_1000000_sep   = [seperate_init_state(N, theta) for theta in Theta]
    N_infinity      = [n_infinity(theta) for theta in Theta]

    plt.rc('text', usetex=True)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.8, bottom=0.113)
    ax.plot(X, N_2_dicke,  color='r', linestyle='-',  label='N=2,   Entangled State')
    ax.plot(X, N_5_dicke,  color='g', linestyle='-',  label='N=5,   Entangled State')
    ax.plot(X, N_10_dicke, color='b', linestyle='-',  label='N=10, Entangled State')
    ax.plot(X, N_1000000_dicke, color='black', linestyle='-',  label='N=$10^6$, Entangled State')
    ax.plot(X, N_2_sep,    color='r', linestyle=':',  label='N=2,   Separable State')
    ax.plot(X, N_5_sep,    color='g', linestyle=':',  label='N=5,   Separable State')
    ax.plot(X, N_10_sep,   color='b', linestyle=':',  label='N=10, Separable State')
    ax.plot(X, N_1000000_sep,   color='black', linestyle=':',  label='N=$10^6$, Separable State')
    # ax.plot(X, N_infinity,      color='pink', linestyle=':')
    
    ax.grid()
    ax.legend(fontsize=44, bbox_to_anchor=(1.02, 1.3), ncol=2)
    xticks = [i for i in range(0, 46, 5)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x}' for x in xticks])
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(y * 100)}' for y in yticks])
    ax.tick_params(axis='x', direction='in',  length=10, width=3, pad=15)
    ax.tick_params(axis='y', direction='out', length=10, width=3, pad=15)
    ax.set_xlim([0, 45])
    ax.set_ylim([0, 1.005])
    ax.set_xlabel('Theta (degree)', labelpad=30)
    ax.set_ylabel('Success Probability (\%)')
    fig.savefig(filename)


if __name__ == '__main__':
    # vary_theta()
    # methods_similar()
    # lemma2()
    # conjecture()
    # symmetry()

    # noise_affect()
    povm_noise()


    # pra()

    # not used #
    # unambiguous_vary_theta()
    # lemma3()


