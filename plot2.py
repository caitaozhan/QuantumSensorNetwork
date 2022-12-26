import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from logger import Logger
from optimize_initial_state import OptimizeInitialState
from utility import Utility
from collections import defaultdict

class Plot:

    plt.rcParams['font.size'] = 65
    plt.rcParams['lines.linewidth'] = 5

    @staticmethod
    def vary_theta(data, filename):
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
        
        fig, ax = plt.subplots(1, 1, figsize=(30, 18))
        fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.13)
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


def vary_theta():
    logs = ['result2/12.22.2022/varying_theta_2sensors', 'result2/12.22.2022/varying_theta_3sensors', \
            'result2/12.22.2022/varying_theta_4sensors', 'result2/12.22.2022/varying_theta_5sensors']
    data = Logger.read_log(logs)
    filename = 'result2/12.22.2022/varying_theta_hillclimbing_nsensors.png'
    Plot.vary_theta(data, filename)



if __name__ == '__main__':
    vary_theta()

    
