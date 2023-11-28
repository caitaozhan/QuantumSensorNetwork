'''customized logger that deals with input and output
'''

import os

from zmq import DEALER
from input_output import ProblemInput, TheoremOutput, HillclimbOutput, SimulatedAnnealOutput, GeneticOutput

class Logger:

    @staticmethod
    def write_log(log_dir, log_file, myinput, myoutputs):
        if os.path.exists(log_dir) is False:
            os.mkdir(log_dir)
        
        with open(os.path.join(log_dir, log_file), 'a') as f:
            f.write(f'{myinput}\n')
            for output in myoutputs:
                f.write(f'{output}\n')
            f.write('\n')
            f.flush()

    @staticmethod
    def read_log(logs):
        data = []
        for log in logs:
            f = open(log, 'r')
            while True:
                line = f.readline()
                if line == '':
                    break
                myinput = ProblemInput.from_json_str(line)
                output_by_method = {}
                line = f.readline()
                while line != '' and line != '\n':
                    if line.find('Theorem') != -1:
                        output = TheoremOutput.from_json_str(line)
                        output_by_method['Theorem'] = output
                    if line.find('GHZ') != -1:                     # GHZ reuse the TheoremOutput
                        output = TheoremOutput.from_json_str(line)
                        output_by_method['GHZ'] = output
                    if line.find('Non entangle') != -1:
                        output = TheoremOutput.from_json_str(line) # Non entangle reuse the TheoremOutput
                        output_by_method['Non entangle'] = output
                    if line.find('Hill climbing') != -1:
                        output = HillclimbOutput.from_json_str(line)
                        output_by_method['Hill climbing'] = output
                    if line.find('Simulated annealing') != -1:
                        output = SimulatedAnnealOutput.from_json_str(line)
                        output_by_method['Simulated annealing'] = output
                    if line.find('Genetic algorithm') != -1:
                        output = GeneticOutput.from_json_str(line)
                        output_by_method['Genetic algorithm'] = output
                    line = f.readline()
                data.append((myinput, output_by_method))
        return data
