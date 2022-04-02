'''customized logger that deals with input and output
'''

import os

from zmq import DEALER
from input_output import ProblemInput, GuessOutput, HillclimbOutput, Default

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
            f = open(log,'r')
            while True:
                line = f.readline()
                if line == '':
                    break
                myinput = ProblemInput.from_json_str(line)
                output_by_method = {}
                line = f.readline()
                while line != '' and line != '\n':
                    if line.find('Guess') != -1:
                        output = GuessOutput.from_json_str(line)
                        output_by_method['Guess'] = output
                    if line.find('Hill climbing') != -1:
                        output = HillclimbOutput.from_json_str(line)
                        output_by_method['Hill climbing'] = output
                    line = f.readline()
                data.append((myinput, output_by_method))
        return data
