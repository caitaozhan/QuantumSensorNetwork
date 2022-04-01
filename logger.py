'''customized logger that deals with input and output
'''

import os

class Logger:

    def __init__(self, log_dir, log_file):
        self.log_dir = log_dir
        self.log_file = log_file

    def write_log(self, myinput, myoutputs):
        if os.path.exists(self.log_dir) is False:
            os.mkdir(self.log_dir)
        
        with open(os.path.join(self.log_dir, self.log_file), 'a') as f:
            f.write(f'{myinput}\n')
            for output in myoutputs:
                f.write(f'{output}\n')
            f.write('\n')
            f.flush()
