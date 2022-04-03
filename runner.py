'''run experiments
'''

import time
import subprocess
from subprocess import Popen
from utility import Utility


def set_numsensor_prior(args: list, num_sensor: int, equal: bool):
    args = args.copy()
    args += ['-ns', str(num_sensor)]
    priors = Utility.generate_priors(num_sensor, equal)
    args += ['-p']
    args += [str(p) for p in priors]
    return args

def set_startseed(args: list, start_seed: int):
    args = args.copy()
    args += ['-ss', str(start_seed)]
    return args

def set_log(args, output_dir, output_file):
    args = args.copy()
    args += ['-od', output_dir, '-of', output_file]
    return args


if __name__ == '__main__':
    command = ['python', 'main.py']
    base_args = ["-us", "2", "-m", "Guess", "Hill climbing", "-mi", "50"]

    num_sensor = 3
    equal = False
    task = 2
    output_dir = 'result-tmp'
    output_file = 'vary-prior'

    ps = []
    for i in range(task):
        args = set_numsensor_prior(base_args, num_sensor, equal)
        args = set_startseed(args, 0)
        args = set_log(args, output_dir, output_file)
        print(args)
        ps.append(Popen(command + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    
    while len(ps) > 0:
        new_ps = []
        for p in ps:
            if p.poll() is None:
                new_ps.append(p)
        ps = new_ps
        time.sleep(0.5)

    print('Done!')