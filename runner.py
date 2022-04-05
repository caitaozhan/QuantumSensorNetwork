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

def set_unitary_theta(args, num_sensor):
    args = args.copy()
    args += ['-ut']
    if num_sensor == 2:
        args += ['45']
    elif num_sensor == 3:
        args += ['180']
    elif num_sensor == 4:
        args += ['120']
    else:
        raise Exception(f'number of sensor {num_sensor} is not considered yet')
    return args

def get_output(p: Popen):
    stderr = p.stderr.readlines()
    if stderr:
        for line in stderr:
            print(line)
    
    stdout = p.stdout.readlines()
    if stdout:
        for line in stdout:
            print(line)


if __name__ == '__main__':
    command = ['python', 'main.py']
    base_args = ["-us", "2", "-m", "Guess", "Hill climbing", "-mi", "50"]

    # base_args = ["-us", "2", "-m", "Guess", "-ut", "60"]

    num_sensor = 3
    equal = True
    task = 1
    output_dir = 'result/4.4.2022'
    output_file = 'foo'

    ps = []
    for i in range(task):
        args = set_numsensor_prior(base_args, num_sensor, equal)
        args = set_unitary_theta(args, num_sensor)
        args = set_startseed(args, 0)
        args = set_log(args, output_dir, output_file)
        print(command + args)
        ps.append(Popen(command + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    
    while len(ps) > 0:
        new_ps = []
        for p in ps:
            if p.poll() is None:
                new_ps.append(p)
            else:
                get_output(p)
        ps = new_ps
        time.sleep(0.5)

    print('Done!')