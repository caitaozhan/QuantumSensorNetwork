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

def set_unitary_theta(args, theta: int):
    args = args.copy()
    args += ['-ut', str(theta)]
    return args

def set_eval_metric(args, eval_metric: str):
    args = args.copy()
    args += ['-em', eval_metric]
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

    # print('sleeping...')
    # time.sleep(60*60*10)
    # print('start working')

    command = ['python', 'main.py']
    # base_args = ["-us", "2", "-m", "Guess", "Hill climbing", "-mi", "150"]

    base_args = ["-us", "2", "-m", "Simulated annealing", "-mi", "100", "-rn", "True"]

    # 5 sensors experiment, in case it crashes again
    num_sensor  = 3
    equal       = True
    eval_metric = 'min error'  # 'min error' or 'unambiguous'
    # output_dir  = 'result/5.3.2022'
    # output_file = 'varying_theta_5sensor_minerror'
    output_dir  = 'result-tmp'
    output_file = 'foo'
    thetas      = [i for i in range(1, 2, 5)]
    start_seed  = [0, 1]

    # num_sensor  = 2
    # equal       = False
    # eval_metric = 'min error'  # 'min error' or 'unambiguous'
    # output_dir  = 'result/5.3.2022'
    # output_file = 'varying_startseed_2sensor_minerror'
    # thetas      = [1]
    # start_seed  = [i for i in range(10)]


    ps = []
    tasks = []
    for x in thetas:
        for y in start_seed:
            args = set_numsensor_prior(base_args, num_sensor, equal)
            args = set_eval_metric(args, eval_metric)
            args = set_unitary_theta(args, x)
            args = set_startseed(args, y)
            args = set_log(args, output_dir, output_file)
            tasks.append(command + args)
    
    print(f'total number of tasks = {len(tasks)}')
    
    parallel = 2
    ps = []
    while len(tasks) > 0 or len(ps) > 0:
        if len(ps) < parallel and len(tasks) > 0:
            time.sleep(1)
            task = tasks.pop(0)
            print(task, f'{len(tasks)} tasks still in queue')
            ps.append(Popen(task, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            # ps.append(Popen(task))
        else:
            time.sleep(1)
            new_ps = []
            for p in ps:
                if p.poll() is None:
                    new_ps.append(p)
                else:
                    # pass
                    get_output(p)
            ps = new_ps
