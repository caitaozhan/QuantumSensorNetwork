'''run experiments
'''

import time
import numpy as np
import subprocess
from subprocess import Popen
from utility import Utility


def set_depolar_noise(args: list, p: float):
    args = args.copy()
    args += ['-dn', '-np', str(p)]
    return args

def set_phaseshift_noise(args: list, t: float):
    args = args.copy()
    args += ['-pn', '-nt', str(t)]
    return args

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

def set_partition(args, partition_i):
    args = args.copy()
    args += ['-pa', str(partition_i)]
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


def main():
    # print('sleeping...')
    # time.sleep(60*60*10)
    # print('start working')

    command = ['python', 'main.py']
    base_args = ["-us", "2", "-m", "Theorem", "Hill climbing",  "-mi", "100"]
    # base_args = ["-us", "2", "-m", "Simulated annealing", "-mi", "100"]
    # base_args = ["-us", "2", "-m", "Hill climbing (NE)", "Guess", "-mi", "100"]
    # base_args = ["-us", "2", "-m", "Genetic algorithm", "Guess", "-mi", "100", "-ps", "32"]
    # base_args = ["-us", "2", "-m", "Particle swarm", "Guess", "-mi", "100", "-ps", "32"]

    # base_args = ["-us", "2", "-m", "Hill climbing C", "-mi", "100"]
    # base_args = ["-us", "2", "-m", "Theorem"]
    # base_args = ["-us", "2", "-m", "Genetic algorithm", "-mi", "100", "-ps", "64"]
    # base_args = ["-us", "2", "-m", "Hill climbing", "Simulated annealing", "Genetic algorithm", "-mi", "100", "-ps", "32"]
    # base_args = ["-us", "2", "-m", "Simulated annealing", "-mi", "100"]

    num_sensor  = 3
    equal       = False
    eval_metric = 'min error'  # 'min error' or 'unambiguous' or 'computational'
    output_dir  = 'result/11.25.2023'
    output_file = 'nonequal-prior'
    # output_dir  = 'result-tmp2'
    # output_file = 'foo'
    thetas      = [i for i in range(70, 90)]
    # thetas      = [86]
    # start_seed  = list(range(5))
    start_seed  = [0]

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
    
    parallel = 1
    ps = []
    while len(tasks) > 0 or len(ps) > 0:
        if len(ps) < parallel and len(tasks) > 0:
            task = tasks.pop(0)
            print(task, f'{len(tasks)} tasks still in queue')
            ps.append(Popen(task, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            # ps.append(Popen(task))
        else:
            time.sleep(0.05)
            new_ps = []
            for p in ps:
                if p.poll() is None:
                    new_ps.append(p)
                else:
                    # pass
                    get_output(p)
            ps = new_ps


def main_noise():
    command = ['python', 'main.py']
    # base_args = ["-us", "2", "-m", "Theorem", "GHZ", "Non entangle"]
    base_args = ["-us", "2", "-m", "Theorem povm-noise"]
    # base_args = ["-us", "2", "-m", "Theorem povm-noise", "GHZ povm-noise", "Non entangle povm-noise"]

    num_sensor  = 3
    equal       = True
    eval_metric = 'min error'  # 'min error' or 'unambiguous' or 'computational'
    output_dir  = 'result/11.29.2023'
    # output_file = 'noise_affect_depolar'
    # output_file = 'noise_affect_phaseshift'
    # output_file = 'povmnoise_depolar'
    output_file = 'povmnoise_phaseshift'
    # output_dir  = 'result-tmp2'
    # output_file = 'foo'
    thetas      = [45]
    start_seed  = 0
    # depolar_noise_prob = list(np.linspace(0, 0.33, 34))
    phaseshift_theta = list(np.linspace(0, np.pi, 181))

    tasks = []

    # for x in thetas:
    #     for p in depolar_noise_prob:
    #         args = set_depolar_noise(base_args, p)
    #         args = set_numsensor_prior(args, num_sensor, equal)
    #         args = set_eval_metric(args, eval_metric)
    #         args = set_unitary_theta(args, x)
    #         args = set_startseed(args, start_seed)
    #         args = set_log(args, output_dir, output_file)
    #         tasks.append(command + args)
    
    for x in thetas:
        for t in phaseshift_theta:
            args = set_phaseshift_noise(base_args, t)
            args = set_numsensor_prior(args, num_sensor, equal)
            args = set_eval_metric(args, eval_metric)
            args = set_unitary_theta(args, x)
            args = set_startseed(args, start_seed)
            args = set_log(args, output_dir, output_file)
            tasks.append(command + args)

    parallel = 6
    print(f'total number of tasks = {len(tasks)}, parallel cores = {parallel}')
    
    ps = []
    while len(tasks) > 0 or len(ps) > 0:
        if len(ps) < parallel and len(tasks) > 0:
            task = tasks.pop(0)
            print(task, f'{len(tasks)} tasks still in queue')
            ps.append(Popen(task, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            # ps.append(Popen(task))
        else:
            time.sleep(0.05)
            new_ps = []
            for p in ps:
                if p.poll() is None:
                    new_ps.append(p)
                else:
                    # pass
                    get_output(p)
            ps = new_ps


if __name__ == '__main__':
    # main()
    main_noise()


