'''run experiments
'''

import time
import numpy as np
import subprocess
from subprocess import Popen
from utility import Utility


def set_depolar_noise(args: list, p: float):
    args += ['-dn', '-np', str(p)]
    return args

def set_amplitude_damping_noise(args: list, gamma: float):
    args += ['-an', '-ga', str(gamma)]
    return args

def set_phase_damping_noise(args: list, gamma: float):
    args += ['-pn', '-ga', str(gamma)]
    return args

def set_numsensor_prior(args: list, num_sensor: int, equal: bool, seed: int):
    args += ['-ns', str(num_sensor)]
    priors = Utility.generate_priors(num_sensor, equal, seed)
    args += ['-p']
    args += [str(p) for p in priors]
    return args

def set_numsensor_non_uniform_prior(args: list, num_sensor: int, seed: int):
    args += ['-ns', str(num_sensor)]
    priors = Utility.generate_non_uniform_priors(num_sensor=num_sensor, seed=seed)
    args += ['-p']
    args += [str(p) for p in priors]
    return args

def set_startseed(args: list, start_seed: int):
    args += ['-ss', str(start_seed)]
    return args

def set_log(args, output_dir, output_file):
    args += ['-od', output_dir, '-of', output_file]
    return args

def set_partition(args, partition_i):
    args += ['-pa', str(partition_i)]
    return args

def set_unitary_theta(args, theta: int):
    args += ['-ut', str(theta)]
    return args

def set_eval_metric(args, eval_metric: str):
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
    # base_args = ["-us", "2", "-m", "Theorem", "Hill climbing",  "-mi", "100"]
    # base_args = ["-us", "2", "-m", "Simulated annealing", "-mi", "100"]
    # base_args = ["-us", "2", "-m", "Genetic algorithm", "Guess", "-mi", "100", "-ps", "32"]
    # base_args = ["-us", "2", "-m", "Particle swarm", "Guess", "-mi", "100", "-ps", "32"]
    # base_args = ["-us", "2", "-m", "Hill climbing C", "-mi", "100"]
    # base_args = ["-us", "2", "-m", "Theorem"]
    # base_args = ["-us", "2", "-m", "Genetic algorithm", "-mi", "100", "-ps", "64"]
    # base_args = ["-us", "2", "-m", "Hill climbing", "Simulated annealing", "Genetic algorithm", "-mi", "50", "-ps", "64"]
    base_args = ["-us", "2", "-m", "Hill climbing", "Simulated annealing", "-mi", "50"]

    num_sensor  = 5
    equal       = False
    eval_metric = 'min error'  # 'min error' or 'unambiguous' or 'computational'
    output_dir  = 'result/12.7.2023'
    output_file = 'nonequal-prior_5sensor'

    # output_dir  = 'result-tmp2'
    # output_file = 'foo'
    thetas      = list(range(65, 66))
    start_seed  = [0]

    ps = []
    tasks = []
    for x in thetas:
        for y in start_seed:
            args = set_unitary_theta(base_args[:], x)
            if equal:
                args = set_numsensor_prior(args, num_sensor, True, seed=y)
            else:
                args = set_numsensor_non_uniform_prior(args, num_sensor, seed=y)
            args = set_eval_metric(args, eval_metric)
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


def main_amplitude_damping_noise():
    command = ['python', 'main.py']
    base_args = ["-us", "2", "-m", "Theorem", "Theorem povm-noise"]

    num_sensor  = 3
    equal       = True
    eval_metric = 'min error'  # 'min error' or 'unambiguous' or 'computational'
    output_dir  = 'result/12.25.2023'
    output_file = 'amplitude_damping'
    thetas      = [45]
    start_seed  = 0

    # experiment: varying gamma
    tasks = []
    gamma = list(np.linspace(0, 1, 101))
    for x in thetas:
        for ga in gamma:
            args = set_amplitude_damping_noise(base_args.copy(), ga)
            args = set_numsensor_prior(args, num_sensor, equal, start_seed)
            args = set_eval_metric(args, eval_metric)
            args = set_unitary_theta(args, x)
            args = set_startseed(args, start_seed)
            args = set_log(args, output_dir, output_file)
            tasks.append(command + args)

    parallel = 2
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


def main_phase_damping_noise():
    command = ['python', 'main.py']
    base_args = ["-us", "2", "-m", "Theorem", "Theorem povm-noise"]

    num_sensor  = 3
    equal       = True
    eval_metric = 'min error'  # 'min error' or 'unambiguous' or 'computational'
    output_dir  = 'result/12.25.2023'
    output_file = 'phase_damping_noise'
    thetas      = [45]
    start_seed  = 0

    # experiment: varying gamma
    tasks = []
    gamma = list(np.linspace(0, 1, 101))
    for x in thetas:
        for ga in gamma:
            args = set_phase_damping_noise(base_args.copy(), ga)
            args = set_numsensor_prior(args, num_sensor, equal, start_seed)
            args = set_eval_metric(args, eval_metric)
            args = set_unitary_theta(args, x)
            args = set_startseed(args, start_seed)
            args = set_log(args, output_dir, output_file)
            tasks.append(command + args)

    parallel = 2
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


def main_depolar_noise():
    command = ['python', 'main.py']
    base_args = ["-us", "2", "-m", "Theorem", "Theorem povm-noise"]

    num_sensor  = 3
    equal       = True
    eval_metric = 'min error'  # 'min error' or 'unambiguous' or 'computational'
    output_dir  = 'result/12.25.2023'
    output_file = 'depolar_noise'
    thetas      = [45]
    start_seed  = 0

    # experiment: varying the noise probability
    tasks = []
    depolar_noise_prob = list(np.linspace(0, 0.75, 76))
    for x in thetas:
        for p in depolar_noise_prob:
            args = set_depolar_noise(base_args.copy(), p)
            args = set_numsensor_prior(args, num_sensor, equal, start_seed)
            args = set_eval_metric(args, eval_metric)
            args = set_unitary_theta(args, x)
            args = set_startseed(args, start_seed)
            args = set_log(args, output_dir, output_file)
            tasks.append(command + args)

    parallel = 2
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
    main_amplitude_damping_noise()
    main_phase_damping_noise()
    main_depolar_noise()


