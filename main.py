'''The main
'''

import numpy as np
import argparse
import copy
from qiskit.quantum_info import random_unitary
from quantum_state import QuantumState
from optimize_initial_state import OptimizeInitialState
from povm import Povm
from utility import Utility
import time
from plot import Plot
from input_output import Default, ProblemInput, GuessOutput, HillclimbOutput, SimulatedAnnealOutput
from logger import Logger


'''Two sensors
'''
def main2():
    print('Two sensors\n')
    seed = 2
    num_sensor = 2
    povm = Povm()
    unitary_operator = random_unitary(dims=2, seed=seed)
    Utility.print_matrix('Unitary matrix', unitary_operator._data)
    optis = OptimizeInitialState(num_sensor)

    # 1: optimize the initial state: Guess
    optis.guess(unitary_operator)
    print(optis)
    # generate the quantums states for SDP, equal prior, then do the SDP
    # priors = [random.uniform(0.1, 0.9)]
    priors = [0.45]
    priors.append(1 - priors[0])
    print('priors:', priors)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(unitary_operator, num_sensor, i)
        optis_copy = copy.deepcopy(optis)
        optis_copy.evolve(evolve_operator)
        quantum_states.append(optis_copy)
    # Optimizing the POVM
    povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
    guess_success = povm.therotical_success
    print(f'SDP error = {povm.theoretical_error}')
    povm.two_state_minerror(quantum_states, priors, debug=False)   # the measurement operators summation is not Identity... But the theoretical error is correct
    print(f'MED error = {povm.theoretical_error}')


    # 2: optimize the initial state: Hill climbing
    mod_step = [0.1]*(2**num_sensor)
    amp_step = [0.1]*(2**num_sensor)
    decrease_rate = 0.96
    iteration = 100
    startState = QuantumState(num_sensor, optis.state_vector)
    scores = optis.hill_climbing(startState=startState, seed=0, unitary_operator=unitary_operator, priors=priors, epsilon=Default.EPSILON_OPT, \
                                 mod_step=mod_step, amp_step=amp_step, decrease_rate=decrease_rate, min_iteration=iteration)
    print(optis)
    quantum_states = []
    for i in range(num_sensor):
        evolve_operator = Utility.evolve_operator(unitary_operator, num_sensor, i)
        optis_copy = copy.deepcopy(optis)
        optis_copy.evolve(evolve_operator)
        quantum_states.append(optis_copy)
    # Optimizing the POVM
    povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
    print(f'SDP error = {povm.theoretical_error}')
    povm.two_state_minerror(quantum_states, priors, debug=False)   # the measurement operators summation is not Identity... But the theoretical error is correct
    print(f'MED error = {povm.theoretical_error}')
    Plot.hillclimbing(scores, guess_success)

    # 3: do random guess for the initial state
    repeat = 1000
    print(f'\nOptimization method is Random.\nRepeat {repeat} times.')
    errors = []
    elapse = 0
    for i in range(repeat):
        # random intial state
        optis.random(i, unitary_operator)
        quantum_states = []
        for i in range(num_sensor):
            evolve_operator = Utility.evolve_operator(unitary_operator, num_sensor, i)
            optis_copy = copy.deepcopy(optis)
            optis_copy.evolve(evolve_operator)
            quantum_states.append(optis_copy)
        # Optimizing the POVM
        start = time.time()
        povm.semidefinite_programming_minerror(quantum_states, priors, debug=False)
        elapse += (time.time() - start)
        errors.append(povm.theoretical_error)
    print(f'min error = {np.min(errors)}\nmax error = {np.max(errors)}\navg error = {np.average(errors)}')
    print(f'SDP time elpase = {elapse:.3}s')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for intial state optimization')
    parser.add_argument('-id', '--experiment_id', type=int, nargs=1, default=[0], help='experiment id number')
    parser.add_argument('-ns', '--num_sensor', type=int, nargs=1, default=[Default.num_sensor], help='number of sensors')
    parser.add_argument('-p',  '--priors', type=float, nargs='+', default=None, help='the prior probability for sensors')
    parser.add_argument('-us', '--unitary_seed', type=int, nargs=1, default=[Default.unitary_seed], help='the seed that affect the unitary operator')
    parser.add_argument('-ut', '--unitary_theta', type=float, nargs=1, default=[None], help='the angle theta of the eigen values')
    parser.add_argument('-m',  '--methods', type=str, nargs='+', default=[Default.method], help='the method for finding the initial state')
    parser.add_argument('-od', '--output_dir', type=str, nargs=1, default=[Default.output_dir], help='output directory')
    parser.add_argument('-of', '--output_file', type=str, nargs=1, default=[Default.output_file], help='output file')

    # below are for hill climbing
    parser.add_argument('-ss', '--start_seed', type=int, nargs=1, default=[Default.start_seed], help='seed that affects the start point of hill climbing')
    parser.add_argument('-ms', '--mod_step', type=float, nargs=1, default=[Default.mod_step], help='step size for modulus')
    parser.add_argument('-as', '--amp_step', type=float, nargs=1, default=[Default.amp_step], help='initial step size for amplitude')
    parser.add_argument('-dr', '--decrease_rate', type=float, nargs=1, default=[Default.decrease_rate], help='decrease rate for the step sizes')

    # below are for simulated annealing
    parser.add_argument('-is', '--init_step', type=float, nargs=1, default=[Default.init_step], help='initial step')
    parser.add_argument('-st', '--max_stuck', type=int, nargs=1, default=[Default.max_stuck], help='max stuck in a same temperature')
    parser.add_argument('-cr', '--cooling_rate', type=float, nargs=1, default=[Default.cooling_rate], help='the cooling rate')

    # below are for both hill climbing and simulated annealing
    parser.add_argument('-mi', '--min_iteration', type=int, nargs=1, default=[Default.min_iteration], help='minimum number of iteration in hill climbing')
    parser.add_argument('-em', '--eval_metric', type=str, nargs=1, default=[Default.eval_metric], help='a state is evaluated by min error or unambiguous')


    args = parser.parse_args()
    experiement_id = args.experiment_id[0]
    num_sensor     = args.num_sensor[0]
    priors         = args.priors
    unitary_seed   = args.unitary_seed[0]
    unitary_theta  = args.unitary_theta[0]
    methods        = args.methods
    eval_metric    = args.eval_metric[0]

    problem_input = ProblemInput(experiement_id, num_sensor, priors, unitary_seed, unitary_theta)
    opt_initstate = OptimizeInitialState(num_sensor)
    if unitary_theta:
        unitary_operator = Utility.generate_unitary_operator(theta=unitary_theta, seed=unitary_seed)
    else:
        # when not specifying the theta, generate a random unitary that has some random thetas
        unitary_operator = random_unitary(dims=2**num_sensor, seed=unitary_seed)
    povm = Povm()
    outputs = []

    if "Guess" in methods:
        opt_initstate.guess(unitary_operator)
        success = opt_initstate.evaluate(unitary_operator, priors, povm, eval_metric)
        success = round(success, 6)
        error = round(1-success, 6)
        guess_output = GuessOutput(experiement_id, opt_initstate.optimize_method, error, success, str(opt_initstate))
        outputs.append(guess_output)

    if "Hill climbing" in methods:
        start_seed = args.start_seed[0]
        epsilon = Default.EPSILON_OPT
        mod_step = [args.mod_step[0]] * 2**num_sensor
        amp_step = [args.amp_step[0]] * 2**num_sensor
        decrease_rate = args.decrease_rate[0]
        min_iteration = args.min_iteration[0]
        start_time = time.time()
        scores = opt_initstate.hill_climbing(None, start_seed, unitary_operator, priors, epsilon, \
                                             mod_step, amp_step, decrease_rate, min_iteration, eval_metric)
        runtime = round(time.time() - start_time, 2)
        success = scores[-1]
        error = round(1 - success, 6)
        real_iteration = len(scores) - 1   # minus the initial score, that is not an iteration
        hillclimb_output = HillclimbOutput(experiement_id, opt_initstate.optimize_method, error, success, start_seed, args.mod_step[0], \
                                           args.amp_step[0], decrease_rate, min_iteration, real_iteration, str(opt_initstate), scores, runtime, eval_metric)
        outputs.append(hillclimb_output)

    if 'Simulated annealing' in methods:
        start_seed   = args.start_seed[0]
        init_step    = args.init_step[0]
        max_stuck    = args.max_stuck[0]
        cooling_rate = args.cooling_rate[0]
        min_iteration = args.min_iteration[0]
        epsilon = Default.EPSILON_OPT
        start_time   = time.time()
        scores = opt_initstate.simulated_annealing(start_seed, unitary_operator, priors, init_step, epsilon, \
                                                   max_stuck, cooling_rate, min_iteration, eval_metric)
        runtime = round(time.time() - start_time, 2)
        success = scores[-1]
        error = round(1 - success, 6)
        real_iteration = len(scores) - 1
        simulateanneal_output = SimulatedAnnealOutput(experiement_id, opt_initstate.optimize_method, error, success, start_seed, init_step,\
                                                      max_stuck, cooling_rate, min_iteration, real_iteration, str(opt_initstate), scores, runtime, eval_metric)
        outputs.append(simulateanneal_output)

    log_dir = args.output_dir[0]
    log_file = args.output_file[0]
    Logger.write_log(log_dir, log_file, problem_input, outputs)
