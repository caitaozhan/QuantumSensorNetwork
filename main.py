'''The main
'''

import argparse
from qiskit.quantum_info import random_unitary
from optimize_initial_state_custom import OptimizeInitialStateCustom
from optimize_initial_state import OptimizeInitialState
from optimize_initial_state_nonpure import OptimizeInitialStateNonpure
from quantum_noise import DepolarisingChannel, RZNoise
from povm import Povm
from utility import Utility
import time
from input_output import Default, GeneticOutput, ParticleSwarmOutput, ProblemInput, TheoremOutput, HillclimbOutput, SimulatedAnnealOutput
from logger import Logger



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
    parser.add_argument('-dn', '--depolar_noise', action='store_true', help='apply depolarising noise')
    parser.add_argument('-np', '--noise_probability', type=float, nargs=1, default=[Default.noise_probability], help='depolarising noise probability, for single qubit X, Y, or Z')
    parser.add_argument('-pn', '--phaseshift_noise', action='store_true', help='apply phase shift noise')
    parser.add_argument('-ne', '--noise_epsilon', type=float, nargs=1, default=[Default.noise_epsilon], help='phase shift epsilon')
    parser.add_argument('-nst', '--noise_std', type=float, nargs=1, default=[Default.noise_std], help='phase shift std')
    parser.add_argument('-r',  '--repeat', type=int, nargs=1, default=[Default.repeat])

    # below are for hill climbing
    parser.add_argument('-ss', '--start_seed', type=int, nargs=1, default=[Default.start_seed], help='seed that affects the start point of hill climbing')
    parser.add_argument('-st', '--step_size', type=float, nargs=1, default=[Default.step_size], help='step size')
    parser.add_argument('-dr', '--decrease_rate', type=float, nargs=1, default=[Default.decrease_rate], help='decrease rate for the step sizes')

    # below are for simulated annealing
    parser.add_argument('-is', '--init_step', type=float, nargs=1, default=[Default.init_step], help='initial step')
    parser.add_argument('-ms', '--max_stuck', type=int, nargs=1, default=[Default.max_stuck], help='max stuck in a same temperature')
    parser.add_argument('-cr', '--cooling_rate', type=float, nargs=1, default=[Default.cooling_rate], help='the cooling rate')
    parser.add_argument('-sd', '--stepsize_decreasing_rate', type=float, nargs=1, default=[Default.stepsize_decreasing_rate], help='the decreasing rate for stepsize')

    # below are for both hill climbing and simulated annealing
    parser.add_argument('-mi', '--min_iteration', type=int, nargs=1, default=[Default.min_iteration], help='minimum number of iteration in hill climbing')
    parser.add_argument('-em', '--eval_metric', type=str, nargs=1, default=[Default.eval_metric], help='a state is evaluated by min error or unambiguous')

    # below are for both genetic algorithm and pariticle swarm optimization
    parser.add_argument('-ps', '--population_size', type=int, nargs=1, default=[Default.population_size], help='the size of the population, i.e. number of solutions')

    # below are for genetic algorihm
    parser.add_argument('-mu', '--mutation_rate', type=float, nargs=1, default=[Default.mutation_rate], help='the probability of doing mutation once during a offspring production')
    parser.add_argument('-co', '--crossover_rate', type=float, nargs=1, default=[Default.crossover_rate], help='the probability of doing crossover once during a offspring production')

    # below are for particle swarm optimization
    parser.add_argument('-w', '--weight', type=float, nargs=1, default=[Default.weight], help='the velocity inertia')
    parser.add_argument('-e1', '--eta1', type=int, nargs=1, default=[Default.eta1], help='cognitive constant')
    parser.add_argument('-e2', '--eta2', type=int, nargs=1, default=[Default.eta2], help='social constant')

    # below are for theorem when theta < T (temporary putting in experiment_id, may be removed)
    parser.add_argument('-pa', '--partition', type=int, nargs=1, default=[1], help='the partition used to give positive coefficients')

    args = parser.parse_args()
    experiment_id = args.experiment_id[0]
    num_sensor    = args.num_sensor[0]
    priors        = args.priors
    unitary_seed  = args.unitary_seed[0]
    unitary_theta = args.unitary_theta[0]
    methods       = args.methods
    eval_metric   = args.eval_metric[0]
    depolar_noise = args.depolar_noise
    noise_prob    = args.noise_probability[0]
    phaseshift_noise = args.phaseshift_noise
    noise_epsilon = args.noise_epsilon[0]
    noise_std     = args.noise_std[0]
    repeat        = args.repeat[0]

    problem_input = ProblemInput(experiment_id, num_sensor, priors, unitary_seed, unitary_theta)
    if unitary_theta is not None:
        unitary_operator = Utility.generate_unitary_operator(theta=unitary_theta, seed=unitary_seed)
    else:
        # when not specifying the theta, generate a random unitary that has some random thetas
        unitary_operator = random_unitary(dims=2, seed=unitary_seed)
    outputs = []


    if depolar_noise is False and phaseshift_noise is False:
        if "Theorem" in methods:
            partition_i = args.partition[0]
            opt_initstate = OptimizeInitialState(num_sensor)
            opt_initstate.theorem(unitary_operator, unitary_theta, partition_i)
            povm = Povm()
            success = opt_initstate.evaluate(unitary_operator, priors, povm, eval_metric)
            # success = opt_initstate.evaluate_orthogonal(unitary_operator)
            # innerprods = opt_initstate.get_innerproducts(unitary_operator)
            # print(innerprods)
            # symmetry_index = opt_initstate.get_symmetry_index(opt_initstate.state_vector, unitary_operator)
            success = round(success, 7)
            error = round(1-success, 7)
            theorem_output = TheoremOutput(partition_i, opt_initstate.optimize_method, error, success, str(opt_initstate))
            outputs.append(theorem_output)  # Theorem and Guess share the same output format
            
        if "Hill climbing" in methods:
            opt_initstate = OptimizeInitialState(num_sensor)
            start_seed = args.start_seed[0]
            epsilon = Default.EPSILON_OPT
            step_size = [args.step_size[0]] * 2**num_sensor
            decrease_rate = args.decrease_rate[0]
            min_iteration = args.min_iteration[0]
            start_time = time.time()
            scores, symmetries = opt_initstate.hill_climbing(None, start_seed, unitary_operator, priors, epsilon, step_size, \
                                                            decrease_rate, min_iteration, eval_metric)
            runtime = round(time.time() - start_time, 2)
            success = scores[-1]
            error = round(1 - success, 7)
            real_iteration = len(scores) - 1   # minus the initial score, that is not an iteration
            hillclimb_output = HillclimbOutput(experiment_id, opt_initstate.optimize_method, error, success, start_seed, args.step_size[0], \
                                            decrease_rate, min_iteration, real_iteration, str(opt_initstate), scores, symmetries, runtime, eval_metric)
            outputs.append(hillclimb_output)

        if "Hill climbing C" in methods:
            custom_basis = Utility.generate_custombasis(num_sensor, unitary_operator)
            opt_initstate = OptimizeInitialStateCustom(num_sensor, custom_basis)
            start_seed = args.start_seed[0]
            epsilon = Default.EPSILON_OPT
            step_size = [args.step_size[0]] * 2**num_sensor
            decrease_rate = args.decrease_rate[0]
            min_iteration = args.min_iteration[0]
            start_time = time.time()
            scores, symmetries = opt_initstate.hill_climbing(start_seed, unitary_operator, priors, epsilon, step_size, \
                                                            decrease_rate, min_iteration, eval_metric)
            runtime = round(time.time() - start_time, 2)
            success = scores[-1]
            error = round(1 - success, 7)
            real_iteration = len(scores) - 1   # minus the initial score, that is not an iteration
            hillclimb_output = HillclimbOutput(experiment_id, opt_initstate.optimize_method, error, success, start_seed, args.step_size[0], \
                                            decrease_rate, min_iteration, real_iteration, str(opt_initstate), scores, symmetries, runtime, eval_metric)
            outputs.append(hillclimb_output)

        if 'Simulated annealing' in methods:
            opt_initstate = OptimizeInitialState(num_sensor)
            start_seed   = args.start_seed[0]
            init_step    = args.init_step[0]
            stepsize_decreasing_rate = args.stepsize_decreasing_rate[0]
            max_stuck    = args.max_stuck[0]
            cooling_rate = args.cooling_rate[0]
            min_iteration = args.min_iteration[0]
            epsilon = Default.EPSILON_OPT
            start_time   = time.time()
            scores, symmetries = opt_initstate.simulated_annealing_new(start_seed, unitary_operator, priors, init_step, stepsize_decreasing_rate, \
                                                                epsilon, max_stuck, cooling_rate, min_iteration, eval_metric)
            runtime = round(time.time() - start_time, 2)
            success = scores[-1]
            error = round(1 - success, 7)
            real_iteration = len(scores) - 1
            simulateanneal_output = SimulatedAnnealOutput(experiment_id, opt_initstate.optimize_method, error, success, start_seed, init_step, stepsize_decreasing_rate,\
                                                        max_stuck, cooling_rate, min_iteration, real_iteration, str(opt_initstate), scores, symmetries, runtime, eval_metric)
            outputs.append(simulateanneal_output)

        if "Genetic algorithm" in methods:
            opt_initstate = OptimizeInitialState(num_sensor)
            start_seed    = args.start_seed[0]
            epsilon       = Default.EPSILON
            min_iteration = args.min_iteration[0]
            population_size = args.population_size[0]
            crossover_rate  = args.crossover_rate[0]
            mutation_rate   = args.mutation_rate[0]
            init_step       = args.init_step[0]
            stepsize_decreasing_rate = args.stepsize_decreasing_rate[0]
            start_time = time.time()
            scores, symmetries = opt_initstate.genetic_algorithm(start_seed, unitary_operator, priors, epsilon, population_size, mutation_rate, \
                                                                crossover_rate, init_step, stepsize_decreasing_rate, min_iteration, eval_metric)
            success = scores[-1]
            error = round(1 - success, 7)
            runtime = round(time.time() - start_time, 2)
            real_iteration = len(scores) - 1
            genetic_output = GeneticOutput(experiment_id, opt_initstate.optimize_method, error, success, population_size, crossover_rate, mutation_rate, start_seed, \
                                        init_step, stepsize_decreasing_rate, min_iteration, real_iteration, str(opt_initstate), scores, symmetries, runtime, eval_metric)
            outputs.append(genetic_output)

        if "Particle swarm" in methods:
            opt_initstate = OptimizeInitialState(num_sensor)
            start_seed    = args.start_seed[0]
            epsilon       = Default.EPSILON
            min_iteration = args.min_iteration[0]
            population_size = args.population_size[0]
            w    = args.weight[0]
            eta1 = args.eta1[0]
            eta2 = args.eta2[0]
            init_step = args.init_step[0]
            start_time = time.time()
            scores = opt_initstate.particle_swarm_optimization(start_seed, unitary_operator, priors, epsilon, population_size, w, \
                                                            eta1, eta2, init_step, min_iteration, eval_metric)
            success = scores[-1]
            error = round(1 - success, 7)
            runtime = round(time.time() - start_time, 2)
            real_iteration = len(scores) - 1
            particleswarm_output = ParticleSwarmOutput(experiment_id, opt_initstate.optimize_method, error, success, population_size, w, eta1, eta2, start_seed, \
                                                    init_step, min_iteration, real_iteration, str(opt_initstate), scores, runtime, eval_metric)
            outputs.append(particleswarm_output)
    else:
        if depolar_noise:
            quantum_noise = DepolarisingChannel(num_sensor, noise_prob)
            problem_input.noise_type = 'depolar'
            problem_input.noise_param = noise_prob
        else:
            quantum_noise = RZNoise(num_sensor, noise_epsilon, noise_std)
            problem_input.noise_type = 'rz'
            problem_input.noise_param = (noise_epsilon, noise_std)
        # get the measurment operators POVM {E} with final states evolved from a initial state without noise, 
        # then use {E} on the final states evolved from noisy initial state
        if 'Theorem' in methods:
            opt_initstate_nonpure = OptimizeInitialStateNonpure(num_sensor)
            opt_initstate_nonpure.theorem(unitary_operator, unitary_theta)
            povm = opt_initstate_nonpure.get_povm_nonoise(unitary_operator, priors, eval_metric)
            error = opt_initstate_nonpure.evaluate_noise(unitary_operator, priors, povm, quantum_noise, repeat)
            error = round(error, 7)
            success = round(1-error, 7)
            theorem_output = TheoremOutput(experiment_id, 'Theorem', error, success, str(opt_initstate_nonpure))
            outputs.append(theorem_output)
        if 'Non entangle' in methods:
            opt_initstate_nonpure = OptimizeInitialStateNonpure(num_sensor)
            opt_initstate_nonpure.non_entangle(unitary_operator)
            povm = opt_initstate_nonpure.get_povm_nonoise(unitary_operator, priors, eval_metric)
            error = opt_initstate_nonpure.evaluate_noise(unitary_operator, priors, povm, quantum_noise, repeat)
            error = round(error, 7)
            success = round(1-error, 7)
            theorem_output = TheoremOutput(experiment_id, 'Non entangle', error, success, str(opt_initstate_nonpure))
            outputs.append(theorem_output)
        # get the measurment operators POVM {E} with final states evolved from a noisy initial state, 
        # then use {E} on the final states evolved from noisy initial state
        if 'Theorem povm-noise' in methods:
            opt_initstate_nonpure = OptimizeInitialStateNonpure(num_sensor)
            opt_initstate_nonpure.theorem(unitary_operator, unitary_theta)
            povm = opt_initstate_nonpure.get_povm_noise(unitary_operator, priors, eval_metric, quantum_noise)
            error = opt_initstate_nonpure.evaluate_noise(unitary_operator, priors, povm, quantum_noise, repeat)
            error = round(error, 7)
            success = round(1-error, 7)
            theorem_output = TheoremOutput(experiment_id, 'Theorem povm-noise', error, success, str(opt_initstate_nonpure))
            outputs.append(theorem_output)
        if 'Non entangle povm-noise' in methods:
            opt_initstate_nonpure = OptimizeInitialStateNonpure(num_sensor)
            opt_initstate_nonpure.non_entangle(unitary_operator)
            povm = opt_initstate_nonpure.get_povm_noise(unitary_operator, priors, eval_metric, quantum_noise)
            error = opt_initstate_nonpure.evaluate_noise(unitary_operator, priors, povm, quantum_noise, repeat)
            error = round(error, 7)
            success = round(1-error, 7)
            theorem_output = TheoremOutput(experiment_id, 'Non entangle povm-noise', error, success, str(opt_initstate_nonpure))
            outputs.append(theorem_output)


    log_dir = args.output_dir[0]
    log_file = args.output_file[0]
    Logger.write_log(log_dir, log_file, problem_input, outputs)
