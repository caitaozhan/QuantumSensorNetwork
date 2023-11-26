'''manage the input and output
'''

from dataclasses import dataclass
from typing import List
import json


@dataclass
class Default:
    '''Some default configurations
    '''
    output_dir  = 'result-tmp2'
    output_file = 'foo'
    methods = ['Theory', 'Hill climbing', 'Simulated annealing']
    # some constants
    EPSILON = 1e-8              # the epsilon for zero
    EPSILON_SEMIDEFINITE = 8e-4 # relaxed for semidefinate programming optimal condition checking......
    EPSILON_OPT          = 1e-7 # the epsilon for optimization termination

    # problem input related
    num_sensor    = 2
    unitary_seed  = 2
    method        = 'Theory'
    depolar_noise = 0

    # below are for Hill climbing method
    start_seed = 0            # seed that affect the starting point of the hill climbing
    step_size  = 0.1          # initial step size
    decrease_rate = 0.96      # decrease rate for the step sizes
    realimag_neighbor = False # changing the real part and the imaginary part
    random_neighbor = True    # random neighbor or predefined direction neighbor

    # below are for simulated annealing
    init_step = 0.1                  # initial step size
    max_stuck = 5                    # max stuck in a same temperature
    cooling_rate = 0.96              # the annealing cooling rate
    stepsize_decreasing_rate = 0.96  # the stepsize decreasing rate

    # below are for hill climbing, simulated annealing, genetic algorithm, and particle swarm optimization
    min_iteration = 100   # minimum interation
    eval_metric = 'min error'

    # below are for genetic algorithm
    population_size = 20
    mutation_rate  = 1
    crossover_rate = 1

    # below are for particle swarm optimization
    weight = 0.5   # inertia
    eta1 = 1       # cognative constant
    eta2 = 1       # social constant


@dataclass
class ProblemInput:
    '''encapsulate the problem's input
    '''
    experiment_id: int
    num_sensor: int       # number of sensors
    priors: List[float]   # prior probability for each sensors
    unitary_seed: int     # seed for generating the unitary operator
    unitary_theta: float  # the angle (theta) of the symmetric eigen values
    depolar_noise: float = 0  # the probability of happening X, Y or Z Pauli error

    def __str__(self):
        return self.to_json_str()

    def to_json_str(self):
        '''return json formatting string
        Return:
            str
        '''
        inputdict = {
            'experiment_id': self.experiment_id,
            'num_sensor': self.num_sensor,
            'priors': [round(p, 4) for p in self.priors],
            'unitary_seed': self.unitary_seed,
            'unitary_theta': self.unitary_theta,
            'depolar_noise': self.depolar_noise
        }
        return json.dumps(inputdict)

    @classmethod
    def from_json_str(cls, json_str: str):
        '''init an Input object from json string
        Args:
            json_str -- a string of json
        Return:
            Input
        '''
        indict = json.loads(json_str)
        return cls(indict['experiment_id'], indict['num_sensor'], indict['priors'], indict['unitary_seed'], indict['unitary_theta'], indict.get('depolar_noise', 0))


@dataclass
class TheoremOutput:
    '''encapsulate the Theory (conjecture) method's output or Theorem's output
       Also used for a guessing such as GHZ state, non-entangled uniform superposition state
    '''
    experiment_id: int
    method: str       # 'Theorem', 'GHZ', 'Non-entangle'
    error: float
    success: float
    init_state: str   # the resulting initial state

    def __str__(self):
        return self.to_json_str()

    def to_json_str(self):
        '''return json formatting str
        Return:
            str
        '''
        outputdict = {
            'experiment_id': self.experiment_id,
            'error': self.error,
            'method': self.method,
            'success': self.success,
            'init_state': self.init_state
        }
        return json.dumps(outputdict)

    @classmethod
    def from_json_str(cls, json_str):
        '''init an Theory (conjecture) output object from json string
        Args:
            json_str -- a string of json
        Return:
            TheoryOutput
        '''
        outdict = json.loads(json_str)
        return cls(outdict['experiment_id'], outdict['method'], outdict['error'], outdict['success'], outdict['init_state'])


@dataclass
class HillclimbOutput:
    '''encapsulate the Hill climbing method's information
    '''
    experiment_id: int
    method: str
    error: float
    success: float          # also 1 - error, also the last value of scores
    start_seed: int         # the seed that affects the starting point of hill climbing
    step_size: float        # the initial step size
    decrease_rate: float    # the decrease rate for the step sizes
    min_iteration: int      # the minimum iteration for hill climbing
    real_iteration: int     # number of iterations in reality
    init_state: str         # the initial state found
    scores: List[float]     # the evaluation value of the quantum state at each iteration
    symmetries: List[float] # the symmetry index of the quantum state at each iteration
    runtime: float          # run time
    eval_metric: str        # 'min error' or 'unambiguous'

    def __str__(self):
        return self.to_json_str()

    def to_json_str(self):
        '''return json formatting str
        Return:
            str
        '''
        outputdict = {
            'experiment_id': self.experiment_id,
            'error': self.error,
            'method': self.method,
            'eval_metric': self.eval_metric,
            'min_iteration': self.min_iteration,
            'real_iteration': self.real_iteration,
            'runtime': self.runtime,
            'start_seed': self.start_seed,
            'decrease_rate': self.decrease_rate,
            'success': self.success,
            'step_size': self.step_size,
            'init_state': self.init_state,
            'scores': self.scores,
            'symmetries': self.symmetries if self.symmetries else None
        }
        return json.dumps(outputdict)

    @classmethod
    def from_json_str(cls, json_str):
        '''init a Hill climbing output object from json string
        Args:
            json_str -- a string of json
        Return:
            HillclimbOutput
        '''
        outdict = json.loads(json_str)
        return cls(outdict['experiment_id'], outdict['method'], outdict['error'], outdict['success'], \
                   outdict['start_seed'], outdict['step_size'], outdict['decrease_rate'], \
                   outdict['min_iteration'], outdict['real_iteration'], outdict['init_state'], outdict['scores'], \
                   outdict['symmetries'] if 'symmetries' in outdict else None,\
                   outdict['runtime'], outdict['eval_metric'] if 'eval_metric' in outdict else None)


@dataclass
class SimulatedAnnealOutput:
    '''encapsulate the simulated annealing method's information
    '''
    experiment_id: int
    method: str
    error: float
    success: float       # also 1 - error, the last value of scores
    start_seed: int      # the seed that affects the starting point of simulated annealing
    init_step: float     # the initial step size
    stepsize_decreasing_rate: float # step size decreasing rate
    max_stuck: int       # frozen criteria
    cooling_rate: float  # the decrease in temperature
    min_iteration: int   # the minimum iteration for simulated annealing
    real_iteration: int  # number of iterations in reality
    init_state: str      # the initial state found
    scores: List[float]  # the evaluation value of each evaluation
    symmetries: List[float] # the symmetry index of the quantum state at each iteration
    runtime: float       # run time
    eval_metric: str     # 'min error' or 'unambiguous'

    def __str__(self):
        return self.to_json_str()
    
    def to_json_str(self):
        '''return json formatting str
        Return:
            str
        '''
        outputdict = {
            'experiment_id': self.experiment_id,
            'error': self.error,
            'method': self.method,
            'eval_metric': self.eval_metric,
            'min_iteration': self.min_iteration,
            'real_iteration': self.real_iteration,
            'runtime': self.runtime,
            'start_seed': self.start_seed,
            'cooling_rate': self.cooling_rate,
            'success': self.success,
            'init_step': self.init_step,
            'stepsize_decreasing_rate': self.stepsize_decreasing_rate,
            'max_stuck': self.max_stuck,
            'init_state': self.init_state,
            'scores': self.scores,
            'symmetries': self.symmetries
        }
        return json.dumps(outputdict)

    @classmethod
    def from_json_str(cls, json_str):
        '''init a Simulated annealing output object from json string
        Args:
            json_str -- a string of json
        Return:
            SimulatedAnnealOutput
        '''
        outdict = json.loads(json_str)
        return cls(outdict['experiment_id'], outdict['method'], outdict['error'], outdict['success'], \
                   outdict['start_seed'], outdict['init_step'], outdict['stepsize_decreasing_rate'], outdict['max_stuck'], \
                   outdict['cooling_rate'], outdict['min_iteration'], outdict['real_iteration'], outdict['init_state'], \
                   outdict['scores'], outdict['symmetries'] if 'symmetries' in outdict else None, \
                   outdict['runtime'], outdict['eval_metric'])

@dataclass
class GeneticOutput:
    '''encapsulate the Genetic algorithm method's information
    '''
    experiment_id: int
    method: str
    error: float
    success: float          # also 1 - error, also the last value of scores
    population_size: int    # number of individual solutions in a population
    crossover_rate: float   # probability of happening a crossover when producing childrens
    mutation_rate: float    # probability of happening a mutation on an individual
    start_seed: int         # the seed that affects the starting point of hill climbing
    init_step: float        # the initial amplitude step size
    stepsize_decreaseing_rate: float # the decrease rate for the step sizes
    min_iteration: int      # the minimum iteration for hill climbing
    real_iteration: int     # number of iterations in reality
    init_state: str         # the initial state found
    scores: List[float]     # the evaluation value of each iteration
    symmetries: List[float] # the symmetry index of the quantum state at each iteration
    runtime: float          # run time
    eval_metric: str        # 'min error' or 'unambiguous'

    def __str__(self):
        return self.to_json_str()
    
    def to_json_str(self) -> str:
        '''return json formatting str
        '''
        outputdict = {
            'experiment_id': self.experiment_id,
            'error': self.error,
            'method': self.method,
            'eval_metric': self.eval_metric,
            'population_size': self.population_size,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'min_iteration': self.min_iteration,
            'real_iteration': self.real_iteration,
            'runtime': self.runtime,
            'success': self.success,
            'start_seed': self.start_seed,
            'init_step': self.init_step,
            'stepsize_decreasing_rate': self.stepsize_decreaseing_rate,
            'init_state': self.init_state,
            'scores': self.scores,
            'symmetries': self.symmetries
        }
        return json.dumps(outputdict)

    @classmethod
    def from_json_str(cls, json_str) -> "GeneticOutput":
        '''init a Genetic algorithm output object from json string
        Args:
            json_str -- a string of json
        '''
        outdict = json.loads(json_str)
        return cls(outdict['experiment_id'], outdict['method'], outdict['error'], outdict['success'], \
                   outdict['population_size'], outdict['crossover_rate'], outdict['mutation_rate'], \
                   outdict['start_seed'], outdict['init_step'], outdict['stepsize_decreasing_rate'], outdict['min_iteration'], \
                   outdict['real_iteration'], outdict['init_state'], outdict['scores'], \
                   outdict['symmetries'] if 'symmetries' in outdict else None, outdict['runtime'], outdict['eval_metric'])


@dataclass
class ParticleSwarmOutput:
    '''encapsulate the Particle swarm optimization method's information
    '''
    experiment_id: int
    method: str
    error: float
    success: float          # also 1 - error, also the last value of scores
    population_size: int    # number of individual solutions in a population
    w: float                # inertia weight
    eta1: float             # cognitive constant
    eta2: float             # social constant
    start_seed: int         # the seed that affects the starting point of hill climbing
    init_step: float        # the initial amplitude step size
    min_iteration: int      # the minimum iteration for hill climbing
    real_iteration: int     # number of iterations in reality
    init_state: str         # the initial state found
    scores: List[float]     # the evaluation value of each iteration
    runtime: float          # run time
    eval_metric: str        # 'min error' or 'unambiguous'

    def __str__(self):
        return self.to_json_str()
    
    def to_json_str(self) -> str:
        '''return json formatting str
        '''
        outputdict = {
            'experiment_id': self.experiment_id,
            'error': self.error,
            'method': self.method,
            'eval_metric': self.eval_metric,
            'population_size': self.population_size,
            'w': self.w,
            'eta1': self.eta1,
            'eta2': self.eta2,
            'min_iteration': self.min_iteration,
            'real_iteration': self.real_iteration,
            'runtime': self.runtime,
            'success': self.success,
            'start_seed': self.start_seed,
            'init_step': self.init_step,
            'init_state': self.init_state,
            'scores': self.scores
        }
        return json.dumps(outputdict)
    
    @classmethod
    def from_json_str(cls, json_str) -> "GeneticOutput":
        '''init a Genetic algorithm output object from json string
        Args:
            json_str -- a string of json
        '''
        outdict = json.loads(json_str)
        return cls(outdict['experiment_id'], outdict['method'], outdict['error'], outdict['success'], \
                   outdict['population_size'], outdict['w'], outdict['eta1'], outdict['eta2'], \
                   outdict['start_seed'], outdict['init_step'], outdict['min_iteration'], \
                   outdict['real_iteration'], outdict['init_state'], outdict['scores'], outdict['runtime'], outdict['eval_metric'])
