'''manage the input and output
'''

from dataclasses import dataclass
from typing import List
import json


@dataclass
class Default:
    '''Some default configurations
    '''
    output_dir  = 'result-tmp'
    output_file = 'foo'
    methods = ['Guess', 'Hill climbing', 'Simulated annealing']
    # some constants
    EPSILON = 1e-8              # the epsilon for zero
    EPSILON_SEMIDEFINITE = 8e-4 # relaxed for semidefinate programming optimal condition checking......
    EPSILON_OPT          = 1e-7 # the epsilon for optimization termination

    # problem input related
    num_sensor   = 2
    unitary_seed = 2
    method       = 'Guess'
    
    # below are for Hill climbing method
    start_seed = 0        # seed that affect the starting point of the hill climbing
    mod_step = 0.1        # initial modulus step size
    amp_step = 0.1        # initial amplitude step size
    decrease_rate = 0.96  # decrease rate for the step sizes

    # below are for simulated annealing
    init_step = 0.1       # initial step size
    max_stuck = 2         # max stuck in a same temperature
    cooling_rate = 0.96   # the annealing cooling rate

    # below are for both hill climbing and simulated annealing
    min_iteration = 100   # minimum interation
    eval_metric = 'min error'



@dataclass
class ProblemInput:
    '''encapsulate the problem's input
    '''
    experiment_id: int
    num_sensor: int       # number of sensors
    priors: List[float]   # prior probability for each sensors
    unitary_seed: int     # seed for generating the unitary operator
    unitary_theta: float  # the angle (theta) of the symmetric eigen values

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
            'unitary_theta': self.unitary_theta
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
        return cls(indict['experiment_id'], indict['num_sensor'], indict['priors'], indict['unitary_seed'], indict['unitary_theta'])


@dataclass
class GuessOutput:
    '''encapsulate the Guess method's output
    '''
    experiment_id: int
    method: str
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
        '''init an Guess output object from json string
        Args:
            json_str -- a string of json
        Return:
            GuessOutput
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
    mod_step: float         # the initial modulus step size
    amp_step: float         # the initial amplitude step size
    decrease_rate: float    # the decrease rate for the step sizes
    min_iteration: int      # the minimum iteration for hill climbing
    real_iteration: int     # number of iterations in reality
    init_state: str         # the initial state found
    scores: List[float]     # the evaluation value of each iteration
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
            'mod_step': self.mod_step,
            'amp_step': self.amp_step,
            'init_state': self.init_state,
            'scores': self.scores
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
                   outdict['start_seed'], outdict['mod_step'], outdict['amp_step'], outdict['decrease_rate'], \
                   outdict['min_iteration'], outdict['real_iteration'], outdict['init_state'], outdict['scores'], \
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
    max_stuck: int       # frozen criteria
    cooling_rate: float  # the decrease in temperature
    min_iteration: int   # the minimum iteration for simulated annealing
    real_iteration: int  # number of iterations in reality
    init_state: str      # the initial state found
    scores: List[float]  # the evaluation value of each evaluation
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
            'max_stuck': self.max_stuck,
            'init_state': self.init_state,
            'scores': self.scores,
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
                   outdict['start_seed'], outdict['init_step'], outdict['max_stuck'], outdict['cooling_rate'],\
                   outdict['min_iteration'], outdict['real_iteration'], outdict['init_state'], outdict['scores'],\
                   outdict['runtime'], outdict['eval_metric'])
