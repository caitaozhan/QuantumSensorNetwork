'''
Optimizing initial state assuming that the different sensors are separable (non-entangled)
'''

import numpy as np
from quantum_state import QuantumState


class OptimizeInitialStateNonentangled(QuantumState):
    '''Similar to the class OptimizeInitalState, but this class operates on non-entangled states
    '''
    def __init__(self, num_sensor: int):
        super().__init__(num_sensor=num_sensor, state_vector=None)
        self.sensor_states = [QuantumState(num_sensor=1, state_vector=None) for _ in range(num_sensor)]

    def init_random_state(self, seed: int = None):
        for sensor_state in self.sensor_states:
            sensor_state.init_random_state(seed)
        self.set_state_vector()
        if self.check_state() is False:
            raise Exception('Not a valid state vector')

    def set_state_vector(self):
        self.state_vector = 1
        for sensor_state in self.sensor_states:
            if sensor_state.state_vector is not None:
                self.state_vector = np.kron(self.state_vector, sensor_state.state_vector)
            else:
                raise Exception('sensor state is None')

    def __str__(self):
        s = ''
        for i, sensor_state in enumerate(self.sensor_states):
            s += f'Sensor-{i+1}\n{sensor_state}\n'

        parent = super().__str__()
        return s + 'Tensor product:\n' + parent




