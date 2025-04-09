import abc
import numpy as np

class dynamics_model_base(abc.ABC):
    '''
    Base class for dynamics models
    '''

    @abc.abstractmethod
    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        '''
        Predict future state given current state and action

        Args:
            state: the current state of the vehicle
            action: the action given
        Returns:
            new_state: the predicted state
        '''
        return
    
class KBM(dynamics_model_base):
    '''
    Kinematic bicycle model
    '''
    def __init__(self):
        # TODO: Load model parameters
        return

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        '''
        Predict future state given current state and action under KBM
        '''

        # TODO: Perform dynamics
        new_state = None

        return new_state