import abc
import numpy as np

class dynamics_model_base(abc.ABC):
    '''
    Base class for dynamics models
    '''

    @abc.abstractmethod
    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        '''
        Vehicle dynamics

        Args:
            state (ndarray): the current state of the vehicle
            action (ndarray): the action given
        Returns:
            new_state (ndarray): the predicted state
        '''
        return
    
    @abc.abstractmethod
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        '''
        RK4 Prediction of updated state
        
        Args:
            state (ndarray): the current state of the vehicle
            action (ndarray): the action given
        Returns:
            new_state (ndarray): the predicted state
        '''
        return
    
class KBM(dynamics_model_base):
    '''
    Kinematic bicycle model
    '''
    def __init__(self, L: float, min_throttle: float, max_throttle: float, max_steer: float, dt: float):
        '''
        Args:
            L (float): Length between wheels (m)
            min_throttle (float): Minimum throttle amount (m/s)
            max_throttle (float): Maximum throttle amount (m/s)
            max_steer (float): Maximum steer amount (rad)
            dt (float): Timestep to predict (s)
        '''

        # Load model parameters
        self.L = L
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.max_steer = max_steer
        self.dt = dt

    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        '''
        KBM vehicle dynamics
        '''

        # TODO: Perform dynamics
        new_state = None

        return new_state
    
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        '''
        KBM RK4 prediction
        '''

        # TODO: Perform prediction
        new_state = None

        return new_state