import abc
import numpy as cp
import cupy as cp

class dynamics_model_base(abc.ABC):
    '''
    Base class for dynamics models
    '''
    dt = None

    @abc.abstractmethod
    def dynamics(self, state: cp.ndarray, action: cp.ndarray) -> cp.ndarray:
        '''
        Vehicle dynamics

        Args:
            state (ndarray): the current state of the vehicle
            action (ndarray): the action given
        Returns:
            state_dot (ndarray): the output derivative
        '''
        return
    
    @classmethod
    def predict_euler(self, state: cp.ndarray, action: cp.ndarray) -> cp.ndarray:
        '''
        Euler Prediction of updated state
        
        Args:
            state (ndarray): the current state of the vehicle
            action (ndarray): the action given
        Returns:
            new_state (ndarray): the predicted state
        '''

        new_state = state + self.dynamics(self, state, action) * self.dt
        return new_state
    
    @classmethod    
    def predict(self, state: cp.ndarray, action: cp.ndarray) -> cp.ndarray:
        '''
        RK4 Prediction of updated state
        
        Args:
            state (ndarray): the current state of the vehicle
            action (ndarray): the action given
        Returns:
            new_state (ndarray): the predicted state
        '''

        # Check dt is set
        assert self.dt is not None
        
        # Perform prediction
        k1 = self.dynamics(self, state, action)
        k2 = self.dynamics(self, state + self.dt/2 * k1, action)
        k3 = self.dynamics(self, state + self.dt/2 * k2, action)
        k4 = self.dynamics(self, state + self.dt * k3, action)

        new_state = state + (self.dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return new_state
    
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
        dynamics_model_base.L = L
        dynamics_model_base.min_throttle = min_throttle
        dynamics_model_base.max_throttle = max_throttle
        dynamics_model_base.max_steer = max_steer
        dynamics_model_base.dt = dt

    def dynamics(self, state: cp.ndarray, action: cp.ndarray) -> cp.ndarray:
        '''
        KBM vehicle dynamics

        Args:
            state (ndarray): Nx3 array (x, y, theta)
            action (ndarray): Nx2 array (v, omega)
        Returns:
            state_dot (ndarray): Nx3 array (x_dot, y_dot, theta_dot)
        '''

        # Check input shapes
        assert state.shape[1] == 3
        assert action.shape[1] == 2
        assert state.shape[0] == action.shape[0]
        
        # Split columns
        _, _, theta = cp.hsplit(state, 3)
        v, omega = cp.hsplit(action, 2)

        # Enforce constraints
        v = cp.clip(v, self.min_throttle, self.max_throttle)
        omega = cp.clip(omega, -self.max_steer, self.max_steer)

        # Perform update
        x_dot = v * cp.cos(theta)
        y_dot = v * cp.sin(theta)
        theta_dot = v * cp.tan(omega) / dynamics_model_base.L

        new_state = cp.concatenate([x_dot, y_dot, theta_dot], axis=1)

        return new_state

# Dummy example driving straight at 1 m/s
if __name__ == "__main__":
    num_trajectories = 2
    steps_trajectories = 10

    model = KBM(0.33, 1.0, 5.0, 0.4189, 0.1)

    v = cp.ones((num_trajectories, steps_trajectories - 1, 1))
    omega = cp.zeros((num_trajectories, steps_trajectories - 1, 1))

    actions = cp.concatenate((v, omega), axis=2)

    trajectories = cp.zeros((num_trajectories, steps_trajectories, 3))
    for i in range(steps_trajectories - 1):
        trajectories[:, i + 1] = model.predict(trajectories[:, i], actions[:, i])