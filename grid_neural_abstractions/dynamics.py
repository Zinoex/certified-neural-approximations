from .translators import TorchTranslator, NumpyTranslator
import numpy as np
import torch
class DynamicalSystem:
    """Base class for dynamical systems."""
    
    def __init__(self):
        self.input_dim = None  # State dimension
        self.output_dim = None  # Derivative dimension
        self.input_domain = None  # Domain for each input dimension [(min_1, max_1), ..., (min_n, max_n)]
    
    def __call__(self, x, translator=None):
        """
        Compute the dynamics for the system.
        
        Args:
            x: The state tensor with shape [input_dim, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            The derivatives of the system with shape [output_dim, batch_size]
        """
        if translator is None and isinstance(x, np.ndarray):
            # Use NumpyTranslator if x is a NumPy array
            translator = NumpyTranslator()
        else:
            translator = TorchTranslator()
        
        return self.compute_dynamics(x, translator)
    
    def compute_dynamics(self, x, translator):
        """
        Compute the dynamics for the system.
        
        Args:
            x: The state tensor with shape [input_dim, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            The derivatives of the system with shape [output_dim, batch_size]
        """
        raise NotImplementedError("Subclasses must implement compute_dynamics")
    
    
class VanDerPolOscillator(DynamicalSystem):
    """A class representing the Van der Pol oscillator dynamics."""

    def __init__(self, mu=1.0):
        super().__init__()
        # Parameter for the Van der Pol oscillator
        self.mu = mu
        self.input_dim = 2  # Van der Pol oscillator state dimension
        self.output_dim = 2  # Van der Pol oscillator derivative dimension
        self.input_domain = [(-3.0, 3.0), (-3.0, 3.0)]  # Typical domain for Van der Pol oscillator

    def compute_dynamics(self, x, translator):
        """
        Compute Van der Pol dynamics.
        
        Args:
            x: Input tensor with shape [2, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            Tensor of shape [2, batch_size] with the dynamics
        """
        dx1 = x[1]
        dx2 = self.mu * (1 - translator.pow(x[0], 2)) * x[1] - x[0]

        return translator.stack([dx1, dx2])

    def compute_lipschitz_constant(self, R):
        """
        Compute the Lipschitz constant for the van der Pol oscillator.
        Using the Jacobian matrix maximum eigenvalue over the domain.

        Args:
            R: The radius of the domain

        Returns:
            The Lipschitz constant for the dynamics
        """
        # For Van der Pol, the Jacobian is:
        # [ 0,  1 ]
        # [-1 - 2*mu*x₁*x₂, mu*(1-x₁²)]
        # The maximum norm occurs at the boundary of the domain
        # Maximum eigenvalue can be bounded by:
        L1 = 1  # From the first row
        L2 = max(
            1 + 2 * self.mu * R * R, self.mu * (1 + R**2)
        )  # From the second row
        return max(L1, L2)

    def max_gradient_norm(self, c, r):
        """
        Compute the maximum for the van der Pol oscillator.
        Using the Jacobian matrix maximum eigenvalue over the domain.

        :param c: The state vector
        :param r: The (hyperrectangular) radius of the domain

        :return: The maximum gradient norm
        """
        import torch
        # For Van der Pol, the Jacobian is:
        # [ 0,  1 ]
        # [-1 - 2*mu*x₁*x₂, mu*(1-x₁²)]

        corners = [
            c + r * torch.tensor([1, 1]),
            c + r * torch.tensor([-1, 1]),
            c + r * torch.tensor([1, -1]),
            c + r * torch.tensor([-1, -1]),
        ]

        # From the first row
        L11 = 0
        L12 = 1

        # From the second row
        L21 = max(*[
            abs(-1 - 2 * self.mu * corner[0] * corner[1]) for corner in corners
        ])
        L22 = self.mu * (1 + max(abs(c[0] + r[0]), abs(c[0] - r[0]))**2)

        L = torch.tensor([[L11, L12], [L21, L22]])

        return L


class Quadcopter(DynamicalSystem):
    """A class representing the 10D dynamics of a quadcopter."""

    def __init__(
        self,
        mass=1.0,
        gravity=9.81,
        arm_length=0.2,
        moment_inertia_x=0.01,
        moment_inertia_y=0.01,
        moment_inertia_z=0.02,
    ):
        super().__init__()
        # Parameters for the quadcopter model
        self.mass = mass
        self.gravity = gravity
        self.arm_length = arm_length
        self.moment_inertia_x = moment_inertia_x
        self.moment_inertia_y = moment_inertia_y
        self.moment_inertia_z = moment_inertia_z

        self.input_dim = 3  # 3D state: roll, pitch, yaw
        self.output_dim = 3  # 3D derivatives
        self.input_domain = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]  # Typical domain for angles in radians

    def compute_dynamics(self, x, translator):
        """
        Compute quadcopter dynamics.
        
        Args:
            x: Input tensor with shape [3, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            Tensor of shape [3, batch_size] with the dynamics
        """
        # Extract state variables
        # Orientation: roll, pitch, yaw
        roll, pitch, yaw = x[0], x[1], x[2]

        # Simplified thrust and control inputs (can be replaced with actual control inputs)
        # Here using a hover thrust and small attitude corrections
        thrust = self.mass * self.gravity

        # Velocity derivatives (from forces)
        # Simplified model assuming small angles and considering yaw
        dvx = (
            (
                translator.sin(pitch) * translator.cos(yaw)
                + translator.sin(roll) * translator.cos(pitch) * translator.sin(yaw)
            )
            * thrust
            / self.mass
        )
        dvy = (
            (
                translator.sin(pitch) * translator.sin(yaw)
                - translator.sin(roll) * translator.cos(pitch) * translator.cos(yaw)
            )
            * thrust
            / self.mass
        )
        dvz = (
            translator.cos(roll) * translator.cos(pitch) * thrust
        ) / self.mass - self.gravity

        # Combine all derivatives
        derivatives = translator.stack([dvx, dvy, dvz])

        return derivatives


class WaterTank(DynamicalSystem):
    """Water Tank dynamical system as described in equation (13)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.input_domain = [(0.1, 10.0)]  # Water level should be positive
        
    def compute_dynamics(self, x, translator):
        # ẋ = 1.5 - √x
        return translator.stack([1.5 - translator.sqrt(x[0])])


class JetEngine(DynamicalSystem):
    """Jet Engine dynamical system as described in equation (14)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.input_domain = [(-2.0, 2.0), (-2.0, 2.0)]  # Typical domain for jet engine state variables
        
    def compute_dynamics(self, x, translator):
        # ẋ = -y - 1.5x² - 0.5x³ - 0.1
        # ẏ = 3x - y
        dx = -x[1] - 1.5 * translator.pow(x[0], 2) - 0.5 * translator.pow(x[0], 3) - 0.1
        dy = 3 * x[0] - x[1]
        
        return translator.stack([dx, dy])


class SteamGovernor(DynamicalSystem):
    """Steam Governor dynamical system as described in equation (15)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 3
        self.output_dim = 3
        self.input_domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]  # Typical domain for steam governor
        
    def compute_dynamics(self, x, translator):
        # ẋ = y
        # ẏ = z² sin(x) cos(x) - sin(x) - 3y
        # ż = -(cos(x) - 1)
        
        dx = x[1]
        dy = translator.pow(x[2], 2) * translator.sin(x[0]) * translator.cos(x[0]) - translator.sin(x[0]) - 3 * x[1]
        dz = -(translator.cos(x[0]) - 1)
        
        return translator.stack([dx, dy, dz])


class Exponential(DynamicalSystem):
    """Exponential dynamical system as described in equation (16)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.input_domain = [(-1.5, 1.5), (-1.0, 1.0)]  # Restricted domain to avoid extremely large values
        
    def compute_dynamics(self, x, translator):
        # ẋ = -sin(exp(y³ + 1)) - y²
        # ẏ = -x
        
        dx = -translator.sin(translator.exp(translator.pow(x[1], 3) + 1)) - translator.pow(x[1], 2)
        dy = -x[0]

        return translator.stack([dx, dy])


class NonLipschitzVectorField1(DynamicalSystem):
    """Non-Lipschitz Vector Field 1 (NL1) dynamical system as described in equation (17)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.input_domain = [(-1.0, 1.0), (-1.0, 1.0)]  # Typical domain for analysis
        
    def compute_dynamics(self, x, translator):
        # ẋ = y
        # ẏ = √|x|
        
        dx = x[1]
        dy = translator.sqrt(translator.abs(x[0]))
        
        return translator.stack([dx, dy])


class NonLipschitzVectorField2(DynamicalSystem):
    """Non-Lipschitz Vector Field 2 (NL2) dynamical system as described in equation (18)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.input_domain = [(-1.0, 1.0), (-1.0, 1.0)]  # Typical domain for analysis
        
    def compute_dynamics(self, x, translator):
        # ẋ = x² + y
        # ẏ = (x²)^(1/3) - x
        
        dx = translator.pow(x[0], 2) + x[1]
        dy = translator.pow(translator.pow(x[0], 2), 1/3) - x[0]
        
        return translator.stack([dx, dy])


class NonlinearOscillator(DynamicalSystem):
    """A nonlinear 1D oscillator system with cubic and sine terms."""
    
    def __init__(self, linear_coeff=1.0, cubic_coeff=0.5, sine_coeff=0.3):
        super().__init__()
        # Parameters for the nonlinear terms
        self.linear_coeff = linear_coeff
        self.cubic_coeff = cubic_coeff
        self.sine_coeff = sine_coeff
        self.input_dim = 1  # 1D system
        self.output_dim = 1  # 1D output
        self.input_domain = [(-3.0, 3.0)]  # Typical domain for oscillator
    
    def compute_dynamics(self, x, translator):
        """
        Compute nonlinear oscillator dynamics.
        
        Args:
            x: Input tensor with shape [1, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            Tensor of shape [1, batch_size] with the dynamics
        """
        # ẋ = -linear_coeff * x - cubic_coeff * x³ + sine_coeff * sin(x)
        dx = -self.linear_coeff * x[0] - self.cubic_coeff * translator.pow(x[0], 3) + self.sine_coeff * translator.sin(x[0])
        
        return translator.stack([dx])
