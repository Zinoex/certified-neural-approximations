import math
import numpy as np
from .translators import NumpyTranslator, TorchTranslator
import torch


class DynamicalSystem:
    """Base class for dynamical systems."""
    
    def __init__(self):
        self.input_dim = None  # State dimension
        self.output_dim = None  # Derivative dimension
        self.input_domain = None  # Domain for each input dimension [(min_1, max_1), ..., (min_n, max_n)]
        self.hidden_sizes = None
        self.delta = None
        self.epsilon = None
        self.system_name = None  # Name of the system
    
    def __call__(self, x, translator=None):
        """
        Compute the dynamics for the system.
        
        Args:
            x: The state tensor with shape [input_dim, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            The derivatives of the system with shape [output_dim, batch_size]
        """
        if translator is None:
            if isinstance(x, np.ndarray):
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
        self.hidden_sizes = [64, 64, 64]
        self.delta = np.array([0.75, 1.5])  # Domain size for the input
        self.epsilon = 0.21   
        self.system_name = "VanDerPolOscillator"

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
    """A class representing the dynamics of a quadcopter based on a full mathematical model.
    
    The state vector includes:
    - Orientation angles (x₃)
    - Angular velocity (x₄)
    
    Control inputs are the squared angular velocities of the rotors (γᵢ = ω²ᵢ).
    """

    def __init__(
        self,
        mass=1.0,
        gravity=9.81,
        arm_length=0.2,
        b=0.01,
        k=0.01,
        moment_inertia_x=0.01,
        moment_inertia_y=0.01,
        moment_inertia_z=0.02,
    ):
        super().__init__()
        # Parameters for the quadcopter model
        self.mass = mass
        self.gravity = gravity
        self.arm_length = arm_length
        self.I_xx = moment_inertia_x  # Moment of inertia around x-axis
        self.I_yy = moment_inertia_y  # Moment of inertia around y-axis
        self.I_zz = moment_inertia_z  # Moment of inertia around z-axis

        self.b = b # Drag coefficient (assumed small for simplicity)
        self.k = k # Thrust coefficient (assumed small for simplicity)
        
        self.orientation = False
        if self.orientation:
            # Define state dimension for orientation and angular velocity only (reduced model)
            self.input_dim = 10  # orientation(3) + angular velocity(3) + rotor speeds(4)
            # We output derivatives for these states
            self.output_dim = 6  # derivatives of orientation(3) + angular velocity(3)
            # Typical domains for each state dimension
            self.input_domain = [
                # Orientation (roll, pitch, yaw)
                (-0.5, 0.5), (-0.2, 0.2), (-np.pi, np.pi),
                # Angular velocity (ωx, ωy, ωz)
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                # Squared angular velocities of rotors (γ1, γ2, γ3, γ4)
                (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)
            ]
            self.delta = np.array([
                0.1, 0.1, 0.1,   # Orientation
                1.0, 1.0, 1.0,  # Angular velocity
                5.0, 5.0, 5.0, 5.0   # Squared angular velocities of rotors
            ])
        else:
            # Define state dimension for orientation and angular velocity only (reduced model)
            self.input_dim = 7  # angular velocity(3) + rotor speeds(4)
            # We output derivatives for these states
            self.output_dim = 3  # derivatives of angular velocity(3)
            # Typical domains for each state dimension
            self.input_domain = [
                # Angular velocity (ωx, ωy, ωz)
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                # Squared angular velocities of rotors (γ1, γ2, γ3, γ4)
                (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)
            ]
            self.delta = np.array([
                1.0, 1.0, 1.0,  # Angular velocity
                5.0, 5.0, 5.0, 5.0   # Squared angular velocities of rotors
            ])
        
        self.hidden_sizes = [128, 128]
        self.epsilon = 0.15
        self.system_name = "QuadcopterFull"

    def compute_dynamics(self, x, translator):
        """
        Compute the quadcopter dynamics based on the full mathematical model.
        
        Args:
            x: Input tensor with shape [10, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            Tensor of shape [6, batch_size] with the derivatives
        """
        # Angular velocity (ωx, ωy, ωz)
        gamma = x[3:7]  # Squared angular velocities of the rotors
        
        # Orientation derivative from angular velocities
        # This is the inverse of the transformation matrix in the image
        # Computing Euler angle rates from body-frame angular velocities
        omega_x = x[0]
        omega_y = x[1]
        omega_z = x[2]

        # Extract state variables
        # Orientation (roll, pitch, yaw)
        if self.orientation:
            roll, pitch, yaw = x[7], x[8], x[9]

            # Rotation matrix terms
            sin_roll = translator.sin(roll)
            cos_roll = translator.cos(roll)
            cos_pitch = translator.cos(pitch)
            tan_pitch = translator.tan(pitch)

            droll = omega_x + sin_roll * tan_pitch * omega_y + cos_roll * tan_pitch * omega_z
            dpitch = cos_roll * omega_y - sin_roll * omega_z
            dyaw = (sin_roll / cos_pitch) * omega_y + (cos_roll / cos_pitch) * omega_z
        
        # Angular velocity derivative
        # Calculate torque differences for roll, pitch, yaw control
        L = self.arm_length
        b = self.b  # Drag coefficient (assumed small for simplicity)
        k = self.k  # Thrust coefficient (assumed small for simplicity)
        tau_roll, tau_pitch, tau_yaw = self.torques(gamma, L, b, k)
        
        # Angular acceleration using rigid body dynamics
        # Using the angular velocity dynamics from the image
        domega = translator.stack([
            (tau_roll / self.I_xx) - ((self.I_yy - self.I_zz) / self.I_xx) * omega_y * omega_z,
            (tau_pitch / self.I_yy) - ((self.I_zz - self.I_xx) / self.I_yy) * omega_x * omega_z,
            (tau_yaw / self.I_zz) - ((self.I_xx - self.I_yy) / self.I_zz) * omega_z * omega_y
        ])
        domega_x = domega[0]
        domega_y = domega[1]
        domega_z = domega[2]
        
        # Combine all derivatives
        if self.orientation:
            derivatives = translator.stack([
                droll, dpitch, dyaw,
                domega_x, domega_y, domega_z
            ])
        else:
            derivatives = translator.stack([
                domega_x, domega_y, domega_z
            ])

        return derivatives

    def torques(self, inputs, L, b, k):
        """
        Compute torques given current inputs, arm length, drag coefficient, and thrust coefficient.

        Args:
            inputs: List or array of squared angular velocities of the rotors.
            L: Arm length of the quadcopter.
            b: Drag coefficient.
            k: Thrust coefficient.

        Returns:
            Torque vector [tau_roll, tau_pitch, tau_yaw].
        """
        tau_roll = L * k * (inputs[0] - inputs[2])
        tau_pitch = L * k * (inputs[1] - inputs[3])
        tau_yaw = b * (inputs[0] - inputs[1] + inputs[2] - inputs[3])
        return tau_roll, tau_pitch, tau_yaw


class WaterTank(DynamicalSystem):
    """Water Tank dynamical system as described in equation (13)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.input_domain = [(0.1, 10.0)]  # Water level should be positive
        self.hidden_sizes = [12]
        self.delta = np.array([10.1 / 16])
        self.epsilon = 0.097 
        self.small_epsilon = 0.002  # is tractable for the larger network
        self.system_name = "WaterTank"
        
    def compute_dynamics(self, x, translator):
        # ẋ = 1.5 - √x
        return translator.stack([1.5 - translator.sqrt(x[0])])


class JetEngine(DynamicalSystem):
    """Jet Engine dynamical system as described in equation (14)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.input_domain = [(-1.0, 1.0), (-1.0, 1.0)]  # Typical domain for jet engine state variables
        self.hidden_sizes = [10, 16]
        self.delta = np.array([0.25, 0.5])
        self.epsilon = 0.039     
        self.small_epsilon = 0.012    # is tractable for the larger network 
        self.system_name = "JetEngine"
        
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
        self.input_domain = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]  # Typical domain for steam governor
        self.hidden_sizes = [12]
        self.delta = np.array([0.5, 0.5, 0.5])
        self.epsilon = 0.105     
        self.small_epsilon = 0.06     # is tractable for the larger network 
        self.system_name = "SteamGovernor"
        
    def compute_dynamics(self, x, translator):
        # ẋ = y
        # ẏ = z² sin(x) cos(x) - sin(x) - 3y
        # ż = -(cos(x) - 1)

        # Use trig identity sin(x) * cos(x) = (1/2) * sin(2x)
        
        dx = x[1]
        dy = translator.pow(x[2], 2) * 0.5 * translator.sin(2 * x[0]) - translator.sin(x[0]) - 3 * x[1]
        dz = -(translator.cos(x[0]) - 1)
        
        return translator.stack([dx, dy, dz])


class Exponential(DynamicalSystem):
    """Exponential dynamical system as described in equation (16)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.input_domain = [(-1.0, 1.0), (-1.0, 1.0)]  # Restricted domain to avoid extremely large values
        self.hidden_sizes = [14, 14]
        self.delta = np.array([0.5, 0.25])
        self.epsilon = 0.112     
        self.small_epsilon = 0.04      # is tractable for the larger network 
        self.system_name = "Exponential"
        
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
        self.input_domain = [(0.0, 1.0), (-1.0, 1.0)]  # Typical domain for analysis
        self.hidden_sizes = [10]
        self.delta = np.array([0.125, 0.5])
        self.epsilon = 0.11     
        self.small_epsilon = 0.03     # is tractable for the larger network 
        self.system_name = "NonLipschitzVectorField1"
        
    def compute_dynamics(self, x, translator):
        # ẋ = y
        # ẏ = √|x|
        
        dx = x[1]
        dy = translator.sqrt(x[0])
        
        return translator.stack([dx, dy])


class NonLipschitzVectorField2(DynamicalSystem):
    """Non-Lipschitz Vector Field 2 (NL2) dynamical system as described in equation (18)."""
    
    def __init__(self):
        super().__init__()
        self.input_dim = 2
        self.output_dim = 2
        self.input_domain = [(-1.0, 1.0), (-1.0, 1.0)]  # Typical domain for analysis
        self.hidden_sizes = [12, 10]
        self.delta = np.array([0.25, 0.5])
        self.epsilon = 0.081     
        self.small_epsilon = 0.02      # is tractable for the larger network 
        self.system_name = "NonLipschitzVectorField2"
        
    def compute_dynamics(self, x, translator):
        # ẋ = x² + y
        # ẏ = (x²)^(1/3) - x
        
        dx = translator.pow(x[0], 2) + x[1]
        dy = translator.cbrt(translator.pow(x[0], 2)) - x[0]
        
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
        self.hidden_sizes = [64, 64, 64]
        self.delta = np.array([0.375])
        self.epsilon = 0.01 * 0.3 * (55 - math.sin(3.0))   
        self.system_name = "NonlinearOscillator"
    
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


class Sine2D(DynamicalSystem):
    """A simple 2D sine dynamical system with configurable frequencies."""
    
    def __init__(self, freq_x=1.0, freq_y=0.5):
        super().__init__()
        # Frequency parameters to control oscillation speed
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.input_dim = 2  # 2D system
        self.output_dim = 2  # 2D output
        self.input_domain = [(-np.pi, np.pi), (-np.pi, np.pi)]  # Domain for both dimensions
        self.hidden_sizes = [64, 64, 64]
        self.delta = np.array([0.5, 1.0])
        self.epsilon = 0.02   
        self.leaky_relu = True
        self.system_name = "Sine2D"
    
    def compute_dynamics(self, x, translator):
        """
        Compute 2D sine dynamics with frequency components.
        
        Args:
            x: Input tensor with shape [2, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            Tensor of shape [2, batch_size] with the dynamics
        """
        # ẋ = sin(freq_y * y)
        # ẏ = -sin(freq_x * x)
        dx = translator.sin(self.freq_y * x[1])
        dy = -translator.sin(self.freq_x * x[0])
        
        return translator.stack([dx, dy])


class VortexShedding3D(DynamicalSystem):
    """
    A 3D vortex shedding dynamical system that models fluid-structure interaction phenomena.
    This is a simplified model capturing the key features of vortex shedding behavior in 3D.
    """
    
    def __init__(self, strouhal=0.2, reynolds=100.0, amplitude=1.0):
        super().__init__()
        # Parameters that control the vortex shedding behavior
        self.strouhal = strouhal      # Strouhal number: dimensionless frequency
        self.reynolds = reynolds      # Reynolds number: ratio of inertial to viscous forces
        self.amplitude = amplitude    # Amplitude of oscillation
        
        self.input_dim = 3            # 3D system
        self.output_dim = 3           # 3D output
        self.input_domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]  # Domain for all dimensions
        self.hidden_sizes = [64, 64]  # Define hidden sizes for neural network
        self.delta = np.array([0.5, 0.5, 0.5])  # Delta for each dimension
        self.epsilon = 0.05           # Precision for verification
        self.system_name = "VortexShedding3D"
    
    def compute_dynamics(self, x, translator):
        """
        Compute 3D vortex shedding dynamics.
        The system exhibits limit cycle behavior with frequency based on Strouhal number
        and amplitude affected by Reynolds number.
        
        Args:
            x: Input tensor with shape [3, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            Tensor of shape [3, batch_size] with the dynamics
        """
        # Extract state variables
        x1, x2, x3 = x[0], x[1], x[2]
        
        # Parameters derived from Reynolds and Strouhal numbers
        damping = 0.5 / self.reynolds
        frequency = self.strouhal * 2.0
        
        # Dynamics inspired by coupled oscillators with nonlinear damping
        # This creates a limit cycle behavior characteristic of vortex shedding
        dx1 = x2
        dx2 = -frequency**2 * x1 - damping * x2 * (translator.pow(x1, 2) + translator.pow(x3, 2))
        dx3 = -damping * x3 + self.amplitude * translator.sin(frequency * x1)
        
        return translator.stack([dx1, dx2, dx3])


class VortexShedding4D(DynamicalSystem):
    """
    A 4D vortex shedding dynamical system that models more complex fluid-structure interaction.
    This extended model captures additional features like vortex stretching and twisting,
    which are important phenomena in 3D fluid flows with a fourth variable representing
    vortex intensity/strength.
    """
    
    def __init__(self, strouhal=0.2, reynolds=100.0, amplitude=1.0, coupling=0.3):
        super().__init__()
        # Parameters that control the vortex shedding behavior
        self.strouhal = strouhal      # Strouhal number: dimensionless frequency
        self.reynolds = reynolds      # Reynolds number: ratio of inertial to viscous forces
        self.amplitude = amplitude    # Amplitude of oscillation
        self.coupling = coupling      # Coupling strength between spatial variables and vortex intensity
        
        self.input_dim = 4            # 4D system (3D space + intensity)
        self.output_dim = 4           # 4D output
        self.input_domain = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0)]  # Domain for all dimensions
        self.hidden_sizes = [96, 96]  # Define hidden sizes for neural network
        self.delta = np.array([0.5, 0.5, 0.5, 0.25])  # Delta for each dimension
        self.epsilon = 0.05           # Precision for verification
        self.system_name = "VortexShedding4D"
    
    def compute_dynamics(self, x, translator):
        """
        Compute 4D vortex shedding dynamics with vortex intensity as fourth dimension.
        The system exhibits limit cycle behavior with stretching and folding characteristic
        of vortex structures in fluid dynamics.
        
        Args:
            x: Input tensor with shape [4, batch_size]
            translator: The translator for mathematical operations
            
        Returns:
            Tensor of shape [4, batch_size] with the dynamics
        """
        # Extract state variables
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        
        # Parameters derived from Reynolds and Strouhal numbers
        damping = 0.5 / self.reynolds
        frequency = self.strouhal * 2.0
        
        # Dynamics inspired by coupled oscillators with nonlinear damping and vortex stretching
        dx1 = x2
        dx2 = -frequency**2 * x1 - damping * x2 * (translator.pow(x1, 2) + translator.pow(x3, 2))
        dx3 = -damping * x3 + self.amplitude * translator.sin(frequency * x1) + self.coupling * x4
        
        # Fourth dimension: vortex intensity dynamics influenced by spatial variables
        # The intensity equation models vortex stretching and folding behavior
        dx4 = -damping * x4 + self.coupling * (
            translator.sin(x1 * x3) - 
            0.5 * x2 * translator.pow(x3, 2) +
            0.2 * translator.cos(frequency * x2)
        )
        
        return translator.stack([dx1, dx2, dx3, dx4])


class LorenzAttractor(DynamicalSystem):
    """A class representing the Lorenz attractor dynamics."""

    def __init__(self, sigma=10.0, rho=28.0, beta=8.0 / 3.0):   # These are the values Lorenz used with the beautiful butterfly
        super().__init__()
        # Parameter for the Lorenz attractor
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.input_dim = 3  # Lorenz attractor state dimension
        self.output_dim = 3  # Lorenz attractor derivative dimension
        self.input_domain = [(-30.0, 30.0), (-30.0, 30.0), (0.0, 60.0)]  # Typical domain for Lorenz attractor
        self.hidden_sizes = [64, 64, 64]
        self.delta = np.array([30.0, 30.0, 30.0])  # Domain size for the input
        self.epsilon = 0.6  # Let's try?
        self.system_name = "LorenzAttractor"

    def compute_dynamics(self, x, translator):
        """
        Compute Lorenz dynamics.

        Args:
            x: Input tensor with shape [3, batch_size]
            translator: The translator for mathematical operations

        Returns:
            Tensor of shape [3, batch_size] with the dynamics
        """
        x, y, z = x[0], x[1], x[2]

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        return translator.stack([dx, dy, dz])


class NNDynamics(DynamicalSystem):
    """A class representing the dynamics of a neural network."""

    def __init__(self, network, input_domain):
        super().__init__()
        self.network = network
        self.input_dim = len(input_domain)
        self.output_dim = self.input_dim
        self.input_domain = input_domain
        self.system_name = "NNDynamics"

    @torch.no_grad()
    def compute_dynamics(self, x, translator):
        """
        Compute the dynamics for the neural network.
        
        Args:
            x: The state tensor with shape [input_dim, batch_size]
            
        Returns:
            The derivatives of the system with shape [output_dim, batch_size]
        """
        assert isinstance(translator, TorchTranslator), "NNDynamics only supports TorchTranslator"
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        # Forward pass through the neural network
        return self.network(x.T).T

class QuadraticSystem(DynamicalSystem):
    """Dynamical system with linear and quadratic components:
       ẋ₁ = μx₁, ẋ₂ = λ(x₂ - x₁²)
    """

    def __init__(self, mu=-0.05, lam=-1.0):
        super().__init__()
        self.mu = mu
        self.lam = lam
        self.input_dim = 2
        self.time_steps = 50  # Number of discrete steps to simulate
        self.output_dim = self.input_dim * self.time_steps
        self.input_domain = [(-0.5, 0.5), (-0.5, 0.5)]
        self.hidden_sizes = [64, 64]
        self.delta = np.array([0.5, 0.5])
        self.epsilon = 0.1
        self.system_name = "QuadraticSystem"  # Name of the system
        self.time_step = 0.02  # Time step for integration

    def continuous_dynamics(self, t, y):
        x1, x2 = y
        dx1dt = self.mu * x1
        dx2dt = self.lam * (x2 - x1**2)
        return [dx1dt, dx2dt]

    def compute_dynamics_integrate(self, x, T):
        from scipy.integrate import solve_ivp
        t_span = (0, T)
        # Solve the system
        sol = solve_ivp(self.continuous_dynamics, t_span, x, t_eval=np.linspace(t_span[0], t_span[1], 500))
        return sol.y[:, -1]

    def compute_dynamics_discrete(self, x_current, translator, dt):
        """
        Computes one discrete time step of the quadratic system.
        x_current: state tensor of shape [2, batch_size]
        translator: math operations translator
        dt: time step duration
        """
        x1_0 = x_current[0]
        x2_0 = x_current[1]

        # x1(t)
        x1_next = x1_0 * np.exp(self.mu * dt)

        # x2(t)
        # Check for the special case 2*mu == lam with a small tolerance
        if abs(2 * self.mu - self.lam) < 1e-9:
            # Special case: 2μ = λ
            # C = x2_0 / translator.exp(self.lam * 0) which is x2_0
            C = x2_0
            x2_next = C * np.exp(self.lam * dt) - \
                      self.lam * translator.pow(x1_0, 2) * dt * np.exp(self.lam * dt)
        else:
            # General case: 2μ != λ
            A = self.lam * translator.pow(x1_0, 2) / (2 * self.mu - self.lam)
            # C = (x2_0 + A) / translator.exp(self.lam * 0) which is x2_0 + A
            C = x2_0 + A
            x2_next = C * np.exp(self.lam * dt) - \
                      A * np.exp(2 * self.mu * dt)

        return translator.stack([x1_next, x2_next])

    def compute_dynamics(self, x0, translator):
        """
        Computes the system dynamics by iterating discrete time steps.
        x0: initial state tensor of shape [2, batch_size]
        translator: math operations translator
        return trajectory of size [2*time_steps, batch_size]
        """
        import torch
        time_steps = self.time_steps  # Number of discrete steps to simulate
        dt = self.time_step    # Duration of each discrete step
        x = x0
        for i in range(time_steps):
            x_next = self.compute_dynamics_discrete(x0, translator, (i+1)*dt)
            x = translator.cat([x, x_next])
        
        return x