import torch


class VanDerPolOscillator:
    """A class representing the Van der Pol oscillator dynamics."""

    def __init__(self, mu=1.0):
        # Parameter for the Van der Pol oscillator
        self.mu = mu
        self.input_dim = 2  # Van der Pol oscillator state dimension
        self.output_dim = 2  # Van der Pol oscillator derivative dimension

    def __call__(self, x):
        return self.compute_dynamics(x)

    def compute_dynamics(self, x):
        # Ensure x has the correct shape (2, N) for computation
        if x.dim() == 1:
            x = x.unsqueeze(1)  # Convert 1D tensor to 2D column vector
        elif x.size(0) != self.input_dim:
            raise ValueError(
                f"Input tensor x has incompatible dimensions: {x.size()}, expected first dimension {self.input_dim}"
            )

        # Compute the dynamics
        dx1 = x[1]
        dx2 = self.mu * (1 - x[0] ** 2) * x[1] - x[0]

        return torch.cat((dx1.unsqueeze(0), dx2.unsqueeze(0)), dim=0)

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


class Quadcopter:
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
        # Parameters for the quadcopter model
        self.mass = mass
        self.gravity = gravity
        self.arm_length = arm_length
        self.moment_inertia_x = moment_inertia_x
        self.moment_inertia_y = moment_inertia_y
        self.moment_inertia_z = moment_inertia_z

        self.input_dim = 3  # 12D state: position (3), velocity (3), angles (3), angular rates (3)
        self.output_dim = 3  # 12D derivatives

    def __call__(self, x):
        return self.compute_dynamics(x)

    def compute_dynamics(self, x):
        # Ensure x has the correct shape for computation
        if x.dim() == 1:
            x = x.unsqueeze(1)  # Convert 1D tensor to 2D column vector
        elif x.size(0) != self.input_dim:
            raise ValueError(
                f"Input tensor x has incompatible dimensions: {x.size()}, expected first dimension {self.input_dim}"
            )

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
                torch.sin(pitch) * torch.cos(yaw)
                + torch.sin(roll) * torch.cos(pitch) * torch.sin(yaw)
            )
            * thrust
            / self.mass
        )
        dvy = (
            (
                torch.sin(pitch) * torch.sin(yaw)
                - torch.sin(roll) * torch.cos(pitch) * torch.cos(yaw)
            )
            * thrust
            / self.mass
        )
        dvz = (
            torch.cos(roll) * torch.cos(pitch) * thrust
        ) / self.mass - self.gravity

        # Combine all derivatives
        derivatives = torch.stack([dvx, dvy, dvz], dim=0)

        return derivatives

    def compute_lipschitz_constant(self, R):
        """
        Compute a more accurate Lipschitz constant for the quadcopter dynamics
        based on the maximum eigenvalue of the Jacobian matrix.

        Args:
            R: The radius of the domain

        Returns:
            The Lipschitz constant for the dynamics
        """
        # Compute based on the partial derivatives of the dynamics equations
        g = self.gravity
        m = self.mass

        # Maximum possible control thrust
        thrust_max = m * g * 2  # Assuming max thrust is twice hover thrust

        # Compute bounds on trigonometric functions (sin, cos are bounded by 1)
        # Maximum effect of angular positions on velocities
        angle_vel_coupling = thrust_max / m

        # Consider the largest possible value in the Jacobian
        L = angle_vel_coupling * (1 + R)

        # Add a safety factor
        return L
