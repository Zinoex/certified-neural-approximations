import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


class DynamicsPlotter:
    """
    Class for visualizing 1D and 2D dynamics and verification results.
    """
    def __init__(self, dynamics_model, resolution=100):
        """
        Initialize the plotter with a dynamics model.
        
        Args:
            dynamics_model: The dynamics model to visualize
            resolution: Number of points to plot for the dynamics function
        """
        self.dynamics_model = dynamics_model
        self.resolution = resolution
        self.input_dim = dynamics_model.input_dim
        self.output_dim = dynamics_model.output_dim
        
        # Initialize figure based on input dimension
        if self.input_dim == 1:
            self._init_1d_plot()
        elif self.input_dim == 2:
            self._init_2d_plot()
        else:
            raise ValueError(f"Visualization only supports 1D and 2D inputs, got {self.input_dim}D")
        
        # Show plot
        plt.ion()  # Turn on interactive mode
        plt.tight_layout()
        self.fig.show()
    
    def _init_1d_plot(self):
        """Initialize plot for 1D dynamics"""
        # Create subplots horizontally (side by side) for each output dimension
        self.fig, self.axes = plt.subplots(1, self.output_dim, figsize=(10 * self.output_dim, 6))
        
        # If there's only one output dimension, axes is not an array
        if self.output_dim == 1:
            self.axes = [self.axes]
            
        for i, ax in enumerate(self.axes):
            ax.set_xlabel('Input')
            ax.set_ylabel(f'Output {i+1}')
            ax.set_title(f'Dynamics - Component {i+1}')
        
        # Track certified and uncertified regions
        self.certified_regions = [[] for _ in range(self.output_dim)]
        self.uncertified_regions = [[] for _ in range(self.output_dim)]
        
        # Initialize plot with the dynamics function
        self.plot_dynamics()
    
    def _init_2d_plot(self):
        """Initialize plot for 2D dynamics"""
        # Create subplots in a grid layout instead of stacked
        # Determine grid dimensions based on output_dim
        if self.output_dim <= 3:
            rows, cols = 1, self.output_dim
        else:
            cols = min(3, self.output_dim)  # Max 3 columns
            rows = (self.output_dim + cols - 1) // cols  # Ceiling division
            
        self.fig = plt.figure(figsize=(6 * cols, 5 * rows))
        self.axes = []
        
        for i in range(self.output_dim):
            ax = self.fig.add_subplot(rows, cols, i+1, projection='3d')
            ax.set_xlabel('Input 1')
            ax.set_ylabel('Input 2')
            ax.set_zlabel(f'Output {i+1}')
            ax.set_title(f'Dynamics - Component {i+1}')
            self.axes.append(ax)
        
        # Initialize the verification result containers
        self.certified_patches = [[] for _ in range(self.output_dim)]
        self.uncertified_patches = [[] for _ in range(self.output_dim)]
        
        # Initialize plot with the dynamics function
        self.plot_dynamics()
        
    def plot_dynamics(self):
        """Plot the dynamics function."""
        if self.input_dim == 1:
            self._plot_1d_dynamics()
        elif self.input_dim == 2:
            self._plot_2d_dynamics()
    
    def _plot_1d_dynamics(self):
        """Plot 1D dynamics function"""
        domain = self.dynamics_model.input_domain
        x = np.linspace(domain[0][0], domain[0][1], self.resolution)
        
        # Reshape for the dynamics model input
        x_input = np.array([x_val for x_val in x]).reshape(-1, 1)
        y_outputs = np.array([self.dynamics_model(x_val.reshape(1, -1))[0] for x_val in x_input])
        
        # Plot each output dimension
        for i, ax in enumerate(self.axes):
            y = y_outputs[:, i] if y_outputs.ndim > 1 else y_outputs
            ax.plot(x, y, 'b-', label='Dynamics')
            ax.legend()
    
    def _plot_2d_dynamics(self):
        """Plot 2D dynamics function as a surface"""
        domain = self.dynamics_model.input_domain
        
        # Create grid points for each dimension
        grid_points_per_dim = [
            np.linspace(domain[i][0], domain[i][1], self.resolution)
            for i in range(self.input_dim)
        ]
        
        # Create mesh grid
        mesh = np.meshgrid(*grid_points_per_dim)
        
        # Reshape inputs for vectorized evaluation
        X = np.vstack(list(map(np.ravel, mesh)))
        Y = self.dynamics_model(X)

        # Plot each output dimension
        for i, ax in enumerate(self.axes):
            Z = Y[i].reshape(mesh[0].shape)
            # Use single color (blue) instead of colormap and remove colorbar
            surface = ax.plot_surface(mesh[0], mesh[1], Z, color='blue', alpha=0.8, 
                                     linewidth=0, antialiased=True)
        
        self.z_min, self.z_max = ax.get_zlim()
    
    def update_figure(self, result):
        """
        Update the figure with verification results.
        
        Args:
            result: The verification result object
        """
        if not hasattr(result.sample, 'center') or not hasattr(result.sample, 'radius'):
            return
            
        center = result.sample.center
        radius = result.sample.radius
        
        if len(center) != self.input_dim:  # Check dimension match
            return
            
        # Choose visualization based on input dimension
        if self.input_dim == 1:
            self._update_1d_figure(result)
        elif self.input_dim == 2:
            self._update_2d_figure(result)
    
    def _update_1d_figure(self, result):
        """Update 1D figure with verification results"""
        center = result.sample.center
        radius = result.sample.radius
        
        # Extract center and radius for 1D case
        x_center = center[0]
        x_radius = radius[0]
        
        # Calculate rectangle coordinates
        x_min = x_center - x_radius
        width = 2 * x_radius
        
        # Create rectangle patch for each output dimension
        for i, ax in enumerate(self.axes):
            y_range = ax.get_ylim()
            y_min = y_range[0]
            height = y_range[1] - y_range[0]
            
            # Create rectangle patch with alpha transparency
            if result.issat():
                color = 'green'
                rect = Rectangle((x_min, y_min), width, height, 
                                color=color, alpha=0.2, label='Certified')
                ax.add_patch(rect)
            elif result.isunsat():
                color = 'red'
                rect = Rectangle((x_min, y_min), width, height, 
                                color=color, alpha=0.2, label='Counterexample')
                ax.add_patch(rect)
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _update_2d_figure(self, result):
        """Update 2D figure with verification results"""
        center = result.sample.center
        radius = result.sample.radius
        
        # Extract center and radius for 2D case
        x_center, y_center = center
        x_radius, y_radius = radius
        
        # Define the grid points for the rectangle corners
        x_min, x_max = x_center - x_radius, x_center + x_radius
        y_min, y_max = y_center - y_radius, y_center + y_radius
        
        # Create a rectangle in each subplot
        for i, ax in enumerate(self.axes):
            # Get z range for the current axis
            if self.z_min is None or self.z_max is None:
                # If z limits are not set, use the default limits of the axis
                z_min, z_max = ax.get_zlim()
            else:
                # Use the z limits set during initialization
                z_min, z_max = self.z_min, self.z_max
            
            # Define the vertices of the rectangular prism
            # Use actual z-axis limits to determine the height of the polygon
            corners = np.array([
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ])
            
            # Define the faces of the rectangular prism
            faces = [
                [corners[0], corners[1], corners[2], corners[3]],  # bottom
                [corners[4], corners[5], corners[6], corners[7]],  # top
                [corners[0], corners[1], corners[5], corners[4]],  # front
                [corners[2], corners[3], corners[7], corners[6]],  # back
                [corners[0], corners[3], corners[7], corners[4]],  # left
                [corners[1], corners[2], corners[6], corners[5]]   # right
            ]
            
            # Create a collection of polygons
            if result.issat():
                color = 'green'
                alpha = 0.15
            elif result.isunsat():
                color = 'red'
                alpha = 0.15
            else:
                continue
                
            # Use Poly3DCollection instead of PolyCollection for 3D plots
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            # Create and add 3D polygon collection with proper z-height
            pc = Poly3DCollection([face for face in faces], 
                                  alpha=alpha, 
                                  facecolor=color, 
                                  edgecolor=None)
            ax.add_collection3d(pc)
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
