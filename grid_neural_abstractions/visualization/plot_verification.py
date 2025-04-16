import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class DynamicsPlotter:
    """
    Class for visualizing 1D dynamics and verification results.
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
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel('Input')
        self.ax.set_ylabel('Output')
        self.ax.set_title('Dynamics and Verification Results')
        
        # Initialize plot with the dynamics function
        self.plot_dynamics()
        
        # Show plot
        plt.ion()  # Turn on interactive mode
        self.fig.show()
        plt.pause(0.1)
        
    def plot_dynamics(self):
        """Plot the dynamics function."""
        if self.dynamics_model.input_dim != 1:
            raise ValueError("Only 1D dynamics are supported for visualization")
            
        domain = self.dynamics_model.input_domain
        x = np.linspace(domain[0][0], domain[0][1], self.resolution)
        
        # Reshape for the dynamics model input
        x_input = np.array([x_val for x_val in x]).reshape(-1, 1)
        y = np.array([self.dynamics_model(x_val)[0] for x_val in x_input])
        
        self.ax.plot(x, y, 'b-', label='Dynamics')
        self.ax.legend()
    
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
        
        if len(center) != 1:  # Only for 1D
            return
            
        # Extract center and radius for 1D case
        x_center = center[0]
        x_radius = radius[0]
        
        # Calculate rectangle coordinates
        x_min = x_center - x_radius
        width = 2 * x_radius
        
        # Get y range for visualization
        y_range = self.ax.get_ylim()
        y_min = y_range[0]
        height = y_range[1] - y_range[0]
        
        # Create rectangle patch with alpha transparency
        if result.issat():
            color = 'green'
            label = 'Certified'
        elif result.isunsat():
            color = 'red'
            label = 'Counterexample'
        else:
            return
            
        rect = Rectangle((x_min, y_min), width, height, 
                         color=color, alpha=0.2, label=label)
        
        # Add the rectangle to the plot
        self.ax.add_patch(rect)
        
        # Update legend (avoiding duplicates)
        handles, labels = self.ax.get_legend_handles_labels()
        if label not in labels:
            self.ax.legend()
            
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
