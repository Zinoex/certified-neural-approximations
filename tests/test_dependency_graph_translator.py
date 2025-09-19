import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certified_neural_approximations.translators import DependencyGraphTranslator
from certified_neural_approximations.dynamics import (
    VanDerPolOscillator, Quadcopter, WaterTank, JetEngine, SteamGovernor,
    Exponential, NonLipschitzVectorField1, NonLipschitzVectorField2,
    NonlinearOscillator, Sine2D, VortexShedding3D, VortexShedding4D,
    LorenzAttractor, LowThrustSpacecraft
)


class TestDependencyGraphTranslator:
    def setup_method(self):
        """Set up test fixtures."""
        self.translator = DependencyGraphTranslator()
        
        # Create instances of all dynamical systems
        self.systems = {
            'VanDerPolOscillator': VanDerPolOscillator(),
            'Quadcopter': Quadcopter(),
            'WaterTank': WaterTank(),
            'JetEngine': JetEngine(),
            'SteamGovernor': SteamGovernor(),
            'Exponential': Exponential(),
            'NonLipschitzVectorField1': NonLipschitzVectorField1(),
            'NonLipschitzVectorField2': NonLipschitzVectorField2(),
            'NonlinearOscillator': NonlinearOscillator(),
            'Sine2D': Sine2D(),
            'VortexShedding3D': VortexShedding3D(),
            'VortexShedding4D': VortexShedding4D(),
            'LorenzAttractor': LorenzAttractor(),
            'LowThrustSpacecraft': LowThrustSpacecraft()
        }
             
    def test_vanderpol_oscillator(self):
        """Test Van der Pol oscillator dynamics with DependencyGraphTranslator."""
        system = self.systems['VanDerPolOscillator']
        
        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(2)
        result = system.compute_dynamics(x, self.translator)
        
        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, 'lin'],
            ['nonlin', 'nonlin']
        ], dtype=object))
             
    def test_quadcopter(self):
        """Test Quadcopter dynamics with DependencyGraphTranslator."""
        system = self.systems['Quadcopter']
        
        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(10)
        result = system.compute_dynamics(x, self.translator)

        
        # Verify structure
        # Commented out rows is for self.orientation == True
        assert np.all(result.dependencies == np.array([
            # ['lin', 'nonlin', 'nonlin', None, None, None, None, 'nonlin', 'nonlin', None],
            # [None, 'nonlin', 'nonlin', None, None, None, None, 'nonlin', None, None],
            # [None, 'nonlin', 'nonlin', None, None, None, None, 'nonlin', 'nonlin', None],
            [None, 'nonlin', 'nonlin', 'lin', None, 'lin', None, None, None, None],
            ['nonlin', None, 'nonlin', None, 'lin', None, 'lin', None, None, None],
            [None, 'nonlin', 'nonlin', 'lin', 'lin', 'lin', 'lin', None, None, None],
        ], dtype=object))
    
    def test_water_tank(self):
        """Test Water Tank dynamics with DependencyGraphTranslator."""
        system = self.systems['WaterTank']
        
        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(1)
        result = system.compute_dynamics(x, self.translator)
        
        # Verify structure
        assert np.all(result.dependencies == np.array([
            ['nonlin'],
        ], dtype=object))

    def test_jet_engine(self):
        """Test Jet Engine dynamics with DependencyGraphTranslator."""
        system = self.systems['JetEngine']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(2)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            ['nonlin', 'lin'],
            ['lin', 'lin'],
        ], dtype=object))

    def test_steam_governor(self):
        """Test Steam Governor dynamics with DependencyGraphTranslator."""
        system = self.systems['SteamGovernor']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(3)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, 'lin', None],
            ['nonlin', 'lin', 'nonlin'],
            ['nonlin', None, None],
        ], dtype=object))

    def test_exponential(self):
        """Test Exponential dynamics with DependencyGraphTranslator."""
        system = self.systems['Exponential']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(2)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, 'nonlin'],
            ['lin', None],
        ], dtype=object))

    def test_non_lipschitz_vector_field_1(self):
        """Test Non-Lipschitz Vector Field 1 dynamics with DependencyGraphTranslator."""
        system = self.systems['NonLipschitzVectorField1']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(2)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, 'lin'],
            ['nonlin', None],
        ], dtype=object))

    def test_non_lipschitz_vector_field_2(self):
        """Test Non-Lipschitz Vector Field 2 dynamics with DependencyGraphTranslator."""
        system = self.systems['NonLipschitzVectorField2']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(2)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            ['nonlin', 'lin'],
            ['nonlin', None],
        ], dtype=object))

    def test_nonlinear_oscillator(self):
        """Test Nonlinear Oscillator dynamics with DependencyGraphTranslator."""
        system = self.systems['NonlinearOscillator']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(1)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            ['nonlin'],
        ], dtype=object))

    def test_sine_2d(self):
        """Test Sine 2D dynamics with DependencyGraphTranslator."""
        system = self.systems['Sine2D']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(2)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, 'nonlin'],
            ['nonlin', None],
        ], dtype=object))

    def test_vortex_shedding_3d(self):
        """Test Vortex Shedding 3D dynamics with DependencyGraphTranslator."""
        system = self.systems['VortexShedding3D']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(3)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, 'lin', None],
            ['nonlin', 'nonlin', 'nonlin'],
            ['nonlin', None, 'lin'],
        ], dtype=object))

    def test_vortex_shedding_4d(self):
        """Test Vortex Shedding 4D dynamics with DependencyGraphTranslator."""
        system = self.systems['VortexShedding4D']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(4)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, 'lin', None, None],
            ['nonlin', 'nonlin', 'nonlin', None],
            ['nonlin', None, 'lin', 'lin'],
            ['nonlin', 'nonlin', 'nonlin', 'lin'],
        ], dtype=object))

    def test_lorenz_attractor(self):
        """Test Lorenz Attractor dynamics with DependencyGraphTranslator."""
        system = self.systems['LorenzAttractor']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(3)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            ['lin', 'lin', None],
            ['nonlin', 'lin', 'nonlin'],
            ['nonlin', 'nonlin', 'lin'],
        ], dtype=object))

    def test_low_thrust_spacecraft(self):
        """Test Low Thrust Spacecraft dynamics with DependencyGraphTranslator."""
        system = self.systems['LowThrustSpacecraft']

        # Compute dependencies with DependencyGraphTranslator
        x = self.translator.identity(7)
        result = system.compute_dynamics(x, self.translator)

        # Verify structure
        assert np.all(result.dependencies == np.array([
            [None, None, 'lin', None, None, None, None],
            ['nonlin', None, None, 'nonlin', None, None, None],
            ['nonlin', None, None, 'nonlin', 'nonlin', 'nonlin', 'nonlin'],
            ['nonlin', None, 'nonlin', 'nonlin', 'nonlin', 'nonlin', 'nonlin'],
            [None, None, None, None, None, 'lin', None],
        ], dtype=object))
