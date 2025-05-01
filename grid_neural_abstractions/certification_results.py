import abc
import numpy as np

class CertificationRegion:
    def __init__(self, center: np.array, radius: np.array, output_dim: int = None, split_dim: int = None):
        self.center = center
        # radius in the sense of a hyperrectangle
        # {x : x[i] = c[i] + \alpha[i] r[i], \alpha \in [-1, 1]^n, i = 1..n}
        self.radius = radius
        self.output_dim = output_dim
        self.min_radius = 1e-6 * self.radius

        if split_dim is None:
            split_dim = center.shape[0] - 1
        self.split_dim = split_dim

    def __iter__(self):
        return iter((self.center, self.radius, self.output_dim))

    def lebesguemeasure(self):
        """
        Calculate the size of the region (hypercube volume).
        """
        return np.prod(2 * self.radius).item()
    
    def nextsplitdim(self, taylor_approximation, dynamics):
        """
        Identify the dimension with the highest approximation error for splitting.
        
        This method tests a random point within the region to find which dimension
        contributes most to the approximation error between the Taylor approximation
        and the actual dynamics.

        :param taylor_approximation: Function that computes the Taylor approximation at a given point
        :param dynamics: The actual dynamics function to compare against
        :return: The dimension index with the highest approximation error
        """
        sample, delta = self.center, self.radius  # Unpack the data tuple
        error = -0.1  # Initialize error to a negative value to ensure it gets updated
        split_dim = None
        approximation_error = taylor_approximation(sample) - dynamics(sample).flatten()[self.output_dim]
        for i, delta_i in enumerate(delta):
            if delta_i < self.min_radius[i]:
                continue
            random_point = sample.copy()
            random_point[i] += 0.1 * delta_i
            # Calculate the Taylor approximation at the random point (corrected by the error from the centre)
            approx = taylor_approximation(random_point) - approximation_error
            true_value = dynamics(random_point).flatten()[self.output_dim]
            current_error = np.abs(approx - true_value)
            if current_error > error:
                split_dim = i
                error = current_error
        return split_dim
    
    def incrementsplitdim(self):
        self.split_dim = (self.split_dim + 1) % self.center.shape[0]
        return self.split_dim
    
    def __repr__(self):
        return f"CertificationRegion(center={self.center}, radius={self.radius}, output_dim={self.output_dim})"


class SampleResult(abc.ABC):
    def __init__(self, sample, computation_time):
        self.sample = sample
        self.computation_time = computation_time
    
    @abc.abstractmethod
    def issat(self) -> bool:
        pass

    @abc.abstractmethod
    def isunsat(self) -> bool:
        pass

    @abc.abstractmethod
    def hasnewsamples(self) -> bool:
        pass

    def newsamples(self):
        raise ValueError("New samples not available for this sample result.")

    @abc.abstractmethod
    def hascounterexamples(self) -> bool:
        pass

    def counterexamples(self):
        raise ValueError("Counterexamples not available for this sample result.")
    
    def lebesguemeasure(self):
        return self.sample.lebesguemeasure()
    

class SampleResultSAT(SampleResult):
    def issat(self) -> bool:
        return True

    def isunsat(self) -> bool:
        return False

    def hasnewsamples(self) -> bool:
        return False

    def hascounterexamples(self) -> bool:
        return False

    def __repr__(self):
        return f"SAT: {self.sample}"


class SampleResultUNSAT(SampleResult):
    def __init__(self, sample, computation_time, counterexamples):
        super().__init__(sample, computation_time)
        self._counterexamples = counterexamples

    def issat(self) -> bool:
        return False

    def isunsat(self) -> bool:
        return True

    def hasnewsamples(self) -> bool:
        return False

    def hascounterexamples(self) -> bool:
        return True
    
    def counterexamples(self):
        return self._counterexamples

    def __repr__(self):
        return f"UNSAT: {self.sample}"
    

class SampleResultMaybe(SampleResult):
    def __init__(self, sample, computation_time, new_samples):
        super().__init__(sample)
        self._new_samples = new_samples

    def issat(self) -> bool:
        return False
    
    def isunsat(self) -> bool:
        return False
    
    def hasnewsamples(self) -> bool:
        return True
    
    def newsamples(self):
        return self._new_samples
    
    def hascounterexamples(self) -> bool:
        return False

    def __repr__(self):
        return f"Maybe: {self.sample}"
