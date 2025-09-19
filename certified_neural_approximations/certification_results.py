import abc
import numpy as np


class CertificationRegion:
    def __init__(self, center: np.array, radius: np.array, output_dim: int = None, split_dims: list[int] = None, last_split_dim: int = None):
        self.center = center
        # radius in the sense of a hyperrectangle
        # {x : x[i] = c[i] + \alpha[i] r[i], \alpha \in [-1, 1]^n, i = 1..n}
        self.radius = radius
        self.output_dim = output_dim
        self.min_radius = 1e-6 * self.radius

        if split_dims is None:
            split_dims = list(range(center.shape[0]))
        self.split_dims = split_dims

        if last_split_dim is None:
            last_split_dim = len(split_dims) - 1
        self.last_split_dim = last_split_dim

    def __iter__(self):
        return iter((self.center, self.radius, self.output_dim))

    def lebesguemeasure(self):
        """
        Calculate the size of the region (hypercube volume).
        """
        return np.prod(2 * self.radius).item()

    def nextsplitdim(self, taylor_approximation, dynamics, timeout=False):
        """
        Identify the dimension with the highest approximation error for splitting.

        This method tests a random point within the region to find which dimension
        contributes most to the approximation error between the Taylor approximation
        and the actual dynamics.

        :param taylor_approximation: Function that computes the Taylor approximation at a given point
        :param dynamics: The actual dynamics function to compare against
        :return: The dimension index with the highest approximation error
        """
        if timeout:
            split_dims = list(range(self.center.shape[0]))
        else:
            split_dims = self.split_dims

        sample, delta = self.center, self.radius  # Unpack the data tuple
        approximation_error = taylor_approximation(sample) - dynamics(sample).flatten()[self.output_dim]
        # i0 = self.incrementsplitdim() # Make sure that we cycle through the dimensions, incase the approximation error is always zero
        error_list = np.ones(len(split_dims)) * 10e-9  # Initialize near zero

        if all(delta < self.min_radius):
            return None

        for i in range(len(split_dims)):
            j = split_dims[i]  # Get the current split dimension index

            delta_j = delta[j]
            if delta_j < self.min_radius[j]:
                error_list[i] = 0.0
                continue

            left_point = sample.copy()
            left_point[j] -= 0.5 * delta_j

            # Calculate the Taylor approximation at the left point (corrected by the error from the centre)
            approx = taylor_approximation(left_point) - approximation_error
            true_value = dynamics(left_point).flatten()[self.output_dim]
            left_error = np.abs(approx - true_value)

            right_point = sample.copy()
            right_point[j] += 0.5 * delta_j

            # Calculate the Taylor approximation at the right point (corrected by the error from the centre)
            approx = taylor_approximation(right_point) - approximation_error
            true_value = dynamics(right_point).flatten()[self.output_dim]
            right_error = np.abs(approx - true_value)

            max_error = np.maximum(left_error, right_error)

            if max_error > 0.0:
                error_list[i] = max_error

        delta_maxmin_ratio = np.max(delta[split_dims]) / np.min(delta[split_dims])
        if delta_maxmin_ratio.item() > 1e2:
            # Softmax calculation
            probabilities = error_list / np.sum(error_list)
            split_dim_idx = np.random.choice(len(error_list), p=probabilities)
        else:
            split_dim_idx = np.argmax(error_list)
        return split_dims[split_dim_idx]

    def incrementsplitdim(self):
        self.last_split_dim = (self.last_split_dim + 1) % len(self.split_dims)
        return self.split_dims[self.last_split_dim]
    
    def islinear(self):
        return len(self.split_dims) == 0

    def __repr__(self):
        return f"CertificationRegion(center={self.center}, radius={self.radius}, output_dim={self.output_dim})"


class AugmentedSample(CertificationRegion):
    def __init__(self, center, radius, first_order_model, output_dim=None, split_dims=None, last_split_dim=None):
        super().__init__(center, radius, output_dim, split_dims, last_split_dim)
        self.first_order_model = first_order_model

    @staticmethod
    def from_certification_region(region, first_order_model):
        aug_sample = AugmentedSample(region.center, region.radius, first_order_model, region.output_dim, region.split_dims, region.last_split_dim)
        aug_sample.min_radius = region.min_radius
        return aug_sample

    def isfinite(self):
        return np.isfinite(self.first_order_model[0][0]).all() and np.isfinite(self.first_order_model[0][1]).all() and \
            np.isfinite(self.first_order_model[1][0]).all() and np.isfinite(self.first_order_model[1][1]).all()


class SampleResult(abc.ABC):
    def __init__(self, sample, start_time):
        self.sample = sample
        self.start_time = start_time

    @abc.abstractmethod
    def issat(self) -> bool:
        pass

    @abc.abstractmethod
    def isunsat(self) -> bool:
        pass

    def isleaf(self) -> bool:
        return self.issat() or self.isunsat()

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
    def __init__(self, sample, start_time, counterexamples):
        super().__init__(sample, start_time)
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
    def __init__(self, sample, start_time, new_samples):
        super().__init__(sample, start_time)
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
