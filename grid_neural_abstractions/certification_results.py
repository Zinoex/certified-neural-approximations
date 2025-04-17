import abc
import numpy as np


class CertificationRegion:
    def __init__(self, center: np.array, radius: np.array, ouput_dim: int = None):
        self.center = center
        # radius in the sense of a hyperrectangle
        # {x : x[i] = c[i] + \alpha[i] r[i], \alpha \in [-1, 1]^n, i = 1..n}
        self.radius = radius
        self.output_dim = ouput_dim

    def __iter__(self):
        return iter((self.center, self.radius, self.output_dim))

    def lebesguemeasure(self):
        """
        Calculate the size of the region (hypercube volume).
        """
        return np.prod(2 * self.radius).item()
    
    def __repr__(self):
        return f"CertificationRegion(center={self.center}, radius={self.radius}, output_dim={self.output_dim})"


class SampleResult(abc.ABC):
    def __init__(self, sample):
        self.sample = sample
    
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
    def __init__(self, sample, counterexamples):
        super().__init__(sample)
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
    def __init__(self, sample, new_samples):
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
