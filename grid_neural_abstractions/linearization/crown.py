
import torch
from bound_propagation import LinearBounds
from grid_neural_abstractions.certification_results import AugmentedSample
from grid_neural_abstractions.translators import BoundPropagationTranslator


class CrownLinearization:
    def __init__(self, dynamics):
        """
        Initialize the Crown linearization strategy.

        :param dynamics: The dynamics of the system.
        """
        self.dynamics = dynamics
        self.translator = BoundPropagationTranslator()
        self.traced_model = None

    def linearize(self, samples):
        """
        Linearizes a batch of samples using Taylor expansion.
        """
        if self.traced_model is None:
            x = self.translator.to_format(samples[0].center)
            self.traced_model = self.dynamics.compute_dynamics(x, self.translator)

        centers = torch.stack([torch.as_tensor(sample.center, dtype=torch.float32) for sample in samples])
        deltas = torch.stack([torch.as_tensor(sample.radius, dtype=torch.float32) for sample in samples])
        linear_bounds = self.translator.bound(self.traced_model, centers, deltas)

        A_lower = linear_bounds.lower[0]
        b_lower = linear_bounds.lower[1]

        A_upper = linear_bounds.upper[0]
        b_upper = linear_bounds.upper[1]

        A_gap = A_upper - A_lower
        b_gap = b_upper - b_lower
        lbp_gap = LinearBounds(linear_bounds.region, None, (A_gap, b_gap))
        interval_gap = lbp_gap.concretize()  # Turn linear bounds into interval bounds

        def to_augmented_sample(i):
            sample = samples[i]
            j = sample.output_dim
            return AugmentedSample.from_certification_region(
                sample,
                ((A_lower[i, j].numpy(), b_lower[i, j].numpy()),
                 (A_upper[i, j].numpy(), b_upper[i, j].numpy()), interval_gap.upper[0, j].item())
            )

        augmented_samples = [to_augmented_sample(i) for i in range(len(samples))]

        return augmented_samples
