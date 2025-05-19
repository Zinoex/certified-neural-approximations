
import torch
from bound_propagation import LinearBounds
from certified_neural_approximations.certification_results import AugmentedSample
from certified_neural_approximations.translators import BoundPropagationTranslator


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
            if torch.cuda.is_available():
                self.traced_model = self.traced_model.to(torch.device("cuda"))

        centers = torch.stack([torch.as_tensor(sample.center, dtype=torch.float32) for sample in samples])
        deltas = torch.stack([torch.as_tensor(sample.radius, dtype=torch.float32) for sample in samples])

        if torch.cuda.is_available():
            centers = centers.to(torch.device("cuda"))
            deltas = deltas.to(torch.device("cuda"))

        linear_bounds = self.translator.bound(self.traced_model, centers, deltas)

        A_lower = linear_bounds.lower[0]
        b_lower = linear_bounds.lower[1]

        A_upper = linear_bounds.upper[0]
        b_upper = linear_bounds.upper[1]

        A_gap = A_upper - A_lower
        b_gap = b_upper - b_lower
        lbp_gap = LinearBounds(linear_bounds.region, None, (A_gap, b_gap))
        interval_gap = lbp_gap.concretize()  # Turn linear bounds into interval bounds

        A_lower = A_lower.cpu().numpy()
        b_lower = b_lower.cpu().numpy()
        A_upper = A_upper.cpu().numpy()
        b_upper = b_upper.cpu().numpy()
        max_gap = interval_gap.upper.cpu().numpy()

        def to_augmented_sample(i):
            sample = samples[i]
            j = sample.output_dim
            return AugmentedSample.from_certification_region(
                sample,
                ((A_lower[i, j], b_lower[i, j]),
                 (A_upper[i, j], b_upper[i, j]), max_gap[i, j].item())
            )

        augmented_samples = [to_augmented_sample(i) for i in range(len(samples))]

        return augmented_samples

    def linearize_sample(self, sample):
        """
        Linearizes a single sample using Taylor expansion.
        """
        return self.linearize([sample])[0]
