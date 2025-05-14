class BackupLinearization:
    def __init__(self, method1, method2):
        self.method1 = method1
        self.method2 = method2

    def linearize(self, samples):
        augmented_samples = self.method1.linearize(samples)
        finite = [sample for sample in augmented_samples if sample.isfinite()]

        # If all samples are finite, return them
        if len(finite) == len(augmented_samples):
            return augmented_samples

        non_finite = [sample for sample in augmented_samples if not sample.isfinite()]

        # Run backup method on non-finite samples
        extra_augmented_samples = self.method2.linearize(non_finite)
        finite.extend(extra_augmented_samples)

        if any(not sample.isfinite() for sample in finite):
            raise ValueError("Some samples are still not finite after backup linearization.")

        return finite
