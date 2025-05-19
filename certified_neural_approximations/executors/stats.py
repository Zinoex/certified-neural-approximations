
class Statistics:
    def __init__(self, samples):
        # Calculate the total domain size
        self.total_domain_size = sum(sample.lebesguemeasure() for sample in samples)
        self.certified_domain_size = 0.0
        self.uncertified_domain_size = 0.0

    def add_sample(self, sample):
        if sample.issat():
            self.certified_domain_size += sample.lebesguemeasure()
        elif sample.isunsat():
            self.uncertified_domain_size += sample.lebesguemeasure()

    def get_certified_percentage(self):
        if self.total_domain_size == 0:
            return 0.0
        return (self.certified_domain_size / self.total_domain_size) * 100

    def get_uncertified_percentage(self):
        if self.total_domain_size == 0:
            return 0.0
        return (self.uncertified_domain_size / self.total_domain_size) * 100
