from .taylor import TaylorLinearization
from .crown import CrownLinearization
from .backup import BackupLinearization
from .python_taylor import PythonTaylorLinearization
from .comparison_taylor import ComparisonTaylorLinearization


def default_linearization(dynamics):
    """
    Returns the default linearization method.
    """
    return BackupLinearization(
        ComparisonTaylorLinearization(dynamics),
        CrownLinearization(dynamics)
    )


__all__ = [
    "TaylorLinearization",
    "CrownLinearization",
    "BackupLinearization",
    "PythonTaylorLinearization",
    "ComparisonTaylorLinearization",
]
