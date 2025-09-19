from .taylor import TaylorLinearization
from .crown import CrownLinearization
from .backup import BackupLinearization


def default_linearization(dynamics):
    """
    Returns the default linearization method.
    """
    return BackupLinearization(
        TaylorLinearization(dynamics),
        CrownLinearization(dynamics)
    )


__all__ = [
    "TaylorLinearization",
    "CrownLinearization",
    "BackupLinearization",
]
