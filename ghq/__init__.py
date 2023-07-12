import importlib.metadata

from .univariate import univariate as univariate
from .univariate import univariate_importance as univariate_importance

__version__ = importlib.metadata.version("ghq")

del importlib
