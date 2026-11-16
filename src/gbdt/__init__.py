"""gbdt: second-order, histogram-based gradient-boosted decision trees from scratch."""

from .binning import BinMapper
from .booster import GBDTClassifier, GBDTRegressor
from .losses import CustomLoss, Logistic, Loss, PseudoHuber, Softmax, SquaredError
from .tree import Tree, TreeGrower, TreeParams

__all__ = [
    "BinMapper",
    "GBDTClassifier",
    "GBDTRegressor",
    "CustomLoss",
    "Logistic",
    "Loss",
    "PseudoHuber",
    "Softmax",
    "SquaredError",
    "Tree",
    "TreeGrower",
    "TreeParams",
]

__version__ = "0.1.0"
