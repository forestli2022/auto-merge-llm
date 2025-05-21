from .linear import LinearMerging
from .breadcrumbs import BreadcrumbsMerging
from .fisher import FisherMerging
from .regmean import RegmeanMerging
from .slerp import SlerpMerging
from .stock import StockMerging
from .task_arithmetic import TaskArithmetic
from .ties import TiesMerging
from .widen import WidenMerging
from .passthrough import PassthroughMerging

merging_methods_dict = {
    "linear": LinearMerging,
    "breadcrumbs": BreadcrumbsMerging,
    "fisher": FisherMerging,
    "regmean": RegmeanMerging,
    "slerp": SlerpMerging,
    "stock": StockMerging,
    "task_arithmetic": TaskArithmetic,
    "ties": TiesMerging,
    "widen": WidenMerging,
    "passthrough": PassthroughMerging,
}


__all__ = ["merging_methods_dict"]