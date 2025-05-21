from .normal_models import NormalModelsMerge
from .normal_slices import NormalSlicesMerge
from .lfs import LfsMerge
from .dis import DisMerge
from .lfs_multiobj import LfsMoMerge
from .prune import PruneMoMerge
from .merge_utils import MergeUtils

strategy_classes = {
    'normal_models': NormalModelsMerge,
    'normal_slices': NormalSlicesMerge,
    'lfs': LfsMerge,
    'dis': DisMerge,
    'lfs_multiobj': LfsMoMerge,
    'prune': PruneMoMerge
}
