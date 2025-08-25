from strategies.fold_gmm import FoldGMM
from .normal_models import NormalModelsMerge
from .normal_slices import NormalSlicesMerge
from .lfs import LfsMerge
from .dis import DisMerge
from .lfs_multiobj import LfsMoMerge
from .prune import PruneMoMerge
from .fold import FoldMerge
from .fold_merge_once import FoldMergeOnce
from .fold_different_params import FoldDifferentParams
from .fold_adamerging import FoldAdamerging
from .merge_utils import MergeUtils
from .prune_continuous import PruneContinuous
from .fold_gmm_cluster import FoldGMMCluster

strategy_classes = {
    'normal_models': NormalModelsMerge,
    'normal_slices': NormalSlicesMerge,
    'lfs': LfsMerge,
    'dis': DisMerge,
    'lfs_multiobj': LfsMoMerge,
    'prune': PruneMoMerge,
    'fold': FoldMerge,
    'fold_merge_once': FoldMergeOnce,
    'fold_different_params': FoldDifferentParams,
    'fold_adamerging': FoldAdamerging,
    'prune_continuous': PruneContinuous,
    'fold_gmm': FoldGMM,
    'fold_gmm_cluster': FoldGMMCluster
}
