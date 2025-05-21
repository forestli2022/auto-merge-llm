from .on_disk_merge_evaluator import OnDiskMergeEvaluator
from .in_memory_merge_evaluator import InMemoryMergeEvaluator
evaluator_classes = {
    'ondisk_evaluate': OnDiskMergeEvaluator,
    'inmemory_evaluate': InMemoryMergeEvaluator
}
