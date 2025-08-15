from contextlib import nullcontext
import numbers
import torch
import torch.nn as nn

from .utils import get_param_names_to_merge


class TaskVector:
    """
    A vector of parameter differences (finetuned - pretrained) that supports:
      - addition: tv3 = tv1 + tv2
      - scaling:  tv2 = 0.5 * tv1, tv1 * 2, tv1 / 3, tv1 *= alpha, tv1 /= alpha
      - gradient-safe scalar tensors (len-1 or 0-dim) for scaling

    If with_grad=False (default), operations run under torch.no_grad().
    If with_grad=True, operations are grad-tracked as usual.
    """

    def __init__(
        self,
        pretrained_model: nn.Module = None,
        finetuned_model: nn.Module = None,
        exclude_param_names_regex: str = None,
        task_vector_param_dict: dict[str, torch.Tensor] = None,
        with_grad: bool = False,
    ):
        self.with_grad = with_grad
        self.context = torch.no_grad() if not with_grad else nullcontext()

        if task_vector_param_dict is not None:
            # Direct construction
            self.task_vector_param_dict = task_vector_param_dict
        else:
            # Build from pretrained/finetuned models
            self.task_vector_param_dict = {}
            if pretrained_model is None or finetuned_model is None:
                raise ValueError(
                    "Must provide either task_vector_param_dict or both pretrained_model and finetuned_model."
                )

            pretrained_param_dict = {
                name: p for name, p in pretrained_model.named_parameters()
            }
            finetuned_param_dict = {
                name: p for name, p in finetuned_model.named_parameters()
            }

            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(pretrained_param_dict.keys()),
                exclude_param_names_regex=exclude_param_names_regex,
            )

            with self.context:
                for name in param_names_to_merge:
                    self.task_vector_param_dict[name] = (
                        finetuned_param_dict[name] - pretrained_param_dict[name]
                    )

    # ---------- helpers ----------
    @staticmethod
    def _ensure_ref_param(tv_dict: dict) -> torch.Tensor:
        try:
            return next(iter(tv_dict.values()))
        except StopIteration:
            raise ValueError("TaskVector is empty; no parameters found.")

    @staticmethod
    def _as_scalar_tensor(scalar, ref_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a Python number or a scalar tensor (0-dim or length-1)
        to a tensor on the same device/dtype as ref_tensor.
        - For Python numbers: creates a constant tensor (no grad).
        - For torch tensors: preserves autograd graph (no .item()/float()).
        """
        if isinstance(scalar, numbers.Number):
            return torch.as_tensor(
                scalar, dtype=ref_tensor.dtype, device=ref_tensor.device
            )

        if torch.is_tensor(scalar):
            if scalar.numel() != 1:
                raise ValueError(
                    "Scalar multiplication/division requires a 0-dim or length-1 tensor."
                )
            # .to(...) preserves graph
            return scalar.to(dtype=ref_tensor.dtype, device=ref_tensor.device)

        raise TypeError(
            f"Expected a Python number or scalar tensor; got {type(scalar)}."
        )

    # ---------- arithmetic ----------
    def __add__(self, other):
        if other == 0:
            # allows sum([tv1, tv2], start=0)
            return self
        assert isinstance(other, TaskVector), \
            "TaskVector can only be added to another TaskVector."
        new_dict = {}
        with self.context:
            for name, v in self.task_vector_param_dict.items():
                assert name in other.task_vector_param_dict, \
                    f"param_name {name} not present in both vectors."
                new_dict[name] = v + other.task_vector_param_dict[name]
        return TaskVector(task_vector_param_dict=new_dict, with_grad=self.with_grad)

    def __radd__(self, other):
        # supports sum([...]) where Python starts with 0
        if other == 0:
            return self
        return self.__add__(other)

    def __neg__(self):
        return self * (-1.0)

    def __mul__(self, scalar):
        ref = self._ensure_ref_param(self.task_vector_param_dict)
        s = self._as_scalar_tensor(scalar, ref)
        new_dict = {}
        with self.context:
            for k, v in self.task_vector_param_dict.items():
                new_dict[k] = v * s  # broadcasted; grad flows into s if s has grad
        return TaskVector(task_vector_param_dict=new_dict, with_grad=self.with_grad)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        ref = self._ensure_ref_param(self.task_vector_param_dict)
        s = self._as_scalar_tensor(scalar, ref)
        s_inv = 1.0 / s  # keep as tensor; preserves grad if s has grad
        new_dict = {}
        with self.context:
            for k, v in self.task_vector_param_dict.items():
                new_dict[k] = v * s_inv
        return TaskVector(task_vector_param_dict=new_dict, with_grad=self.with_grad)

    # in-place variants
    def __imul__(self, scalar):
        ref = self._ensure_ref_param(self.task_vector_param_dict)
        s = self._as_scalar_tensor(scalar, ref)
        with self.context:
            for k in list(self.task_vector_param_dict.keys()):
                self.task_vector_param_dict[k] = self.task_vector_param_dict[k] * s
        return self

    def __itruediv__(self, scalar):
        ref = self._ensure_ref_param(self.task_vector_param_dict)
        s = self._as_scalar_tensor(scalar, ref)
        inv = 1.0 / s
        with self.context:
            for k in list(self.task_vector_param_dict.keys()):
                self.task_vector_param_dict[k] = self.task_vector_param_dict[k] * inv
        return self

    # ---------- application ----------
    def combine_with_pretrained_model(
        self,
        pretrained_model: nn.Module,
        scaling_coefficient=1.0,
    ):
        """
        Returns a dict[param_name] -> merged tensor (pretrained + scale * delta).
        Does not mutate the model; you can load these tensors into a model if desired.
        """
        pretrained_param_dict = {
            name: p for name, p in pretrained_model.named_parameters()
        }
        with self.context:
            merged_params = {}
            for name, delta in self.task_vector_param_dict.items():
                merged_params[name] = (
                    pretrained_param_dict[name]
                    + self._as_scalar_tensor(
                        scaling_coefficient, pretrained_param_dict[name]
                    )
                    * delta
                )
        return merged_params

    def combine_with_base_tensor(
        self,
        base_tensor: dict[str, torch.Tensor],
        scaling_coefficient=1.0,
    ):
        """
        Same as combine_with_pretrained_model, but with a plain dict base.
        """
        # pick any tensor from base to place the scalar properly
        ref = self._ensure_ref_param(base_tensor)
        s = self._as_scalar_tensor(scaling_coefficient, ref)
        with self.context:
            merged_params = {}
            for name, delta in self.task_vector_param_dict.items():
                ref = base_tensor[name]
                delta = delta.to(device=ref.device, dtype=ref.dtype)
                merged_params[name] = ref + s * delta
        return merged_params
