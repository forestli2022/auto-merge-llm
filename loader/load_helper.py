import codecs
import collections
import contextlib
import json
import operator
import os
import os.path
import pickle
import zipfile
from functools import reduce
from typing import Any, Optional, Tuple, Union, Sequence, Dict

import accelerate
import numpy
import torch
from pydantic import BaseModel, PrivateAttr
import safetensors
import safetensors.torch
import huggingface_hub

from utils import get_model_storage_path, logger


ACCEPTABLE_TYPES = {
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("numpy.core.multiarray", "scalar"): numpy.core.multiarray.scalar,
    ("numpy", "dtype"): numpy.core.multiarray.scalar,
    ("_codecs", "encode"): codecs.encode,
    **{
        ("torch", name): getattr(torch, name)
        for name in [
            "DoubleStorage",
            "FloatStorage",
            "HalfStorage",
            "LongStorage",
            "IntStorage",
            "ShortStorage",
            "CharStorage",
            "ByteStorage",
            "BoolStorage",
            "BFloat16Storage",
        ]
    },
}


class DeferredLoad(BaseModel, arbitrary_types_allowed=True):
    name: str
    location: str
    dtype: torch.dtype

    # set after construction by rebuild()
    file_offset: Optional[int] = None
    shape: Optional[Union[torch.Size, Tuple[int, ...]]] = None
    stride: Optional[Tuple[int, ...]] = None

    # set arbitrarily in Torch innards
    requires_grad: bool = False
    _backward_hooks: Any = PrivateAttr(None)

    @staticmethod
    def rebuild(
        load: "DeferredLoad",
        offset: int,
        shape: Union[torch.Size, Tuple[int, ...]],
        stride: Tuple[int, ...],
    ) -> "DeferredLoad":
        load.shape = shape
        load.stride = stride
        load.file_offset = offset * dtype_bytes(load.dtype)
        return load

    def execute(
        self,
        reader: "TorchArchiveReader",
        map_location: Any = None,
    ) -> torch.Tensor:
        total_params = reduce(operator.mul, self.shape)
        total_bytes = total_params * dtype_bytes(self.dtype)

        f = reader.open_file(file_name=self.name, offset=self.file_offset)
        storage = torch.UntypedStorage.from_buffer(
            f.read(total_bytes), "little", dtype=self.dtype
        )
        storage = torch.serialization._get_restore_location(map_location)(
            storage, self.location
        )

        tensor = torch.tensor([], dtype=self.dtype, device=storage.device)
        tensor.set_(storage, 0, self.shape, self.stride)
        tensor.requires_grad = self.requires_grad
        tensor._backward_hooks = self._backward_hooks
        return tensor


class LazyTorchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ACCEPTABLE_TYPES:
            return ACCEPTABLE_TYPES[(module, name)]
        raise pickle.UnpicklingError(f"Unsupported type {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        if not isinstance(pid, tuple) or pid[0] != "storage":
            raise RuntimeError(f"Unpickling object with unexpected PID: {repr(pid)}")

        storage_type, key, location, _ = pid[1:]
        return DeferredLoad(name=key, location=location, dtype=get_dtype(storage_type))


class TorchArchiveReader:
    """
    Class for lazily reading (sections of) files from a torch ZIP archive.

    Maintains a handle to the most recently opened file for faster access with
    consecutive reads from the same file.
    """

    archive: zipfile.ZipFile
    archive_name: str
    file_name: Optional[str] = None
    file: Optional[zipfile.ZipExtFile] = None

    def __init__(self, path: str):
        self.archive = zipfile.ZipFile(path, mode="r")
        self.archive_name = os.path.basename(os.path.normpath(path)).split(".")[0]

    def open_file(self, file_name: str, offset: int = 0) -> zipfile.ZipExtFile:
        if self.file_name != file_name or (
            self.file is not None and self.file.tell() > offset
        ):
            if self.file is not None:
                self.file.close()

            try:
                fd = self.archive.open(f"archive/data/{file_name}", mode="r")
            except Exception:
                fd = self.archive.open(
                    f"{self.archive_name}/data/{file_name}", mode="r"
                )
            self.file = fd
            self.file_name = file_name

        skip_bytes = offset - self.file.tell()
        assert skip_bytes >= 0
        self.file.seek(skip_bytes, os.SEEK_CUR)

        return self.file


@contextlib.contextmanager
def torch_lazy_load():
    """
    Context manager under which `torch.load` will return a `DeferredLoad` instead
    of `torch.Tensor.`
    """
    old_unpickler = pickle.Unpickler
    old_load = pickle.load
    old_rebuild_tensor = torch._utils._rebuild_tensor
    try:

        def load_monkeypatch(*args, **kwargs):
            return pickle.Unpickler(*args, **kwargs).load()

        pickle.Unpickler = LazyTorchUnpickler
        pickle.load = load_monkeypatch
        torch._utils._rebuild_tensor = DeferredLoad.rebuild

        with accelerate.init_empty_weights():
            yield

    finally:
        torch._utils._rebuild_tensor = old_rebuild_tensor
        pickle.Unpickler = old_unpickler
        pickle.load = old_load


def dtype_bytes(dtype: torch.dtype) -> int:
    """Return the number of bytes used to store a single instance of `dtype`."""
    if dtype.is_floating_point:
        ti = torch.finfo(dtype)
    else:
        ti = torch.iinfo(dtype)
    return max(1, ti.bits // 8)


def get_dtype(storage_type: Any):
    if isinstance(storage_type, torch.dtype):
        return storage_type
    dtype = storage_type.dtype
    if not isinstance(dtype, torch.dtype):
        dtype = storage_type(0).dtype
    return dtype


class LazyPickleLoader():
    """Loader for pytorch files using a custom unpickler and vigorous monkeypatching."""

    zip_reader: TorchArchiveReader
    index: Dict[str, DeferredLoad]
    device: Optional[str] = None

    def __init__(self, path: str, device: Optional[str] = None):
        self.zip_reader = TorchArchiveReader(path)
        self.device = device
        with torch_lazy_load():
            self.index = torch.load(path)

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.index:
            raise KeyError(key)

        return self.index[key].execute(self.zip_reader, map_location=self.device)

    def keys(self) -> Sequence[str]:
        return self.index.keys()


def find_model_path(repo_id):
    model_name = repo_id.replace("/", "--")
    CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE')
    for root, dirs, files in os.walk(CACHE_DIR):
        for dir_name in dirs:
            if model_name in dir_name:
                model_path = os.path.join(root, dir_name)
                return model_path
    
    return None


def get_shards_from_disk(model_name):
    print("getting: ", model_name)
    base_path = get_model_storage_path(model_name)
    model_path = None
    for model_file_name in [
        "model.safetensors",
        "pytorch_model.bin",
    ]:
        candidate_path = os.path.join(base_path, model_file_name)
        if os.path.exists(candidate_path) or os.path.exists(
            candidate_path + ".index.json"
        ):
            model_path = candidate_path
            print("model_path:", candidate_path)
            break

    if not model_path:
        try:
            download_model(model_name)
            model_path = os.path.join(base_path, "model.safetensors")
        except:
            raise RuntimeError(f"Unable to find model files at {base_path}")

    is_safetensors = model_path.endswith(".safetensors")
    tensor_paths = None
    shards = []

    if os.path.exists(model_path + ".index.json"):
        # shared model - parse index
        with open(model_path + ".index.json", "r") as fd:
            weight_map = json.load(fd)["weight_map"]
        tensor_paths = weight_map
        shard_names = list(sorted(set(tensor_paths[e] for e in tensor_paths)))
        for shard_name in shard_names:
            info = {
                "shard_name": shard_name,
                "keys": [
                    key for key in tensor_paths 
                    if tensor_paths[key] == shard_name
                ],
            }
            shards.append(info)

    elif os.path.exists(model_path):
        shard_name = os.path.basename(model_path)

        # get list of tensors contained in single-file checkpoint
        if model_path.lower().endswith(".safetensors"):
            with safetensors.safe_open(model_path, framework="pt") as st:
                tensor_paths = {key: shard_name for key in st.keys()}
        else:
            # this is ugly but not much else can be done
            shard = torch.load(model_path, map_location="meta")
            if "state_dict" in shard:
                shard = shard["state_dict"]

            tensor_paths = {key: shard_name for key in shard}

        shards.append(
            {
                "shard_name": os.path.basename(model_path), 
                "keys": list(tensor_paths.keys())
            }
        )

    return is_safetensors, tensor_paths, shards, base_path


def download_model(model_name):
    CACHE_DIR = os.environ.get('TRANSFORMERS_CACHE')
    has_safetensors = any(
        fn.lower().endswith(".safetensors")
        for fn in huggingface_hub.list_repo_files(
            model_name, repo_type="model")
    )
    patterns = ["tokenizer.model", "*.json"]
    if has_safetensors:
        patterns.append("*.safetensors")
    else:
        patterns.append("*.bin")

    huggingface_hub.snapshot_download(
        model_name,
        cache_dir=CACHE_DIR,
        allow_patterns=patterns,
    )