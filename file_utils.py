# Copyright 2020 The HuggingFace Team, the AllenNLP library authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for working with the local dataset cache. Parts of this file is adapted from the AllenNLP library at
https://github.com/allenai/allennlp.
"""
import importlib.util
import os
import re
import types
import functools
import sys
import json
import requests


from collections import OrderedDict
from utils import logging
from types import ModuleType
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
from utils.versions import importlib_metadata
from urllib.parse import urlparse
from pathlib import Path
from huggingface_hub import HfApi, HfFolder, Repository
# from __init__ import __version__
from uuid import uuid4
from hashlib import sha256


__version__ = "4.9.0.dev0"
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logger.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    logger.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False

if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if version.parse(_tf_version) < version.parse("2"):
            logger.info(f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum.")
            _tf_available = False
        else:
            logger.info(f"TensorFlow version {_tf_version} available.")
else:
    logger.info("Disabling Tensorflow because USE_TORCH is set")
    _tf_available = False


_datasets_available = importlib.util.find_spec("datasets") is not None

hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "transformers")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://moon-staging.huggingface.co" if _staging_mode else "https://huggingface.co"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", _default_endpoint)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"

PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    return _is_offline_mode


def is_torch_available():
    return _torch_available


def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def is_tf_available():
    return _tf_available


def is_vision_available():
    return importlib.util.find_spec("PIL") is not None


def is_datasets_available():
    return _datasets_available


# docstyle-ignore
DATASETS_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip i

BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
    ]
)
"""

# docstyle-ignore
TENSORFLOW_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip i

BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
    ]
)
"""

# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip i

BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
    ]
)
"""

# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip i

BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
    ]
)
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
    ]
)


PT_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(torch.FloatTensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`torch.FloatTensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising
        various elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        :class:`~{full_output_type}` or :obj:`tuple(tf.Tensor)`: A :class:`~{full_output_type}` or a tuple of
        :obj:`tf.Tensor` (if ``return_dict=False`` is passed or when ``config.return_dict=False``) comprising various
        elements depending on the configuration (:class:`~transformers.{config_class}`) and inputs.

"""


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    if not all(BACKENDS_MAPPING[backend][0]() for backend in backends):
        raise ImportError("".join([BACKENDS_MAPPING[backend][1].format(name) for backend in backends]))


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        class_name = f":class:`~transformers.{fn.__qualname__.split('.')[0]}`"
        intro = f"   The {class_name} forward method, overrides the :func:`__call__` special method."
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within this function, one should call the
        :class:`Module` instance afterwards instead of this since the former takes care of running the pre and post
        processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files. If True,
            will get token from ~/.huggingface.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(f"Archive format of {output_path} could not be identified")

        return output_path_extracted

    return output_path


def hf_bucket_url(
    model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if mirror:
        endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)
        legacy_format = "/" not in model_id
        if legacy_format:
            return f"{endpoint}/{model_id}-{filename}"
        else:
            return f"{endpoint}/{model_id}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


def _prepare_output_docstrings(output_type, config_class):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    docstrings = output_type.__doc__

    # Remove the head of the docstring to keep the list of args only
    lines = docstrings.split("\n")
    i = 0
    while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
        i += 1
    if i < len(lines):
        docstrings = "\n".join(lines[(i + 1) :])
        docstrings = _convert_output_args_doc(docstrings)

    # Add the return introduction
    full_output_type = f"{output_type.__module__}.{output_type.__name__}"
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    return intro + docstrings


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            lines[i] = _prepare_output_docstrings(output_type, config_class)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename



def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
            r.raise_for_status()
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Otherwise, our Internet connection is down.
            # etag is None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise FileNotFoundError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}")

            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)

        logger.info(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def is_training_run_on_sagemaker():
    return "SAGEMAKER_JOB_NAME" in os.environ


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(import_structure.values(), [])
        self.__file__ = module_file
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        return super().__dir__() + self.__all__

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        return importlib.import_module("." + module_name, self.__name__)

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
