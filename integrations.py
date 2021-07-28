# Copyright 2020 The HuggingFace Team. All rights reserved.
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
Integrations with other Python libraries.
"""
import importlib.util
import numbers
import os
import tempfile
from pathlib import Path

from utils import logging


logger = logging.get_logger(__name__)


# comet_ml requires to be imported before any ML frameworks
_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"
if _has_comet:
    try:
        import comet_ml  # noqa: F401

        if hasattr(comet_ml, "config") and comet_ml.config.get_config("comet.api_key"):
            _has_comet = True
        else:
            if os.getenv("COMET_MODE", "").upper() != "DISABLED":
                logger.warning("comet_ml is installed but `COMET_API_KEY` is not set.")
            _has_comet = False
    except (ImportError, ValueError):
        _has_comet = False

from file_utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402


# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


def is_comet_available():
    return _has_comet


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


def is_ray_available():
    return importlib.util.find_spec("ray") is not None


def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


def is_fairscale_available():
    return importlib.util.find_spec("fairscale") is not None


def is_neptune_available():
    return importlib.util.find_spec("neptune") is not None


def hp_params(trial):
    if is_optuna_available():
        import optuna

        if isinstance(trial, optuna.Trial):
            return trial.params
    if is_ray_tune_available():
        if isinstance(trial, dict):
            return trial

    raise RuntimeError(f"Unknown type for trial {trial.__class__}")


def default_hp_search_backend():
    if is_optuna_available():
        return "optuna"
    elif is_ray_tune_available():
        return "ray"


def get_available_reporting_integrations():
    integrations = []
    if is_azureml_available():
        integrations.append("azure_ml")
    if is_comet_available():
        integrations.append("comet_ml")
    if is_mlflow_available():
        integrations.append("mlflow")
    if is_tensorboard_available():
        integrations.append("tensorboard")
    if is_wandb_available():
        integrations.append("wandb")
    return integrations


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d

