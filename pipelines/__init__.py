# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from ..configuration_utils import PretrainedConfig
# from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import is_tf_available, is_torch_available
# from ..models.auto.configuration_auto import AutoConfig
# from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
from ..utils import logging
from .base import (
    Pipeline,
    get_default_model,
    infer_framework_load_model
)
from .image_classification import ImageClassificationPipeline


if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import (
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
        TFAutoModel,
        TFAutoModelForCausalLM,
        TFAutoModelForMaskedLM,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSeq2SeqLM,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
    )

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import (
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForImageClassification,
        AutoModelForMaskedLM,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification,
    )
if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


SUPPORTED_TASKS = {
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "tf": (),
        "pt": (AutoModelForImageClassification,) if is_torch_available() else (),
        "default": {"model": {"pt": "google/vit-base-patch16-224"}},
    },
}


def check_task(task: str) -> Tuple[Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`
            - :obj:`"text-classification"`
            - :obj:`"sentiment-analysis"` (alias of :obj:`"text-classification")
            - :obj:`"token-classification"`
            - :obj:`"ner"` (alias of :obj:`"token-classification")
            - :obj:`"question-answering"`
            - :obj:`"fill-mask"`
            - :obj:`"summarization"`
            - :obj:`"translation_xx_to_yy"`
            - :obj:`"translation"`
            - :obj:`"text-generation"`
            - :obj:`"conversational"`

    Returns:
        (task_defaults:obj:`dict`, task_options: (:obj:`tuple`, None)) The actual dictionary required to initialize the
        pipeline and some extra task options for parametrized tasks like "translation_XX_to_YY"


    """
    if task in SUPPORTED_TASKS:
        targeted_task = SUPPORTED_TASKS[task]
        return targeted_task, None

    raise KeyError(
        f"Unknown task {task}, available tasks are {list(SUPPORTED_TASKS.keys())}"
    )


def pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    framework: Optional[str] = None,
    model_kwargs: Dict[str, Any] = {},
    **kwargs
) -> Pipeline:
    # Retrieve the task
    targeted_task, task_options = check_task(task)
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        # At that point framework might still be undetermined
        model = get_default_model(targeted_task, framework, task_options)

    # Config is the primordial information item.
    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config, revision=revision, _from_pipeline=task, **model_kwargs)
    elif config is None and isinstance(model, str):
        config = AutoConfig.from_pretrained(model, revision=revision, _from_pipeline=task, **model_kwargs)

    model_name = model if isinstance(model, str) else None

    # Infer the framework from the model
    # Forced if framework already defined, inferred if it's None
    # Will load the correct model if possible
    model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]}
    framework, model = infer_framework_load_model(
        model,
        model_classes=model_classes,
        config=config,
        framework=framework,
        task=task,
        **model_kwargs
    )

    return task_class(model=model, framework=framework, task=task, **kwargs)
