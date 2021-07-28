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
""" Auto Model class. """


import warnings
from collections import OrderedDict

from utils import logging

# Add modeling imports here
from ..vit.modeling_vit import ViTForImageClassification, ViTModel
from .auto_factory import _BaseAutoModelClass, auto_class_update

from .configuration_auto import (
    ViTConfig,
)


logger = logging.get_logger(__name__)


MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Image Classification mapping
        (ViTConfig, ViTForImageClassification),
    ]
)

MODEL_MAPPING = OrderedDict(
    [
        ]
)

MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        ]
)


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")

