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
from .auto_factory import auto_class_factory
from .configuration_auto import (
    ViTConfig,
)


logger = logging.get_logger(__name__)


MODEL_MAPPING = OrderedDict(
    [
        (ViTConfig, ViTModel),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Image Classification mapping
        (ViTConfig, ViTForImageClassification),
    ]
)


AutoModel = auto_class_factory("AutoModel", MODEL_MAPPING)


AutoModelForImageClassification = auto_class_factory(
    "AutoModelForImageClassification", MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, head_doc="image classification"
)