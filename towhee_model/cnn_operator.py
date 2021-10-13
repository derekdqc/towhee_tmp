# Copyright 2021 Zilliz. All rights reserved.
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


from abc import abstractclassmethod, ABC
from typing import Any

import torchvision.models
from torch import Tensor


class Operator:
    """
    The base operator class.
    """

    def __init__(self):
        self._params = {}
        self.name = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, dict):
        # todo update parameter dict
        raise NotImplementedError

    @abstractclassmethod
    def op_class(cls):
        """
        Get the interface Operator class
        """
        raise NotImplementedError


class PyTorchOperator(Operator, ABC):
    """
    The pytorch operator class
    """
    def __init__(self):
        super().__init__()
        self.model_name = None

    def resnet50img2vec(self):
        # model should be network class code
        """
        Examples:
            model = ResNet50Img2Vec()
            return model.resnet50()
        """

    def train_start(self):
        """
        Examples:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
            )
            trainer.train()
        """


class ResNet50Img2Vec:
    def __init__(self):
        pass

    def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any):
        r"""ResNet-50 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr

        Examples:
            return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                       **kwargs)
        """


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
):
    """
    Examples:
        model = ResNet(block, layers, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                        progress=progress)
            model.load_state_dict(state_dict)
        return model

    """


class ResNet(ResNet50Img2Vec, ABC):
    """
    The underlying component of resnet class

    Examples:
        def __init__(self):
            super().__init__()

        def _make_layer(self):
            raise NotImplementedError

        def _forward_impl(self):
            raise NotImplementedError

        def forward(self, x: Tensor) -> Tensor:
            return self._forward_impl(x)
    """



