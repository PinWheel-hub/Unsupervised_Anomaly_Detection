# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from qinspector.cvlib.workspace import register

models = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
}


@register
class ResNet_MS3_torch(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True):
        super(ResNet_MS3_torch, self).__init__()
        assert arch in models.keys(), 'arch {} not supported'.format(arch)
        net = models[arch](pretrained=pretrained)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in enumerate(self.model):
            x = module(x)
            if name in [4, 5, 6]:
                res.append(x)
        return res


class ResNet_MS3_EXPORT_torch(nn.Module):
    def __init__(self, student, teacher):
        super(ResNet_MS3_EXPORT_torch, self).__init__()
        self.student = student
        self.teacher = teacher

    def forward(self, x):
        result = []
        result.append(self.student(x))
        result.append(self.teacher(x))
        return result
