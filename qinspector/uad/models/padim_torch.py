import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, wide_resnet50_2


from qinspector.cvlib.workspace import register

models = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet50_2": wide_resnet50_2
}


@register
class ResNet_PaDiM_torch(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True):
        super(ResNet_PaDiM_torch, self).__init__()
        assert arch in models.keys(), 'arch {} not supported'.format(arch)
        self.model = models[arch](pretrained)

    def forward(self, x):
        res = []
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            res.append(x)
            x = self.model.layer2(x)
            res.append(x)
            x = self.model.layer3(x)
            res.append(x)
        return res
