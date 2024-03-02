import torchvision
from torch import nn
import torch


class shufflenet_v2_x0_5(nn.Module):
    # out channel 24
    def __init__(self, *args,**kwargs) -> None:
        super().__init__()
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
        modules = list(model.children())[:-2]
        self.stem = nn.Sequential(*modules[:-3])
        self.stage1 = modules[-3]
        self.stage2 = modules[-2]
        self.stage3 = modules[-1]

        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
    def forward(self, x):
        out = []
        x = self.stem(x)
        for stage in [self.stage1, self.stage2, self.stage3]:
            x = stage(x) 
            out.append(x)
        return out
