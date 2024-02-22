import torchvision
from torch import nn


'''
模型：mobilenet_v3_small
'''
class MobileNetV3s_1(nn.Module):
    # out channel 24
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class MobileNetV3s_2(nn.Module):
    # out 48 channel
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        modules = list(model.children())
        modules = modules[0][4:9]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class MobileNetV3s_3(nn.Module):
    # out 576 channel
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        modules = list(model.children())
        modules = modules[0][9:]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

'''
模型：RegNety400
'''
class RegNety400_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.regnet_y_400mf()
        modules = list(model.children())
        self.model = nn.Sequential(modules[0], *modules[1][:2])

    def forward(self, x):
        return self.model(x)

class RegNety400_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.regnet_y_400mf()
        modules = list(model.children())
        modules = modules[1][2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class RegNety400_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.regnet_y_400mf()
        modules = list(model.children())
        modules = modules[1][3]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


'''
模型：mobilenet_v2
'''
class mobilenet_v2_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v2()
        modules = list(model.children())
        modules = modules[0][:7]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class mobilenet_v2_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v2()
        modules = list(model.children())
        modules = modules[0][7:14]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class mobilenet_v2_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v2()
        modules = list(model.children())
        modules = modules[0][14:19]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

'''
模型：convnext_tiny
'''
class convnext_tiny_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.convnext_tiny()
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class convnext_tiny_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.convnext_tiny()
        modules = list(model.children())
        modules = modules[0][4:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class convnext_tiny_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.convnext_tiny()
        modules = list(model.children())
        modules = modules[0][6:8]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)