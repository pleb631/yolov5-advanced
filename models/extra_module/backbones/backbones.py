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
模型：efficientnet_b0
'''
class efficientnet_b0_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_b0()
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class efficientnet_b0_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_b0()
        modules = list(model.children())
        modules = modules[0][4:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class efficientnet_b0_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_b0()
        modules = list(model.children())
        modules = modules[0][6:]
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
模型：resnet18
'''
class resnet18_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet18()
        modules = list(model.children())
        modules = modules[:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet18_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet18()
        modules = list(model.children())
        modules = modules[6:7]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet18_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet18()
        modules = list(model.children())
        modules = modules[7:8]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

'''
模型：resnet34
'''
class resnet34_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet34()
        modules = list(model.children())
        modules = modules[:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet34_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet34()
        modules = list(model.children())
        modules = modules[6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet34_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet34()
        modules = list(model.children())
        modules = modules[7]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

'''
模型：resnet50
'''
class resnet50_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())
        modules = modules[:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet50_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())
        modules = modules[6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet50_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())
        modules = modules[7]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    

'''
模型：efficientnet_v2_s
'''
class efficientnet_v2_s_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_v2_s()
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class efficientnet_v2_s_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_v2_s()
        modules = list(model.children())
        modules = modules[0][4:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class efficientnet_v2_s_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_v2_s()
        modules = list(model.children())
        modules = modules[0][6:]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
'''
模型：efficientnet_b1
'''
class efficientnet_b1_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_b1()
        modules = list(model.children())
        modules = modules[0][:4]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class efficientnet_b1_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_b1()
        modules = list(model.children())
        modules = modules[0][4:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class efficientnet_b1_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.efficientnet_b0()
        modules = list(model.children())
        modules = modules[0][6:]
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
模型：wide_resnet50_2
'''
class wide_resnet50_2_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.wide_resnet50_2()
        modules = list(model.children())
        modules = modules[:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class wide_resnet50_2_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.wide_resnet50_2()
        modules = list(model.children())
        modules = modules[6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class wide_resnet50_2_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.wide_resnet50_2()
        modules = list(model.children())
        modules = modules[7]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

'''
模型：vgg11_bn
'''
class vgg11_bn_1(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.vgg11_bn()
        modules = list(model.children())
        modules = modules[0][:14]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class vgg11_bn_2(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.vgg11_bn()
        modules = list(model.children())
        modules = modules[0][14:21]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class vgg11_bn_3(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = torchvision.models.vgg11_bn()
        modules = list(model.children())
        modules = modules[0][21:28]
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