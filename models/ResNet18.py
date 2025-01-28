import torch
import torch.nn as nn
import torch.nn.functional as F
# from modules.quantization_cpu_np_infer import QConv2d, QLinear
from modules.quantization_RRAM import QConv2d, QLinear

class BasicBlock(nn.Module):
    '''
    replace standard conv and linear to QConv2d and QLinear
    '''
    expansion = 1

    def __init__(self, args, in_channels, out_channels, stride=1, padding=0, layer_id=0, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = QConv2d(in_channels, out_channels, kernel_size=3, padding=padding,
                            wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                            RRAM=args.RRAM,
                            subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                            name='Conv'+str(layer_id)+'_', model=args.model)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = QConv2d(out_channels, out_channels, kernel_size=3, padding=padding,
                            wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                            RRAM=args.RRAM,
                            subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                            name='Conv'+str(layer_id+1)+'_', model=args.model)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, args, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.layer_id = 0  # Initialize layer ID

        # Initial convolution and pooling
        self.conv1 = QConv2d(3, 64, kernel_size=7, padding=3,
                             wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error, wl_weight=args.wl_weight,
                             RRAM=args.RRAM, subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio,
                             vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                             name=f'Conv{self.layer_id}_', model=args.model)
        self.layer_id += 1  # Increment after creating a layer
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(args, block, 64, layers[0])
        self.layer2 = self._make_layer(args, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(args, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(args, block, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QLinear(512 * block.expansion, num_classes,  # Use QLinear here
                          wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error, wl_weight=args.wl_weight,
                          RRAM=args.RRAM, subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio,
                          vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                          name=f'FC{self.layer_id}_', model=args.model)
        self.layer_id += 1  # Increment for the final layer

    def _make_layer(self, args, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                QConv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, padding=0,
                        wl_input=args.wl_activate, wl_activate=args.wl_activate, wl_error=args.wl_error, wl_weight=args.wl_weight,
                        RRAM=args.RRAM, subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio,
                        vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                        name=f'Conv{self.layer_id}_', model=args.model),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
            self.layer_id += 1  # Increment for the downsample layer

        layers = [block(args, self.in_channels, out_channels, stride, padding=1, layer_id=self.layer_id, downsample=downsample)]
        self.layer_id += 2  # Increment for the block (assumes 2 conv layers inside block)
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(args, self.in_channels, out_channels, padding=1, layer_id=self.layer_id))
            self.layer_id += 2  # Increment for each additional block

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(args, num_classes=1000):
    return ResNet(args, BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
