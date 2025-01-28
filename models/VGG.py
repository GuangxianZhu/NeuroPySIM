import torch.nn as nn
# from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.quantization_RRAM import QConv2d, QLinear

from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch

class VGG(nn.Module):
    def __init__(self, args, features, num_classes):
        super(VGG, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        if args.model == 'VGG8':
            self.classifier = make_layers([('L', 8192, 1024),
                                        ('L', 1024, num_classes)], 
                                        args)
        elif args.model == 'VGG16':
            self.classifier = make_layers([('L', 512, 512),
                                        ('L', 512, 512),
                                        ('L', 512, num_classes)], 
                                        args)
        else:
            raise ValueError("Unknown model type")

        # print(self.features)
        # print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, args):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            if args.mode == "WAGE":
                conv2d = QConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                                wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                                RRAM=args.RRAM,
                                subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                                name='Conv'+str(i)+'_', model=args.model)
                
            elif args.mode == "FP":
                conv2d = FConv2d(in_channels, out_channels, kernel_size=v[2], padding=padding,
                                wl_input=args.wl_activate,wl_weight=args.wl_weight,
                                inference=args.inference,ADCprecision=args.ADCprecision,
                                t=args.t,v=args.v,detect=args.detect,target=args.target,name='Conv'+str(i)+'_')
                
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d, non_linearity_activation]
            in_channels = out_channels
        if v[0] == 'L':
            if args.mode == "WAGE":
                linear = QLinear(in_features=v[1], out_features=v[2], 
                                wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                                RRAM=args.RRAM,
                                subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                                name='FC'+str(i)+'_', model=args.model)
                
            elif args.mode == "FP":
                linear = FLinear(in_features=v[1], out_features=v[2],
                                wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,
                                ADCprecision=args.ADCprecision,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                name='FC'+str(i)+'_')
                
            if i < len(cfg)-1:
                non_linearity_activation =  nn.ReLU()
                layers += [linear, non_linearity_activation]
            else:
                layers += [linear]
    return nn.Sequential(*layers)



# cfg_list = {
#     'vgg8': [   ('C', 128, 3, 'same'),  # conv, out_channels, kernel_size, padding
#                 ('C', 128, 3, 'same'),
#                 ('M', 2, 2),            # maxpool, kernel_size, stride
#                 ('C', 256, 3, 'same'),
#                 ('C', 256, 3, 'same'),
#                 ('M', 2, 2),
#                 ('C', 512, 3, 'same'),
#                 ('C', 512, 3, 'same'),
#                 ('M', 2, 2)]
# }

cfg_list = {
    'vgg8': [
        ('C', 128, 3, 'same'),  # conv, out_channels, kernel_size, padding
        ('C', 128, 3, 'same'),
        ('M', 2, 2),            # maxpool, kernel_size, stride
        ('C', 256, 3, 'same'),
        ('C', 256, 3, 'same'),
        ('M', 2, 2),
        ('C', 512, 3, 'same'),
        ('C', 512, 3, 'same'),
        ('M', 2, 2)
    ],
    'vgg16': [
        ('C', 64, 3, 'same'),
        ('C', 64, 3, 'same'),
        ('M', 2, 2),
        ('C', 128, 3, 'same'),
        ('C', 128, 3, 'same'),
        ('M', 2, 2),
        ('C', 256, 3, 'same'),
        ('C', 256, 3, 'same'),
        ('C', 256, 3, 'same'),
        ('M', 2, 2),
        ('C', 512, 3, 'same'),
        ('C', 512, 3, 'same'),
        ('C', 512, 3, 'same'),
        ('M', 2, 2),
        ('C', 512, 3, 'same'),
        ('C', 512, 3, 'same'),
        ('C', 512, 3, 'same'),
        ('M', 2, 2)
    ]
}

def vgg8( args, pretrained=None):
    cfg = cfg_list['vgg8']
    layers = make_layers(cfg, args)
    model = VGG(args, layers, num_classes=10)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location=torch.device('cpu'), weights_only=True))
    return model

def vgg16(args, pretrained=None):
    cfg = cfg_list['vgg16']
    layers = make_layers(cfg, args)
    model = VGG(args, layers, num_classes=10)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location=torch.device('cpu'), weights_only=True))
    return model

