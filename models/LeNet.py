import torch.nn as nn
# from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.quantization_RRAM import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
from modules.SC_infer import SCConv2d, SCLinear
import torch

class LeNet5(nn.Module):
    def __init__(self, args, num_classes):
        super(LeNet5, self).__init__()
        self.args = args

        # Original LeNet-5
        # self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=5//2, bias=False)
        # self.conv2 = nn.Conv2d(6, 10, kernel_size=5, stride=1, padding=5//2, bias=False)
        # self.fc1 = nn.Linear(640, 120, bias=False)
        # self.fc2 = nn.Linear(120, 84, bias=False)
        # self.fc3 = nn.Linear(84, num_classes, bias=False)
        
        if self.args.mode == "WAGE":
            self.conv1 = QConv2d(1, 6, kernel_size=5, padding=5//2,
                                    wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                                    RRAM=args.RRAM,
                                    subArray = args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                                    name='Conv0_', model=args.model)
            self.conv2 = QConv2d(6, 10, kernel_size=5, padding=5//2,
                                    wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                                    RRAM=args.RRAM,
                                    subArray = args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                                    name='Conv1_', model=args.model)
            self.fc1 = QLinear(640, 120,
                                wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                                RRAM=args.RRAM,
                                subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                                name='FC2_', model=args.model)
            self.fc2 = QLinear(120, 84,
                                wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                                RRAM=args.RRAM,
                                subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                                name='FC3_', model=args.model)
            self.fc3 = QLinear(84, num_classes,
                                wl_input=args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,wl_weight=args.wl_weight,
                                RRAM=args.RRAM,
                                subArray=args.subArray, ADCprecision=args.ADCprecision, onoffratio=args.onoffratio, vari=args.vari, t=args.t, v=args.v, detect=args.detect, target=args.target,
                                name='FC4_', model=args.model)
        
        elif self.args.mode == "SC":
            self.conv1 = SCConv2d(1, 6, kernel_size=5, padding=5//2, len=args.SClen, APC_relu=True)
            self.conv2 = SCConv2d(6, 10, kernel_size=5, padding=5//2, len=args.SClen, APC_relu=True)
            self.fc1 = SCLinear(640, 120, len=args.SClen, APC_relu=True)
            self.fc2 = SCLinear(120, 84, len=args.SClen, APC_relu=True)
            self.fc3 = SCLinear(84, num_classes, len=args.SClen, APC_relu=False)

    def forward(self, x):

        if self.args.mode == "WAGE":
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, kernel_size=2, stride=2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

        elif self.args.mode == "SC":
            x = self.conv1(x)
            x = torch.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = torch.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(x.size(0), -1)
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)

        return x