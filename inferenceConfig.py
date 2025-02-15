import argparse

parser = argparse.ArgumentParser(description='SuperSIM config.')

# dataset and model
parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|mnist|imagenet')
parser.add_argument('--model', default='VGG8', help='VGG8|DenseNet40|LeNet5|ViT')

# training parameters
parser.add_argument('--batch_size', type=int,   default=200, help='input batch size for training (default: 64)')
parser.add_argument('--seed',       type=int,   default=1, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir',                 default='log/default', help='folder to save to the log')

# data format and precision
parser.add_argument('--mode',       default='WAGE', help='Quantization: WAGE|FP')
parser.add_argument('--wl_weight',  type=int, default=8)
parser.add_argument('--wl_activate',type=int, default=8)
parser.add_argument('--wl_error',   type=int, default=8)

# architecture config
parser.add_argument('--subArray', type=int, default=128, help='size of subArray, # of rows')

# RRAM configuration
parser.add_argument('--RRAM',       type=str,   default='NoRRAM', help='RRAM_analog|RRAM_digital|NoRRAM')
parser.add_argument('--ADCprecision', type=int, default=5)
parser.add_argument('--onoffratio', type=float, default=10., help='device on/off ratio (e.g. Gmax/Gmin = 3)')
parser.add_argument('--vari',       type=float, default=0., help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t',          type=float, default=0., help='retention time')
parser.add_argument('--v',          type=float, default=0., help='drift coefficient')
parser.add_argument('--detect',     type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target',     type=float, default=0., help='drift target for fixed-direction drift, range 0-1')

args = parser.parse_args()


if __name__ == '__main__':

    from models import LeNet
    args_jupyter = parser.parse_args()
    modelCF = LeNet.LeNet5(args = args_jupyter, num_classes = 10)
    print(modelCF)
    num_params = sum(p.numel() for p in modelCF.parameters())
    print("lenet5: ", num_params / 1e6)

    from models import VGG
    args_jupyter.model = 'VGG8'
    model_path = './log/VGG8.pth'
    modelCF = VGG.vgg8(args = args_jupyter, pretrained = model_path)
    print(modelCF)
    num_params = sum(p.numel() for p in modelCF.parameters())
    print("vgg8: ", num_params / 1e6)

    from models import DenseNet
    args_jupyter.model = 'DenseNet40'
    model_path = './log/DenseNet40.pth'     # WAGE mode pretrained model
    modelCF = DenseNet.densenet40(args = args_jupyter, pretrained = model_path)
    print(modelCF)
    num_params = sum(p.numel() for p in modelCF.parameters())
    print("densenet40: ", num_params / 1e6)