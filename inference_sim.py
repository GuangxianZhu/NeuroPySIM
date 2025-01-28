import os
import time
from utee import misc
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utee import make_path
from models import dataset
from utee import hook
from datetime import datetime
from inferenceConfig import args

# Count weight parameters (excluding bias)
def count_weights(model):
    total_weights = 0
    for param in model.parameters():
        if param.requires_grad and len(param.shape) > 1:  # Weight parameters have more than one dimension
            total_weights += param.numel()
    return total_weights

if __name__ == '__main__':
    
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
    # args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])
    logname = make_path.makefile(args,['log_interval','test_interval','logdir','debug'])

    misc.logger.init(args.logdir, 'test_log' + current_time)
    logger = misc.logger.info

    misc.ensure_dir(args.logdir)
    logger("=================FLAGS==================")
    for k, v in args.__dict__.items():
        logger('{}: {}'.format(k, v))
    logger("========================================")

    # seed
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # data loader and model
    assert args.dataset in ['cifar10', 'cifar100', 'imagenet','mnist','fashionmnist'], "dataset {} is not supported".format(args.dataset)
    if args.dataset == 'cifar10':
        test_loader = dataset.get_cifar10(batch_size=args.batch_size, num_workers=1, train=False)
    elif args.dataset == 'cifar100':
        test_loader = dataset.get_cifar100(batch_size=args.batch_size, num_workers=1, train=False)
    elif args.dataset == 'imagenet':
        # test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1, train=False)
        raise ValueError("imagenet dataset is not supported")
    elif args.dataset == 'mnist':
        if args.mode == 'WAGE':
            train_loader, test_loader = dataset.get_mnist(batch_size=args.batch_size, num_workers=1)
        elif args.mode == 'SC': raise ValueError("mnist dataset is not supported in SC mode")
            # train_loader, test_loader = dataset.get_mnist_sc(batch_size=args.batch_size)
    elif args.dataset == 'fashionmnist':
        if args.mode == 'WAGE': train_loader, test_loader = dataset.get_fashionmnist(batch_size=args.batch_size, num_workers=1)
        elif args.mode == 'SC': raise ValueError("fashionmnist dataset is not supported in SC mode") # train_loader, test_loader = dataset.get_fashionmnist_sc(batch_size=args.batch_size)
    else:
        raise ValueError("Unknown dataset type")
        
    assert args.model in ['VGG8', 'VGG16', 'DenseNet40', 'ResNet18', 'LeNet5', 'Transformer'], \
                        "model {} is not supported".format(args.model)
    if args.model == 'VGG8':
        from models import VGG
        model_path = './log/VGG8.pth'   # WAGE mode pretrained model
        modelCF = VGG.vgg8(args = args, pretrained = model_path)
    elif args.model == 'VGG16':
        from models import VGG
        model_path = None
        modelCF = VGG.vgg16(args = args, pretrained = model_path)
    elif args.model == 'DenseNet40':
        from models import DenseNet
        model_path = './log/DenseNet40.pth'     # WAGE mode pretrained model
        modelCF = DenseNet.densenet40(args = args, pretrained = model_path)
    elif args.model == 'LeNet5':
        from models import LeNet
        model_path = None
        modelCF = LeNet.LeNet5(args = args, num_classes = 10)
        if model_path is not None: modelCF.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    elif args.model == 'ViT':
        from models.ViT import ViT
        model_path = None
        modelCF = ViT(args)
        
    elif args.model == 'ResNet18':
        from models.ResNet18 import resnet18
        modelCF = resnet18(num_classes=1000) # for imagenet
        
    else:
        raise ValueError("Unknown model type")

    if args.cuda:
        modelCF.cuda()
    
    # Get total weight parameters count (no bias)
    weight_count = count_weights(modelCF)
    print(f"\nTotal weight parameters (M, no bias): {weight_count / 1e6:.02f} \n\n")

    best_acc, old_file = 0, None
    t_begin = time.time()
    # ready to go
    modelCF.eval()

    test_loss = 0
    correct = 0
    total_num = 0
    trained_with_quantization = True

    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    j_word_count = 14
    e_word_count = 14

    print("****** Start Testing ****** \n")
    with torch.no_grad():
        for i, data_target in enumerate(test_loader):

            data, target = data_target
            data, target = Variable(data), Variable(target)

            if i==0 and args.mode == 'WAGE':
                hook_handle_list = hook.hardware_evaluation(modelCF, args.model, args.mode)
            
            indx_target = target.clone()

            output = modelCF(data)
            if args.mode == 'SC': output = output.reshape(output.shape[0], output.shape[1])
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total_num += target.size(0)
            print(f'correct: {correct}, total_num: {total_num}, accuracy: {100. * correct / total_num:.00f}%')
                
            if i==0 and args.mode == 'WAGE':
                hook.remove_hook_list(hook_handle_list)

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total_num
        
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_num} ({test_accuracy:.00f}%)')

    # # reshape input files: (bs*dim, seq_len) -> (dim, bs*seq_len)
    # if args.model == 'Transformer':
    #     csvfiles = os.listdir('./layer_record_'+str(args.model))
    #     inputfiles = [file for file in csvfiles if file.startswith('input')]
    #     weightfiles = [file for file in csvfiles if file.startswith('weight')]
    #     inputfiles.sort()
    #     weightfiles.sort()
    #     assert len(inputfiles) == len(weightfiles), "number of input files {} not equal to number of weight files {}".format(len(inputfiles), len(weightfiles))
    #     print("reshaping input files ... ")
        
    #     for f_i in range(len(inputfiles)):
    #         # read input file
    #         input_matrix = np.genfromtxt('./layer_record_'+str(args.model)+'/'+"inputFC_"+str(f_i)+".csv", delimiter=',') # inputFC_0.csv
    #         weight_matrix = np.genfromtxt('./layer_record_'+str(args.model)+'/'+"weightFC_"+str(f_i)+".csv", delimiter=',')
            
    #         if input_matrix.shape[0] % weight_matrix.shape[0] != 0:
    #             print(inputfiles[f_i], weightfiles[f_i])
    #             raise ValueError("input matrix shape {} not divisible by weight matrix shape {}".format(input_matrix.shape, weight_matrix.shape))
            
    #         split_num = int(input_matrix.shape[0]/weight_matrix.shape[0]) # 1 pair 1
    #         split_matrices = np.split(input_matrix, split_num, axis=0)
    #         concatenated_matrix = np.concatenate(split_matrices, axis=1)
    #         # replace input file with concatenated matrix
    #         np.savetxt('./layer_record_'+str(args.model)+'/'+"inputFC_"+str(f_i)+".csv", concatenated_matrix, delimiter=",",fmt='%s')


    
