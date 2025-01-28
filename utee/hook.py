import os
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import numpy as np
import torch
from utee import wage_quantizer
from utee import float_quantizer

def Q_and_Record(self, input, output): 
    global model_n, FP

    print("quantize layer ", self.name)
    input_file_name =  './layer_record_' + str(model_n) + '/input' + str(self.name) + '.csv'
    weight_file_name =  './layer_record_' + str(model_n) + '/weight' + str(self.name) + '.csv'
    # f = open('./layer_record_' + str(model_n) + '/trace_command.sh', "a")
    # f.write(weight_file_name+' '+input_file_name+' ')
    if FP: # FP
        weight_q = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
    else: # WAGE
        weight_q = wage_quantizer.Q(self.weight,self.wl_weight)
    write_matrix_weight( weight_q.cpu().data.numpy(),weight_file_name)
    if len(self.weight.shape) > 2: # conv
        k=self.weight.shape[-1]
        padding = self.padding
        stride = self.stride  
        write_matrix_activation_conv(stretch_input_with_unfold(input[0].cpu().data, k, padding, stride),None, self.wl_input, input_file_name)
    else: # fc
        if len(input[0].shape) == 2: # (bs, dim)
            write_matrix_activation_fc(input[0].cpu().data.numpy(),None ,self.wl_input, input_file_name)
        elif len(input[0].shape) == 3: # (bs, seq_len, dim)
            bs, seq_len, dim = input[0].shape
            input_for_write = input[0].view(bs, seq_len*dim)
            write_matrix_activation_fc(input_for_write.cpu().data.numpy(),None ,self.wl_input, input_file_name)
        else:
            raise NotImplementedError



def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')


def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length) # only save the first sample in a batch
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):

    filled_matrix_b = np.zeros([input_matrix.shape[1],length],dtype=str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length) # only save the first sample in a batch
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def stretch_input_with_unfold(input_matrix:torch.Tensor, window_size=5, padding=(0,0), stride=(1,1)) -> np.ndarray:

    unfold = nn.Unfold(kernel_size=(window_size, window_size), padding=padding, stride=stride)

    # The shape of output_tensor will be (batch_size, C * kernel_size * kernel_size, L)
    output_tensor = unfold(input_matrix)

    # Reshape: The shape will be (batch_size, L, C * kernel_size * kernel_size)
    output_tensor = output_tensor.transpose(1, 2)
    output_tensor_np = output_tensor.numpy()

    return output_tensor_np




def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list

def bin2dec(x,n):
    bit = x.pop(0)
    base = 2**(n-1)
    delta = 1.0/(2**(n-1))
    y = -bit*base
    base = base/2
    for bit in x:
        y = y+base*bit
        base= base/2
    out = y*delta
    return out

def remove_hook_list(hook_handle_list):
    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(model,model_name,mode): # wl_weight,wl_activation,subArray,parallelRead,
    global model_n, FP
    model_n = model_name
    FP = 1 if mode=='FP' else 0
    
    hook_handle_list = []
    if not os.path.exists('./layer_record_'+str(model_name)):
        os.makedirs('./layer_record_'+str(model_name))
    if os.path.exists('./layer_record_'+str(model_name)+'/trace_command.sh'):
        os.remove('./layer_record_'+str(model_name)+'/trace_command.sh')
    # f = open('./layer_record_'+str(model_name)+'/trace_command.sh', "w")
    # f.write('./NeuroSIM/main ./NeuroSIM/NetWork_'+str(model_name)+'.csv '+str(wl_weight)+' '+str(wl_activation)+' '+str(subArray)+' '+str(parallelRead)+' ')
    
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (FConv2d, QConv2d, nn.Conv2d)) or isinstance(layer, (FLinear, QLinear, nn.Linear)):
            hook_handle_list.append(layer.register_forward_hook(Q_and_Record))
            
    return hook_handle_list
