import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer,wage_quantizer
import numpy as np
import math
from torch.nn import Unfold

def quantize_and_reconstruct(a_in, nb, NormScale, fault_rate=0.0):
    BS, C, H, W, L = 0, 0, 0, 0, 0
    assert a_in.dim() in [4, 3, 2], "Input tensor must be 2D, 3D or 4D"
    if a_in.dim() == 4:
        BS, C, H, W = a_in.shape
    elif a_in.dim() == 3:
        BS, C, L = a_in.shape
    elif a_in.dim() == 2:
        BS, L = a_in.shape
    a = a_in.flatten()
    
    # Step 1: Scale and round to nearest integer
    a_scaled = ((a / NormScale) * (2 ** (nb - 1)))
    a_dec = torch.round(a_scaled).long()  # Scale and round
    # print(f"Scaled decimal tensor: {a_dec}")
    
    # Step 2: Handle 2's complement conversion for negative values
    a_dec = torch.where(a_dec < 0, (1 << nb) + a_dec, a_dec)  # Convert to unsigned 2's complement
    # print(f"Two's complement tensor: {a_dec}")

    # Step 3: Extract binary representation
    # Convert to binary representation (as a tensor of bits)
    a_bin = torch.stack([(a_dec >> i) & 1 for i in range(nb)], dim=-1).flip(dims=[-1])
    # print(f"Binary representation tensor:\n{a_bin}") # shp: (BS, nb)
    # add fault rate, reverse value, developing

    # Step 4: Reconstruct decimal value
    # Convert binary back to decimal
    is_negative = a_bin[:, 0] == 1  # Check if MSB is 1
    positive_part = torch.sum(a_bin[:, 1:] * (2 ** torch.arange(nb - 1 - 1, -1, -1)), dim=1)

    a_dec_recon = torch.where(is_negative, -((1 << (nb - 1)) - positive_part), positive_part)
    # print(f"Reconstructed decimal tensor: {a_dec_recon}")

    # Step 5: Reconstruct float value
    a_recon = a_dec_recon / (2 ** (nb - 1)) * NormScale
    if a_in.dim() == 4:
        a_recon = a_recon.view(BS, C, H, W)
    elif a_in.dim() == 3:
        a_recon = a_recon.view(BS, C, L)
    elif a_in.dim() == 2:
        a_recon = a_recon.view(BS, L)
    
    return a_recon

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                RRAM='NoRRAM',
                wl_input=8, wl_activate=8, wl_error=8, wl_weight=8,
                subArray = 128, 
                ADCprecision=5, onoffratio=10, vari=0, t=0, v=0, detect=0, target=0,
                name = 'Qconv', model = None):
        
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.name = name
        self.model = model

        # data format and precision
        self.wl_weight = wl_weight  # weight precision bits
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.Qweight = None         # quantized weight
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0) # limit in [-1,1]

        # architecture config
        self.subArray = subArray    # num of rows in subArray
        self.ADCprecision = ADCprecision

        # RRAM parameters
        self.RRAM = RRAM
        self.cellBit = 1
        self.onoffratio = onoffratio
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target

    def forward(self, input):

        if self.RRAM == 'RRAM_analog': # non-ideal effect on the weight directly (analog weight: conductance)
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target) # Add retention effect
            weight = weight + weight*torch.normal(0., torch.full(weight.size(),self.vari, device='cpu')) # Add weight variation
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
            output = output/self.scale
            output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)
        
        elif self.RRAM == 'RRAM_digital': # non-ideal effect on single bit weight (digital weight: 1 or 0)

            input_recon = quantize_and_reconstruct(input, self.wl_input, self.scale, fault_rate=0.0)
            weight_recon = quantize_and_reconstruct(self.weight, self.wl_weight, self.scale, fault_rate=0.0)
            output = F.conv2d(input_recon, weight_recon, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = quantize_and_reconstruct(output, self.wl_error, self.scale, fault_rate=0.0)

        else:
            # original WAGE QConv2d
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            self.Qweight = weight
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = output/self.scale
            output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)

        return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False,
	            RRAM='NoRRAM',
                wl_input=8, wl_activate=8, wl_error=8, wl_weight=8,
                subArray = 128, 
                ADCprecision=5, onoffratio=10, vari=0, t=0, v=0, detect=0, target=0,
                name = 'Qconv', model = None):
        
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.name = name
        self.model = model

        # data format and precision
        self.wl_weight = wl_weight  # weight precision bits
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.Qweight = None         # quantized weight
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0) # limit in [-1,1]

        # architecture config
        self.subArray = subArray    # num of rows in subArray
        self.ADCprecision = ADCprecision

        # RRAM parameters
        self.RRAM = RRAM
        self.cellBit = 1
        self.onoffratio = onoffratio
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target

    def forward(self, input):

        if self.RRAM == 'RRAM_digital': # non-ideal effect on single bit weight (digital weight: 1 or 0)

            input_recon = quantize_and_reconstruct(input, self.wl_input, self.scale, fault_rate=0.0)
            weight_recon = quantize_and_reconstruct(self.weight, self.wl_weight, self.scale, fault_rate=0.0)
            output = F.linear(input_recon, weight_recon, self.bias)
            output = quantize_and_reconstruct(output, self.wl_error, self.scale, fault_rate=0.0)
        
        elif self.RRAM == 'RRAM_analog': # non-ideal effect on the weight directly (analog weight: conductance)
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            weight = weight + weight*torch.normal(0., torch.full(weight.size(),self.vari, device='cpu')) # Add weight variation
            input = wage_quantizer.Q(input,self.wl_input)
            output= F.linear(input, weight, self.bias)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
            output = output/self.scale
            output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)
            
        else:
            # original WAGE QLinear
            weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
            weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            output = F.linear(input, weight, self.bias)
            output = output/self.scale
            output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)
        
        return output

