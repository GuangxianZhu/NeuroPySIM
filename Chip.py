import argparse
import numpy as np
import csv
import os
import math
import time

from Tile import Tile
from constant import *


def checkArraySize(args):
    # Make sure the array size is 2^n
    if args.ArrRowSize != 2 ** int(math.log2(args.ArrRowSize)):
        raise ValueError(f"ArrRowSize not 2^n!: {args.ArrRowSize}")

def checkArrayYnum(args):
    # Make sure the array size is 2^n
    if args.ArrNumY != 2 ** int(math.log2(args.ArrNumY)):
        raise ValueError(f"ArrNumY not 2^n, not for Addertree design!: {args.ArrNumY}")
    if args.ArrNumY > 16:
        raise ValueError(f"ArrNumY's Addertree design not implemented!: {args.ArrNumY}")

def calc_num_stage(matrix, num_subarray_row, num_subarray_col, subarray_row_size, subarray_col_size):
    stageY = math.ceil(matrix.shape[0] / (num_subarray_row * subarray_row_size))
    stageX = math.ceil(matrix.shape[1] / (num_subarray_col * subarray_col_size))
    speedUpY, speedUpX = 1, 1
    if matrix.shape[0] < subarray_row_size:
        speedUpY = num_subarray_row
    if matrix.shape[1] < subarray_col_size:
        speedUpX = num_subarray_col

    return stageY, stageX, speedUpY, speedUpX

def get_csv_size(input_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        num_rows = len(rows)
        num_columns = len(rows[0]) if rows else 0
        return num_rows, num_columns


def create_tiles(input_list, weight_list, args, log_text, param, tech, gate_params):
    assert len(input_list) == len(weight_list)
    checkArrayYnum(args)
    # print("creating tiles...")

    tiles = []
    speedUp = [] # speedup for each tile(layer)
    OP = 0.0
    overall_utilization = 0.0
    total_input_length = 0

    for i, (input_file, weight_file) in enumerate(zip(input_list, weight_list)):
        # in_rows, in_cols = get_csv_size(input_file)
        # input_matrix = np.zeros((in_rows, in_cols), dtype=int)

        input_matrix = load_input(input_file, args)

        total_input_length += input_matrix.shape[1]

        # accurate mode
        weight_matrix = load_weight(weight_file, args)
        # make rand weight matrix old, same as weight matrix
        # weight_matrix_old = load_weight(weight_file_old, args)
        weight_matrix_old = np.random.randint(0, 2, weight_matrix.shape)
        
        if weight_matrix.shape[0] != input_matrix.shape[0]:
            raise ValueError(f"{weight_file}: w_mat:{weight_matrix.shape[0]} != i_mat:{input_matrix.shape[0]}")

        OP += input_matrix.shape[0] * input_matrix.shape[1] * weight_matrix.shape[1]

        if args.MappingMode == 0:
            stageY, stageX, speedUpY, speedUpX = calc_num_stage(weight_matrix, args.ArrNumY, args.ArrNumX, args.ArrRowSize, args.ArrColSize)
            speedUp.append(speedUpY * speedUpX)
            tile = Tile(stageY, stageX, args.ArrNumY, args.ArrNumX, args.ArrRowSize, args.ArrColSize,
                        param, tech, gate_params)

            tile.copy_weight(weight_matrix)
            tile.copy_weight_old(weight_matrix_old)
            tile.copy_input(input_matrix)

            tile.utilization *= speedUpY * speedUpX

            log_text+=f"Tile{i} util%: {tile.utilization}\n"
            overall_utilization += tile.utilization

            tiles.append(tile)
        else:
            raise NotImplementedError("MappingMode: {} not implemented".format(args.MappingMode))

    TOP = OP / 1e12
    args.total_input_length = total_input_length

    log_text += f"Total OP: {OP}\n"
    log_text += f"Total TOP: {TOP}\n"
    log_text += f"Overall util%: {overall_utilization / len(input_list)}\n\n"

    return tiles, speedUp, log_text

def load_weight(weightfile, args):
    weight = []

    with open(weightfile, 'r') as fileone:
        reader = csv.reader(fileone)

        for line in reader:
            weightrow = []

            for val in line:
                f = float(val) # -0.14844

                NormalizedMin = 0
                NormalizedMax = 2 ** args.nBits
                RealMax = 1
                RealMin = -1

                newdata = ((NormalizedMax - NormalizedMin) // (RealMax - RealMin) * (f - RealMax)) + NormalizedMax # 108.996
                newdata = floor(newdata+0.5) if newdata >= 0 else ceil(newdata-0.5) # 109

                cellrange = 2
                synapsevector = [0] * args.nBits
                value = int(newdata)

                for z in range(0, args.nBits):
                    remainder = value % cellrange
                    value = value // cellrange
                    synapsevector[args.nBits - 1 - z] = remainder

                for u in range(args.nBits):
                    cellvalue = synapsevector[u]
                    weightrow.append(cellvalue)  # debug, test cell value

            weight.append(weightrow)

    return np.array(weight, dtype=np.int8)

def load_input(inputfile, args):
    input_matrix = []

    with open(inputfile, 'r') as input_file:
        reader = csv.reader(input_file)

        for line in reader:
            row = [int(float(val)) for val in line]
            input_matrix.append(row)

    return np.array(input_matrix, dtype=np.int8)

def ceil(x):
    return math.ceil(x)

def floor(x):
    return math.floor(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SuperSIM config.')
    # General parameters
    parser.add_argument('--model_name', type=str, default='None', help='Model name: Transformer, VGG8, LeNet5, DenseNet40')

    # Architecture parameters
    parser.add_argument('--ArrNumY', type=int, default=1, help='Must be (2^n) and (<=16), for easy AdderTree design')
    parser.add_argument('--ArrNumX', type=int, default=1, help='Number of arrays in X direction')
    parser.add_argument('--ArrRowSize', type=int, default=128, help='Size of array in Y direction')
    parser.add_argument('--ArrColSize', type=int, default=128, help='Size of array in X direction')
    parser.add_argument('--nBits', type=int, default=8, help='Value quantization bits')
    parser.add_argument('--cellBit', type=int, default=1, help='one cell can represent 2^cellBit values')
    parser.add_argument('--MappingMode', type=int, default=0, choices=[0, 1], help='Mapping mode: 0 for auto, 1 for manual')
    parser.add_argument('--levelOutput', type=int, default=32, help='Output level for amplifier')

    # tech parameters
    parser.add_argument('--technode', type=int, default=22, help='Technology node')
    parser.add_argument('--temperature', type=int, default=300, help='Temperature in K')

    args = parser.parse_args()
    
    from gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(45, FormulaBindings.DeviceRoadmap.LSTP, 
                    FormulaBindings.TransistorType.conventional)
    param = FormulaBindings.Param()
    
    param.memcelltype = RRAM
    param.numRowSubArray = args.ArrRowSize
    param.numColSubArray = args.ArrColSize
    param.synapseBit = args.nBits   # 8: INT8
    param.cellBit = args.cellBit    # 1
    param.technode = args.technode
    param.temp = args.temperature
    param.levelOutput = args.levelOutput
    
    gate_params = compute_gate_params(param, tech)
    
    start_time = time.time()
    print("Arguments:")
    for arg in vars(args):
        print(f'    {arg}: {getattr(args, arg)}')
    print("\n")
        
    input_list = ["layer_record_LeNet5/inputConv0_.csv",
                    "layer_record_LeNet5/inputConv1_.csv",
                    "layer_record_LeNet5/inputFC2_.csv",
                    "layer_record_LeNet5/inputFC3_.csv",
                    "layer_record_LeNet5/inputFC4_.csv"]
    weight_list = ["layer_record_LeNet5/weightConv0_.csv",
                    "layer_record_LeNet5/weightConv1_.csv",
                    "layer_record_LeNet5/weightFC2_.csv",
                    "layer_record_LeNet5/weightFC3_.csv",
                    "layer_record_LeNet5/weightFC4_.csv"]
        
    # input_list = ["layer_record_VGG8/inputConv0_.csv",
    #                 "layer_record_VGG8/inputConv1_.csv",
    #                 "layer_record_VGG8/inputConv3_.csv",
    #                 "layer_record_VGG8/inputConv4_.csv",
    #                 "layer_record_VGG8/inputConv6_.csv",
    #                 "layer_record_VGG8/inputConv7_.csv",
    #                 "layer_record_VGG8/inputFC0_.csv",
    #                 "layer_record_VGG8/inputFC1_.csv"]
    # weight_list = ["layer_record_VGG8/weightConv0_.csv",
    #                 "layer_record_VGG8/weightConv1_.csv",
    #                 "layer_record_VGG8/weightConv3_.csv",
    #                 "layer_record_VGG8/weightConv4_.csv",
    #                 "layer_record_VGG8/weightConv6_.csv",
    #                 "layer_record_VGG8/weightConv7_.csv",
    #                 "layer_record_VGG8/weightFC0_.csv",
    #                 "layer_record_VGG8/weightFC1_.csv"]
    
    log_text = ""
    tiles, speedUp, log_text = create_tiles(input_list, weight_list, args,log_text, 
                                            param, tech, gate_params)
    print("Util for tiles")
    for tile in tiles:
        print('\t',tile.utilization)
    print("\n")
    
    nTilesH = math.ceil(math.sqrt(len(tiles)))
    nTilesW = nTilesH
    chipReadLatency, chipWriteLatency = 0, 0
    chipReadDynamicEnergy = 0
    chipWriteDynamicEnergy = 0
    
    for i in range(len(tiles)):
        tile = tiles[i]
        tile.CalculateArea()
        chipH = tile.height * nTilesH
        chipW = tile.width * nTilesW
        
        tile.CalculateLatency(speedUp[i])
        chipReadLatency += tile.readLatency
        chipWriteLatency += tile.writeLatency
        
        tile.CalculatePower()
        chipReadDynamicEnergy += tile.readDynamicEnergy
        chipWriteDynamicEnergy += tile.writeDynamicEnergy
        
    print(f"Chip H: {chipH*1e6:.3f}um, W: {chipW*1e6:.3f}um")
    print("-"*50)
    
    print(f"Chip ReadLatency (inference 1 sample): {chipReadLatency*1e9:.3f}ns")
    print(f"\tper bit: {chipReadLatency/args.total_input_length*1e9:.3f}ns")
    print(f"Chip WriteLatency (write weight into crossbar): {chipWriteLatency*1e9:.3f}ns")
    print("-"*50)
    
    print(f"Chip read dynamic energy (inference 1 sample): {chipReadDynamicEnergy*1e9:.3f}nJ")
    print(f"Chip write dynamic energy (write weight into crossbar): {chipWriteDynamicEnergy*1e9:.3f}nJ")
    print("-"*50)
    
    print(f"Time elapsed: {time.time()-start_time:.3f}s")
    print("\n")
    
    tile.array.printInfo()