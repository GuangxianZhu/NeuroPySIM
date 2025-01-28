import csv
import os
import time
import argparse

from Chip import Chip

# Function to load configurations from a CSV file
def load_configurations(filename):
    config = {}
    with open(filename, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                key, value = row
                config[key] = value
    return config


if __name__ == "__main__":
    if os.path.exists("records") == False:
        os.makedirs("records")

    parser = argparse.ArgumentParser(description='SuperSIM config.')

    # General parameters
    parser.add_argument('--model_name', type=str, default='None', help='Model name: Transformer, VGG8, LeNet5, DenseNet40')

    # Architecture parameters
    parser.add_argument('--ArrNumY', type=int, default=1, help='Must be (2^n) and (<=16), for easy AdderTree design')
    parser.add_argument('--ArrNumX', type=int, default=1, help='Number of arrays in X direction')
    parser.add_argument('--ArrRowSize', type=int, default=4, help='Size of array in Y direction')
    parser.add_argument('--ArrColSize', type=int, default=4, help='Size of array in X direction')
    parser.add_argument('--nBits', type=int, default=1, help='Value quantization bits')
    parser.add_argument('--MappingMode', type=int, default=0, choices=[0, 1], help='Mapping mode: 0 for auto, 1 for manual')

    # HW parameters
    parser.add_argument('--technode', type=int, default=45, choices=[45, 28], help='Technology node in nm')
    parser.add_argument('--freq', type=str, default='5.0', help='Frequency in GHz')
    parser.add_argument('--MemCellType', type=str, default='RRAM')

    args = parser.parse_args()

    configurations = load_configurations('SIMconfigs.csv')
    
    start_time = time.time()

    # Override default or command-line values with values from the CSV
    for key, value in configurations.items():
        if hasattr(args, key):
            setattr(args, key, type(getattr(args, key))(value))
            
    print("Arguments:")
    for arg in vars(args):
        print(f'    {arg}: {getattr(args, arg)}')
        
    input_list = ["layer_record_VGG8/inputConv0_.csv",
                    "layer_record_VGG8/inputConv1_.csv",
                    "layer_record_VGG8/inputConv3_.csv",
                    "layer_record_VGG8/inputConv4_.csv",
                    "layer_record_VGG8/inputConv6_.csv",
                    "layer_record_VGG8/inputConv7_.csv",
                    "layer_record_VGG8/inputFC0_.csv",
                    "layer_record_VGG8/inputFC1_.csv"]
    weight_list = ["layer_record_VGG8/weightConv0_.csv",
                    "layer_record_VGG8/weightConv1_.csv",
                    "layer_record_VGG8/weightConv3_.csv",
                    "layer_record_VGG8/weightConv4_.csv",
                    "layer_record_VGG8/weightConv6_.csv",
                    "layer_record_VGG8/weightConv7_.csv",
                    "layer_record_VGG8/weightFC0_.csv",
                    "layer_record_VGG8/weightFC1_.csv"]
    
    print("\nRunning Python script for Power, area, latency, JJ simulation...\n")
    
    log_text = ""
    
    chip = Chip(args)
    log_text += chip.chip_floorplan(args, input_list, weight_list)
    print(chip.speedUp)
    print(chip.utilization)
    
    chip_power = chip.cal_power()
    
    