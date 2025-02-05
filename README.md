# Crossbar-Based Neural Network Inference Simulation
This project provides tools to simulate and evaluate neural network (NN) inference using crossbar arrays. The simulations focus on key performance metrics such as NN accuracy, chip-level power consumption, hardware inference time, and hardware area.

## Quick Start Guide
## Step 1: Neural Network Accuracy Simulation
Define your model architecture and dataset in `./models`

Configure simulation parameters in `inferenceConfig.py`, note that this config will only effect inference!

Run the simulation script:

    python inference_sim.py

It will generate NN traces (each layer's output), saved in `"./layer_record_[NN]/"`, these traces will be used in NeuroPySim core program, to calculate power, latency and area. 

## Step 2: Chip-Level Simulation

### build:

    cd from_neurosim
    mkdir build
    cd build
    cmake ..
    make

This creates the FormulaBindings.so shared object file, which allows Python scripts to access NeuroSIM parameters and formulas.

### Run chip sim

    python Chip.py

This script calculates:
1. Chip Area
2. Inference Time
3. Energy Consumption
4. Performance Breakdown for crossbar arrays

### NOTE: 

#### The project defaults to RRAM, 1T1R, and conventional sequential crossbar configurations.

#### You can extend support for other memory types by modifying related files (see "Customization").

## Customization Guide

### Read before using your config:
1. `./from_neurosim/param.cpp`: param defined by NeuroSIM.
2. `./from_neurosim/bindings.cpp`: bind the NeuroSIM's formula.cpp to python.

### Crossbar
#### Energy
1. Check Memory cell read energy in `Array.py`.
2. Check Memory cell write energy in `Tile.py`.
#### Latency
1. Check Memory cell read latency in `Array.py`.
2. Check Memory cell write latency in `Tile.py`.
#### Area
1. Check `Array.py`.

### Peripheral Circuits
1. Check each class.


## References

NeuroSIM v1.4 (inference)

NeuroSIM v2.0 (train)

## Requirements

    Python > 3.10
    Pybind11
    torch
    numpy

# Appendix

## Data relations on simulation
1. Acc.: Related.
2. Read Power: Related, on-off ratio of input vectors only.
3. Write Energy: Related, Weight matrix before and after update.
4. Latency: Related, numbit of input only.
5. Area: Not related.
