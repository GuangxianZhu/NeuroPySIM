# Crossbar-Based Neural Network Inference Simulation
This project provides tools to simulate and evaluate neural network (NN) inference using crossbar arrays. The simulations focus on key performance metrics such as NN accuracy, chip-level power consumption, hardware inference time, and hardware area.

## Quick Start Guide
### Step 1: Neural Network Accuracy Simulation
1. Define your model architecture and dataset in `./models`

2. Configure simulation parameters in `inferenceConfig.py`

3. Run the simulation script:

    `python inference_sim.py`

### Step 2: Chip-Level Simulation

#### build:

    cd from_neurosim
    mkdir build
    cd build
    cmake ..
    make

This creates the FormulaBindings.so shared object file, which allows Python scripts to access NeuroSIM parameters and formulas.

#### Run chip sim

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

### Read:
you need to read.
1. `./from_neurosim/param.cpp`: param defined by NeuroSIM.
2. `./from_neurosim/bindings.cpp`: bind the NeuroSIM's formula.cpp to python.

### Customization

1. `Change the parser in `Chip.py`
2. Add your customize cell in `./MemCell.py`

## References

NeuroSIM v1.4

## Requirements

    Python > 3.10
    Pybind11
    torch
    numpy