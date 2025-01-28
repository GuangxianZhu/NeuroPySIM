import math

from constant import *
from from_neurosim.build import FormulaBindings

class Mux:
    def __init__(self, numInput, numSelection, param, tech, gate_params):
        self.numInput = numInput
        self.numSelection = numSelection
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.widthTgN = self.gate_params["widthTgN"]
        self.widthTgP = self.gate_params["widthTgP"]
        self.resTg = self.gate_params["resTg"]
        self.wTg = self.gate_params["wTg"]
        self.hTg = self.gate_params["hTg"]
        self.resTg = self.gate_params["resTg"]
        self.capTgDrain = self.gate_params["capTgDrain"]
        self.capTgGateN = self.gate_params["capTgGateN"]
        self.capTgGateP = self.gate_params["capTgGateP"]
        
        
        self.initialized = True
        
    def CalculateArea(self):
        numTg = self.numInput * self.numSelection
        area, height, width = 0, 0, 0
        # digital mux
        self.width = self.wTg * numTg
        self.height = self.hTg
        self.area = self.height * self.width
        
    def CalculateLatency(self, _capLoad, _numRead):
        capLoad = 1e20
        tr = self.resTg * (self.capTgDrain + 0.5*self.capTgGateN + 0.5*self.capTgGateP + capLoad)
        self.readLatency = 2.3 * tr;	# 2.3 means charging from 0% to 90%
        self.readLatency *= _numRead
        
    def CalculatePower(self, _numRead):
        readDynamicEnergy = self.capTgGateN * self.numInput * self.tech.vdd * self.tech.vdd
        readDynamicEnergy += (self.capTgDrain * 2) * self.numInput * (self.param.readVoltage**2);	# Selected pass gates (OFF to ON)
        readDynamicEnergy *= _numRead
        self.readDynamicEnergy = readDynamicEnergy

if __name__ == "__main__":
    
    from gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(22, FormulaBindings.DeviceRoadmap.LSTP, FormulaBindings.TransistorType.conventional)
    param = FormulaBindings.Param()
    param.memcelltype = RRAM
    gate_params = compute_gate_params(param, tech)
    
    mux = Mux(4, 2**4, param, tech, gate_params)
    mux.CalculateArea()
    mux.CalculateLatency(45*0.2e-15/1e-6, 128)
    mux.CalculatePower(128)
    
    print("Mux Area: ", mux.area)
    print("Mux Read Latency: ", mux.readLatency)
    print("Mux Read Dynamic Energy: ", mux.readDynamicEnergy)
    