import math
from constant import *
from Gate_calculator import horowitz, CalculateTransconductance
from from_neurosim.build import FormulaBindings

from DFF import DFF
from Adder import Adder

class ShiftAdd:
    def __init__(self, numUnit, numAdderBit, numReadPulse,
                 param, tech, gate_params):
        self.numUnit = numUnit
        self.numAdderBit = numAdderBit
        self.numReadPulse = numReadPulse
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.numDff = (numAdderBit+1 + numReadPulse-1) * numUnit
        self.dff = DFF(self.numDff, param, tech, gate_params)
        self.adder = Adder(numAdderBit, numUnit, param, tech, gate_params)
        
    def CalculateArea(self):
        self.area = 0
        self.height = 0
        self.width = 0
        
        self.adder.CalculateArea()
        self.dff.CalculateArea()
        
        self.height = self.adder.height + self.dff.height
        self.width = self.adder.width + self.gate_params["wInv"] + self.gate_params["wNand"]
        
        self.area = self.height * self.width
        
    def CalculateLatency(self, numRead):
        """
        // We can shift and add the weighted sum data in the next vector pulse integration cycle
        // Thus the shift-and-add time can be partially hidden by the vector pulse integration time at the next cycle
        // But there is at least one time of shift-and-add, which is at the last vector pulse cycle
        """
        
        self.adder.CalculateLatency(self.gate_params["capTgDrain"], 1)
        self.dff.CalculateLatency(1)
        shiftAddLatency = self.adder.readLatency + self.dff.readLatency
        
        self.readLatency = (shiftAddLatency - self.param.readPulseWidth) * (numRead - 1)
        
        
    def CalculatePower(self, numRead):
        
        self.adder.CalculatePower(numRead, self.numUnit)
        self.dff.CalculatePower(numRead, self.numDff)
        
        self.readDynamicEnergy = self.adder.readDynamicEnergy + self.dff.readDynamicEnergy
        
        self.leakage = self.adder.leakage + self.dff.leakage
        
        
        
        
        