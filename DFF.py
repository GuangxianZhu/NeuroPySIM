import math
from constant import * # include all constants, CMOS tech specifications, cal funcs, etc.
from from_neurosim.build import FormulaBindings

class DFF:
    def __init__(self, numDff, param, tech, gate_params):
        self.numDff = numDff
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
    
    def CalculateArea(self):
        hDff = self.gate_params["hInv"]
        wDff = self.gate_params["wInv"] * 12
        self.width = wDff * self.numDff
        self.height = hDff
        self.area = self.width * self.height
        
    def CalculateLatency(self, _numRead):
        readLatency = 1/self.param.clkFreq/2
        readLatency *= _numRead
        self.readLatency = readLatency
    
    def CalculatePower(self, _numRead, numDffPerOperation):
        # leakage
        self.leakage = self.gate_params["leakageInv"] * self.numDff
        
        # dynamic
        # Assume input D=1 and the energy of CLK INV and CLK TG are for 1 clock cycles
        # CLK INV (all DFFs have energy consumption)
        # readDynamicEnergy = (capInvInput + capInvOutput) * tech.vdd * tech.vdd * 4 * self.numDff
        readDynamicEnergy = (self.gate_params["capInvInput"] + self.gate_params["capInvOutput"]) * (self.tech.vdd**2) * 4 * self.numDff
        # CLK TG (all DFFs have energy consumption)
        readDynamicEnergy += self.gate_params["capTgGateN"] * (self.tech.vdd**2) * 2 * self.numDff
        readDynamicEnergy += self.gate_params["capTgGateP"] * (self.tech.vdd**2) * 2 * self.numDff
        # D to Q path (only selected DFFs have energy consumption)
        readDynamicEnergy += (self.gate_params["capTgDrain"] * 3 + self.gate_params["capInvInput"]) * (self.tech.vdd**2) * min(numDffPerOperation, self.numDff)
        readDynamicEnergy += (self.gate_params["capTgDrain"] + self.gate_params["capInvOutput"]) * (self.tech.vdd**2) * min(numDffPerOperation, self.numDff)
        readDynamicEnergy += (self.gate_params["capInvInput"] + self.gate_params["capInvOutput"]) * (self.tech.vdd**2) * min(numDffPerOperation, self.numDff)
        
        readDynamicEnergy *= _numRead
        writeDynamicEnergy = readDynamicEnergy
        
        self.readDynamicEnergy = readDynamicEnergy
        self.writeDynamicEnergy = writeDynamicEnergy
        
if __name__ == "__main__":
    
    from gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(22, FormulaBindings.DeviceRoadmap.LSTP, FormulaBindings.TransistorType.conventional)
    param = FormulaBindings.Param()
    param.memcelltype = RRAM
    gate_params = compute_gate_params(param, tech)
    
    dff = DFF(64, param, tech, gate_params)
    dff.CalculateArea()
    dff.CalculateLatency(64)
    dff.CalculatePower(64, 64)
    
    print(dff.area, dff.readLatency, dff.readDynamicEnergy, dff.writeDynamicEnergy)
        
        
        
        
        
        