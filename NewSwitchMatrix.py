import math
from constant import *
from from_neurosim.build import FormulaBindings
from MemCell import MemCell
from Gate_calculator import CalculateGateArea, CalculateOnResistance, CalculateGateCapacitance, CalculateTransconductance, horowitz, CalculateGateLeakage
from DFF import DFF

class NewSwitchMatrix:
    def __init__(self, numOutput, activityRowRead,
                 param, tech, gate_params):
        self.numOutput = numOutput
        self.activityRowRead = activityRowRead
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.dff = DFF(numOutput, param, tech, gate_params)
        
    def CalculateArea(self):
        tech = self.tech
        GP = self.gate_params
        hTg, wTg = GP["hTg"], GP["wTg"]
        self.height = hTg * self.numOutput
        self.dff.CalculateArea()
        self.width = (wTg*4) + self.dff.width
        self.area = self.height * self.width
        
    def CalculateLatency(self, capLoad, resLoad, numRead, numWrite):
        # DFF
        self.dff.CalculateLatency(numRead)
        
        # TG
        rampInput = 1e20
        GP = self.gate_params
        capOutput = GP["capTgDrain"] * 5
        tr = GP["resTg"] * (capOutput + capLoad) + resLoad * capLoad/2 # elmore delay model
        self.readLatency = horowitz(tr, 0, rampInput)['result']
        self.readLatency *= numRead
        self.readLatency += self.dff.readLatency
        
        self.writeLatency = horowitz(tr, 0, rampInput)['result']
        self.writeLatency *= numWrite
            
    def CalculatePower(self, numRead, numWrite, activityRowRead):
        # DFF
        self.dff.CalculatePower(numRead, self.numOutput)
        
        self.leakage = self.dff.leakage
        
        # read dynamic energy
        GP = self.gate_params
        PM = self.param
        TC = self.tech
        self.readDynamicEnergy = GP["capTgDrain"]*2*(PM.accessVoltage**2)*self.numOutput*activityRowRead
        self.readDynamicEnergy += GP["capTgDrain"]*5*(PM.readVoltage**2)*self.numOutput*activityRowRead
        self.readDynamicEnergy += (GP["capTgGateN"] + GP["capTgGateP"]) * 3 * (TC.vdd**2) * self.numOutput * activityRowRead
        self.readDynamicEnergy *= numRead
        self.readDynamicEnergy += self.dff.readDynamicEnergy
        
        # write dynamic energy(2-step write and average case half SET and half RESET)
        """
        // Write dynamic energy (2-step write and average case half SET and half RESET)
		// 1T1R
		// connect to rows, when writing, pass GND to BL, no transmission energy acrossing BL
		writeDynamicEnergy += (capTgDrain * 2) * cell.accessVoltage * cell.accessVoltage;    // 1 TG pass Vaccess to CMOS gate to select the row
		writeDynamicEnergy += (capTgGateN + capTgGateP) * 2 * tech.vdd * tech.vdd * 2;    // open 2 TG when Q selected, and *2 means switching from one selected row to another
        writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd;    // always open one TG when writing	
		writeDynamicEnergy *= numWrite;
		if (numWrite != 0)
			writeDynamicEnergy += dff.readDynamicEnergy;
        """
        self.writeDynamicEnergy = GP["capTgDrain"]*2*(PM.accessVoltage**2)
        self.writeDynamicEnergy += (GP["capTgGateN"] + GP["capTgGateP"]) * 2 * (TC.vdd**2) * 2
        self.writeDynamicEnergy += (GP["capTgGateN"] + GP["capTgGateP"]) * (TC.vdd**2)
        self.writeDynamicEnergy *= numWrite
        if numWrite != 0:
            self.writeDynamicEnergy += self.dff.readDynamicEnergy
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        