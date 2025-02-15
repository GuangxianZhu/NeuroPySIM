import math
from constant import * # include all constants, CMOS tech specifications, cal funcs, etc.
from from_neurosim.build import FormulaBindings
from Gate_calculator import horowitz
from DFF import DFF

class SwitchMatrix:
    def __init__(self, mode, numOutput, param, tech, gate_params):
        
        self.mode = mode
        self.numOutput = numOutput
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        # DFF
        self.dff = DFF(numOutput, param, tech, gate_params)
        
        # Tg
        self.resTg = gate_params["resTg"]
        self.widthTgN = gate_params["widthTgN"]
        self.widthTgP = gate_params["widthTgP"]
        
    def CalculateArea(self):
        
        if self.mode == ROW_MODE:
            minCellHeight = self.gate_params["minCellHeight"]
            if self.tech.featureSize == 14 * 1e-9:
                minCellHeight *= (MAX_TRANSISTOR_HEIGHT_14nm/MAX_TRANSISTOR_HEIGHT)
            elif self.tech.featureSize == 10 * 1e-9:
                minCellHeight *= (MAX_TRANSISTOR_HEIGHT_10nm/MAX_TRANSISTOR_HEIGHT)
            elif self.tech.featureSize == 7 * 1e-9:
                minCellHeight *= (MAX_TRANSISTOR_HEIGHT_7nm/MAX_TRANSISTOR_HEIGHT)
            elif self.tech.featureSize == 5 * 1e-9:
                minCellHeight *= (MAX_TRANSISTOR_HEIGHT_5nm/MAX_TRANSISTOR_HEIGHT)
            elif self.tech.featureSize == 3 * 1e-9:
                minCellHeight *= (MAX_TRANSISTOR_HEIGHT_3nm/MAX_TRANSISTOR_HEIGHT)
            elif self.tech.featureSize == 2 * 1e-9:
                minCellHeight *= (MAX_TRANSISTOR_HEIGHT_2nm/MAX_TRANSISTOR_HEIGHT)
            elif self.tech.featureSize == 1 * 1e-9:
                minCellHeight *= (MAX_TRANSISTOR_HEIGHT_1nm/MAX_TRANSISTOR_HEIGHT)
            else:
                minCellHeight *= 1
            
            hTg = self.gate_params["hTg"]
            wTg = self.gate_params["wTg"]
            self.height = hTg * self.numOutput
            self.dff.CalculateArea()
            self.width = wTg*2 + self.dff.width
        
        else:
            hTg = self.gate_params["hTg"]
            wTg = self.gate_params["wTg"]
            self.width = wTg * 2 * self.numOutput
            self.dff.CalculateArea()
            self.height = hTg + self.dff.height
            
        self.area = self.width * self.height
        
    def CalculateLatency(self, capLoad, resLoad, numRead, numWrite):
        
        GP = self.gate_params
        rampInput = 1e20
        
        # dff
        self.dff.CalculateLatency(numRead)
        # TG
        capOutput = GP["capTgDrain"] * 3
        tr = GP["resTg"] * (capOutput + capLoad) + resLoad * capLoad / 2
        self.readLatency = horowitz(tr, 0, rampInput)["result"]
        self.readLatency *= numRead
        self.readLatency += self.dff.readLatency
        
        self.writeLatency = horowitz(tr, 0, rampInput)["result"]
        self.writeLatency *= numWrite
        self.writeLatency += self.dff.readLatency
        
    def CalculatePower(self, numRead, numWrite, numWriteCellPerOperationNeuro,
                       activityRowRead, activityColWrite):
        self.readDynamicEnergy = 0
        self.writeDynamicEnergy = 0
        self.leakage = 0
        # dff
        self.dff.CalculatePower(numRead, self.numOutput)
        
        self.leakage += self.dff.leakage
        
        # Neuro mode
        """
        if (mode == ROW_MODE) {
				readDynamicEnergy += (capTgDrain * 3) * cell.readVoltage * cell.readVoltage * numOutput * activityRowRead;
				readDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput * activityRowRead;
			} // No read energy in COL_MODE
        """
        if self.mode == ROW_MODE:
            readDynamicEnergy = (self.gate_params["capTgDrain"] * 3) * self.param.readVoltage * self.param.readVoltage * self.numOutput * activityRowRead
            readDynamicEnergy += (self.gate_params["capTgGateN"] + self.gate_params["capTgGateP"]) * self.tech.vdd * self.tech.vdd * self.numOutput * activityRowRead
            self.readDynamicEnergy += readDynamicEnergy
        # No read energy in COL_MODE
        self.readDynamicEnergy *= numRead
        self.readDynamicEnergy += self.dff.readDynamicEnergy
        
        # CMOS access, 1T1R
        if self.mode == ROW_MODE:
            """
            writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * 2;
            """
            writeDynamicEnergy = (self.gate_params["capTgGateN"] + self.gate_params["capTgGateP"]) * self.tech.vdd * self.tech.vdd * 2
            self.writeDynamicEnergy += writeDynamicEnergy
        else:
            """
            // LTP
                    writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numWritePulse * MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite) / 2;   // Selected columns
                    writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * (numOutput - MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite)/2);   // Unselected columns 
                    // LTD
                    writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numWritePulse * MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite) / 2;   // Selected columns
                    
                    writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
            """
            writeDynamicEnergy = (self.gate_params["capTgDrain"] * 3) * self.param.writeVoltage * self.param.writeVoltage * numWrite * min(numWriteCellPerOperationNeuro, self.numOutput*activityColWrite) / 2
            writeDynamicEnergy += (self.gate_params["capTgDrain"] * 3) * self.param.writeVoltage * self.param.writeVoltage * (self.numOutput - min(numWriteCellPerOperationNeuro, self.numOutput*activityColWrite)/2)
            writeDynamicEnergy += (self.gate_params["capTgDrain"] * 3) * self.param.writeVoltage * self.param.writeVoltage * numWrite * min(numWriteCellPerOperationNeuro, self.numOutput*activityColWrite) / 2
            writeDynamicEnergy += (self.gate_params["capTgGateN"] + self.gate_params["capTgGateP"]) * self.tech.vdd * self.tech.vdd * self.numOutput
            self.writeDynamicEnergy += writeDynamicEnergy
            
        self.writeDynamicEnergy *= numWrite
        self.writeDynamicEnergy += self.dff.writeDynamicEnergy