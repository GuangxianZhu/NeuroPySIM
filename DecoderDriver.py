import math
from constant import *
from MemCell import MemCell
from from_neurosim.build import FormulaBindings
from Gate_calculator import CalculateGateLeakage, CalculateGateArea, CalculateGateCapacitance, CalculateOnResistance, horowitz

class DecoderDriver:
    def __init__(self, numOutput:int, numLoad:int, param, tech, gate_params: dict):
        self.numOutput = numOutput
        self.numLoad = numLoad
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        # cell
        self.cell = MemCell(param)
        # INV
        self.widthInvN = self.gate_params["widthInvN"]
        self.widthInvP = self.gate_params["widthInvP"]
        # TG
        self.resTg = self.cell.resMemCellOn / self.numLoad * IR_DROP_TOLERANCE
        # special Tg for driver, not digital
        self.widthTgN = CalculateOnResistance(tech.featureSize, NMOS, param.temp, self.tech) * tech.featureSize / (self.resTg*2)
        self.widthTgP = CalculateOnResistance(tech.featureSize, PMOS, param.temp, self.tech) * tech.featureSize / (self.resTg*2)
        self.initialized = True
    
    def CalculateArea(self):
        # TG
        hTg = CalculateGateArea(INV, 1, self.widthTgN, self.widthTgP, self.gate_params["minCellHeight"], self.tech)['height']
        wTg = CalculateGateArea(INV, 1, self.widthTgN, self.widthTgP, self.gate_params["minCellHeight"], self.tech)['width']
        
        # INV
        hInv = CalculateGateArea(INV, 1, self.widthInvN, self.widthInvP, self.gate_params["minCellHeight"], self.tech)['height']
        wInv = CalculateGateArea(INV, 1, self.widthInvN, self.widthInvP, self.gate_params["minCellHeight"], self.tech)['width']
        
        # 1T1R row mode
        hUnit = max(hInv, hTg)
        wUnit = wInv + wTg*3
        self.height = hUnit * self.numOutput
        self.width = wUnit
        self.area = self.height * self.width
        
        # calculate cap by width
        # INV
        self.capInvInput = self.gate_params["capInvInput"]
        self.capInvOutput = self.gate_params["capInvOutput"] 
        
        self.capTgGateN = FormulaBindings.CalculateGateCap(self.widthTgN, self.tech)
        self.capTgGateP = FormulaBindings.CalculateGateCap(self.widthTgP, self.tech)
        self.capTgDrain = CalculateGateCapacitance(INV, 1, self.widthTgN, self.widthTgP, hTg, self.tech)['capOutput']
        
    def CalculateLatency(self, capLoad1, capLoad2, resLoad, 
                         numRead, numWrite):
        self.rampInput = 1e20
        capOutput = self.capTgDrain*4 + self.capTgGateN*0.5 + self.capTgGateP*0.5
        tr = self.resTg * (capOutput + capLoad1) + resLoad * capLoad1 / 2
        self.readLatency = horowitz(tr, 0, self.rampInput)['result']
        self.readLatency *= numRead
        
        self.writeLatency = horowitz(tr, 0, self.rampInput)['result']
        self.writeLatency *= numWrite
        
    def CalculatePower(self, numReadCellPerOp, numWriteCellPerOp,
                       numRead, numWrite):
        self.leakage, self.readDynamicEnergy, self.writeDynamicEnergy = 0,0,0
        
        # leakage power
        self.leakage += CalculateGateLeakage(INV, 1, self.widthInvN, self.widthInvP, self.param.temp, self.tech)
    
        # Read dynamic energy
        # 1T1R
        # Selected SLs and BLs are floating
        # Unselected SLs and BLs are GND
        self.readDynamicEnergy += (self.capInvInput + self.capTgGateN * 2 + self.capTgGateP) * (self.tech.vdd ** 2) * numReadCellPerOp
        self.readDynamicEnergy += (self.capInvOutput + self.capTgGateP * 2 + self.capTgGateN) * (self.tech.vdd ** 2) * numReadCellPerOp
        self.readDynamicEnergy *= numRead
        self.readPower = self.readDynamicEnergy/self.readLatency
        
        # Write dynamic energy
        # 1T1R, Worst case: RESET operation (because SL cap is larger than BL cap)
        self.writeDynamicEnergy += (self.capInvInput + self.capTgGateN * 2 + self.capTgGateP) * (self.tech.vdd ** 2) * numWriteCellPerOp
        self.writeDynamicEnergy += (self.capInvOutput + self.capTgGateP * 2 + self.capTgGateN) * (self.tech.vdd ** 2) * numWriteCellPerOp
        self.writeDynamicEnergy += (self.capTgDrain * 2) * (self.param.writeVoltage**2) * numWriteCellPerOp
        self.writeDynamicEnergy *= numWrite
        self.writePower = self.writeDynamicEnergy/self.writeLatency

if __name__ == "__main__":
    
    from Gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(22, FormulaBindings.DeviceRoadmap.LSTP, FormulaBindings.TransistorType.conventional)
    param = FormulaBindings.Param()
    param.memcelltype = RRAM
    gate_params = compute_gate_params(param, tech)
    
    dec_dri = DecoderDriver(1024, 1024, param, tech, gate_params) # numRow, numCol
    dec_dri.CalculateArea()
    dec_dri.CalculateLatency(1e-15, 1e-15, 1e-3, 1, 1)
    dec_dri.CalculatePower(1, 1, 1, 1)
    
    print("DecoderDriver")
    print("Area: {:.5f}".format(dec_dri.area))
    print("Read Latency: {:.5f}".format(dec_dri.readLatency))
    print("Read Power: {:.5f}".format(dec_dri.readPower))
    print("Write Latency: {:.5f}".format(dec_dri.writeLatency))
    print("Write Power: {:.5f}".format(dec_dri.writePower))
    print("Leakage: {:.5f}".format(dec_dri.leakage))
    print("Dynamic Energy: {:.5f}".format(dec_dri.readDynamicEnergy))
    print("Dynamic Energy: {:.5f}".format(dec_dri.writeDynamicEnergy))
    