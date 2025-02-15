import math
from constant import *
from from_neurosim.build import FormulaBindings
from Gate_calculator import CalculateGateLeakage, horowitz

class RowDecoder:
    
    def __init__(self, numAddrRow:int, MUX:bool, param, tech, gate_params: dict):

        self.numAddrRow = numAddrRow
        self.MUX = MUX
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        # INV
        self.widthInvN = self.gate_params["widthInvN"]
        self.widthInvP = self.gate_params["widthInvP"]
        self.numInv = numAddrRow
        
        # NAND2 (2-input NAND)
        self.widthNandN = self.gate_params["widthNandN"]
        self.widthNandP = self.gate_params["widthNandP"]
        self.numNand = int(4*math.floor(numAddrRow/2))
        
        # NOR2 (ceil(N/2) inputs)
        self.widthNorN = self.gate_params["widthNorN"]
        self.widthNorP = math.ceil(numAddrRow/2) * self.gate_params["widthNorP"]
        self.numNor = int(math.pow(2, numAddrRow))
        
        if numAddrRow > 2: 
            self.numNor = pow(2, numAddrRow)
            self.numMetalConnection = self.numNand + (numAddrRow%2) * 2
        else: 
            self.numNor = 0
            self.numMetalConnection = 0
            
        self.widthDriverInvN = self.gate_params["widthDriverInvN"]
        self.widthDriverInvP = self.gate_params["widthDriverInvP"]
        self.initialized = True
        
    def CalculateArea(self):
        # INV: constant
        self.hInv, self.wInv, self.aInv = self.gate_params["hInv"], self.gate_params["wInv"], self.gate_params["aInv"]
        # NAND2: constant
        self.hNand, self.wNand, self.aNand = self.gate_params["hNand"], self.gate_params["wNand"], self.gate_params["aNand"]
        # NOR (ceil(N/2) inputs): dynamic
        AHW_Nor = FormulaBindings.CalculateGateArea(1, math.ceil(self.numAddrRow/2), self.widthNorN, self.widthNorP,
                                                    self.tech.featureSize * MAX_TRANSISTOR_HEIGHT,
                                                    self.tech)
        self.hNor, self.wNor = AHW_Nor['height'], AHW_Nor['width']
        # Output Driver INV
        self.hDriverInv, self.wDriverInv = self.gate_params["hInv"], self.gate_params["wInv"]
        
        Metal2_pitch = M2_PITCH
        Metal3_pitch = M3_PITCH
        height = max(self.hNor*self.numNor, self.hNand*self.numNand)
        width = self.wInv + self.wNand + Metal3_pitch * self.numMetalConnection * self.tech.featureSize + self.wNor
        if self.MUX:
            width += self.wNand + self.wDriverInv * 2
        else:
            width += self.wDriverInv * 2
            
        self.height = height
        self.width = width
        self.area = height * width
        
        return self.area
    
    def CalculateLatency(self, capLoad1, capLoad2, numRead, numWrite):
        """
        capLoad1: 线路的等效电容负载
        capLoad2: 线路的等效电容负载
        resLoad: 线路的等效电阻负载
        colnum: 读取的列数
        numRead, numWrite: 读取和写入的次数
        """
        self.rampInput = 1e20 # 输入信号的上升/下降斜率
        self.capLoad1, self.capLoad2 = capLoad1, capLoad2
        
        self.readLatency, self.writeLatency = 0, 0
        
        # capIn Out
        # INV
        self.capInvInput, self.capInvOutput = self.gate_params["capInvInput"], self.gate_params["capInvOutput"]
        # NAND2
        if self.numNand > 0:
            self.capNandInput, self.capNandOutput = self.gate_params["capNandInput"], self.gate_params["capNandOutput"]
        else:
            self.capNandInput, self.capNandOutput = 0, 0
        # NOR (ceil(N/2) inputs)
        if self.numNor > 0:
            self.capNorInput, self.capNorOutput = self.gate_params["capNorInput"], self.gate_params["capNorOutput"]
        else:
            self.capNorInput, self.capNorOutput = 0, 0
        # Output Driver INV
        capIO_DriverInv = FormulaBindings.CalculateGateCapacitance(INV, 1, self.widthDriverInvN, self.widthDriverInvP,
                                                                  self.hDriverInv, self.tech)
        self.capDriverInvInput, self.capDriverInvOutput = capIO_DriverInv['capInput'], capIO_DriverInv['capOutput']
        
        # Latency 
        # INV
        resPullDown = FormulaBindings.CalculateOnResistance(self.widthInvN, NMOS, self.param.temp, self.tech)
        if self.numNand > 0:
            tr = resPullDown * (self.capInvOutput + self.capNandInput * 2)
        else:
            tr = resPullDown * (self.capInvOutput + capLoad1)
        gm = FormulaBindings.CalculateTransconductance(self.widthInvN, NMOS, self.tech)
        beta = 1 / (resPullDown * gm)
        read_lat = FormulaBindings.horowitz(tr, beta, self.rampInput)
        self.readLatency += read_lat['result']
        # self.rampInvOutput = read_lat['rampOutput']
        write_lat = FormulaBindings.horowitz(tr, beta, self.rampInput)
        self.writeLatency += write_lat['result']
        
        # NAND2
        if self.numNand>0:
            resPullDown = FormulaBindings.CalculateOnResistance(self.widthNandN, NMOS, self.param.temp, self.tech)
            tr = resPullDown * (self.capNandOutput + capLoad1)
            gm = FormulaBindings.CalculateTransconductance(self.widthNandN, NMOS, self.tech)
            beta = 1 / (resPullDown * gm)
            read_lat = FormulaBindings.horowitz(tr, beta, self.rampInput)
            self.readLatency += read_lat['result']
            self.rampNandOutput = read_lat['rampOutput']
            write_lat = FormulaBindings.horowitz(tr, beta, self.rampInput)
            self.writeLatency += write_lat['result']
        
        # NOR
        if self.numNor>0:
            resPullUp = FormulaBindings.CalculateOnResistance(self.widthNorP, PMOS, self.param.temp, self.tech)
            tr = resPullUp * (self.capNorOutput + capLoad2)
            gm = FormulaBindings.CalculateTransconductance(self.widthNorP, PMOS, self.tech)
            beta = 1 / (resPullUp * gm)
            read_lat = FormulaBindings.horowitz(tr, beta, self.rampNandOutput)
            self.readLatency += read_lat['result']
            self.rampNorOutput = read_lat['rampOutput']
            write_lat = FormulaBindings.horowitz(tr, beta, self.rampNandOutput)
            self.writeLatency += write_lat['result']
            
        if self.MUX:
            # 1st NAND
            resPullDown = FormulaBindings.CalculateOnResistance(self.widthNandN, NMOS, self.param.temp, self.tech)
            tr = resPullDown * (self.capNandOutput + self.capInvInput * 2)
            gm = FormulaBindings.CalculateTransconductance(self.widthNandN, NMOS, self.tech)
            beta = 1 / (resPullDown * gm)
            read_lat = FormulaBindings.horowitz(tr, beta, self.rampNorOutput)
            self.readLatency += read_lat['result']
            self.writeLatency += read_lat['result']
            self.rampNandOutput = read_lat['rampOutput']
            
            # 2nd INV
            resPullUp = FormulaBindings.CalculateOnResistance(self.widthDriverInvN, NMOS, self.param.temp, self.tech)
            tr = resPullUp * (self.capNandOutput + capLoad1)
            gm = FormulaBindings.CalculateTransconductance(self.widthDriverInvN, NMOS, self.tech)
            beta = 1 / (resPullUp * gm)
            self.readLatency += horowitz(tr, beta, self.rampNandOutput)['result']
            self.writeLatency += horowitz(tr, beta, self.rampNandOutput)['result']
            self.rampInvOutput = horowitz(tr, beta, self.rampNandOutput)['rampOutput']
            
            # 3rd INV
            resPullDown = FormulaBindings.CalculateOnResistance(self.widthDriverInvN, NMOS, self.param.temp, self.tech)
            tr = resPullDown * (self.capDriverInvOutput + self.capLoad2)
            gm = FormulaBindings.CalculateTransconductance(self.widthDriverInvN, NMOS, self.tech)
            beta = 1 / (resPullDown * gm)
            self.readLatency += horowitz(tr, beta, self.rampInvOutput)['result']
            self.writeLatency += horowitz(tr, beta, self.rampInvOutput)['result']
            
        self.readLatency *= numRead
        self.writeLatency *= numWrite
        
    def CalculatePower(self, numRead, numWrite):
        self.leakage = 0.0
        # leakage
        # INV
        self.leakage += self.gate_params["leakageInv"] * self.numInv
        # NAND2
        self.leakage += self.gate_params["leakageNand"] * self.numNand
        # NOR: dynamic
        self.leakage += CalculateGateLeakage(NOR, math.ceil(self.numAddrRow/2), self.widthNorN, self.widthNorP, self.param.temp, self.tech) * self.tech.vdd * self.numNor
        # Output Driver INV
        if self.MUX:
            self.leakage += self.gate_params["leakageNand"] * 2
            self.leakage += self.gate_params["leakageInv"] * 2
        else:
            self.leakage += self.gate_params["leakageInv"] * 2
            
        # read dynamic power

        # INV
        self.readDynamicEnergy = 0.0
        self.readDynamicEnergy += (self.capInvInput + self.capNandInput * 2) * self.tech.vdd * self.tech.vdd * math.floor(self.numAddrRow/2)*2
        self.readDynamicEnergy += (self.capInvInput + self.capNandInput * 2) * self.tech.vdd * self.tech.vdd * (self.numAddrRow -  math.floor(self.numAddrRow/2)*2)
        # NAND2
        self.readDynamicEnergy += (self.capNandOutput + self.capNorInput * self.numNor/4) * self.tech.vdd * self.tech.vdd * self.numNand/4
        # INV
        self.writeDynamicEnergy = 0.0
        self.writeDynamicEnergy += (self.capInvInput + self.capNandInput * 2) * self.tech.vdd * self.tech.vdd * math.floor(self.numAddrRow/2)*2
        self.writeDynamicEnergy += (self.capInvInput + self.capNandInput * self.numNor/2) * self.tech.vdd * self.tech.vdd * (self.numAddrRow -  math.floor(self.numAddrRow/2)*2)
        # NAND2
        self.writeDynamicEnergy += (self.capNandOutput + self.capNorInput * self.numNor/4) * self.tech.vdd * self.tech.vdd * self.numNand/4
        
        # NOR (ceil(N/2) inputs)
        if self.MUX:
            self.readDynamicEnergy += (self.capNorOutput + self.capNandInput) * self.tech.vdd * self.tech.vdd
        else:
            self.readDynamicEnergy += (self.capNorOutput + self.capInvInput) * self.tech.vdd * self.tech.vdd
            
        # Output driver or Mux enable circuit
        if self.MUX:
            self.readDynamicEnergy += (self.capNandOutput + self.capDriverInvInput) * self.tech.vdd * self.tech.vdd
            self.readDynamicEnergy += (self.capDriverInvOutput + self.capDriverInvInput) * self.tech.vdd * self.tech.vdd
            self.readDynamicEnergy += self.capDriverInvOutput * self.tech.vdd * self.tech.vdd
            
            self.writeDynamicEnergy += (self.capNandOutput + self.capDriverInvInput) * self.tech.vdd * self.tech.vdd
            self.writeDynamicEnergy += (self.capDriverInvOutput + self.capDriverInvInput) * self.tech.vdd * self.tech.vdd
            self.writeDynamicEnergy += self.capDriverInvOutput * self.tech.vdd * self.tech.vdd
        else:
            self.readDynamicEnergy += (self.capDriverInvInput + self.capDriverInvOutput) * self.tech.vdd * self.tech.vdd * 2
            self.writeDynamicEnergy += (self.capDriverInvInput + self.capDriverInvOutput) * self.tech.vdd * self.tech.vdd * 2
        
        self.readDynamicEnergy *= numRead
        self.writeDynamicEnergy *= numWrite
        

if __name__ == "__main__":
    
    from Gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(22, FormulaBindings.DeviceRoadmap.LSTP, FormulaBindings.TransistorType.conventional)
    param = FormulaBindings.Param()
    param.memcelltype = RRAM
    gate_params = compute_gate_params(param, tech)
    
    rowDecoder = RowDecoder(numAddrRow=4, MUX=True, param=param, tech=tech, gate_params=gate_params)
    
    print(f"Number of INV: {rowDecoder.numInv}")
    print(f"Number of NAND2: {rowDecoder.numNand}")
    print(f"Number of NOR2: {rowDecoder.numNor}")
    print(f"Number of Metal Connections: {rowDecoder.numMetalConnection}")
    print(f"Width of Driver INV N: {rowDecoder.widthDriverInvN}")
    print(f"Width of Driver INV P: {rowDecoder.widthDriverInvP}")
    print(f"Width of INV N: {rowDecoder.widthInvN}")
    print(f"Width of INV P: {rowDecoder.widthInvP}")
    print(f"Width of NAND2 N: {rowDecoder.widthNandN}")
    print(f"Width of NAND2 P: {rowDecoder.widthNandP}")
    print(f"Width of NOR2 N: {rowDecoder.widthNorN}")
    print(f"Width of NOR2 P: {rowDecoder.widthNorP}")
    print(f"Initialized: {rowDecoder.initialized}")
    
    area = rowDecoder.CalculateArea()
    print(f"Area: {area:.4e} µm²")
    print(f"Height: {rowDecoder.height} µm")
    print(f"Width: {rowDecoder.width} µm")
    
    rowDecoder.CalculateLatency(1e-15, 1e-15, 1e-3, 1, 1, 1)
    print(f"Read Latency: {rowDecoder.readLatency:.4e} s")
    
    rowDecoder.CalculatePower(1, 1)
    print(f"Leakage: {rowDecoder.leakage:.4e} W")
    print(f"Read Dynamic Energy: {rowDecoder.readDynamicEnergy:.4e} J")
    print(f"Write Dynamic Energy: {rowDecoder.writeDynamicEnergy:.4e} J")
    