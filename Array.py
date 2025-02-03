import math
from MemCell import MemCell
from constant import *
from from_neurosim.build import FormulaBindings
from gate_calculator import horowitz

from Decoder import RowDecoder
from DecoderDriver import DecoderDriver
from WLNewDecoderDriver import WLNewDecoderDriver
from Mux import Mux
from MultilevelSenseAmp import MultilevelSenseAmp
from Adder import Adder
from DFF import DFF
from SwitchMatrix import SwitchMatrix
from ShiftAdd import ShiftAdd

def ceil(x):
    return math.ceil(x)

def log2(x):
    return math.log2(x)

class Array:
    
    def __init__(self, numRow, numCol, param, tech, gate_params):
        self.numRow = numRow
        self.numCol = numCol
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        # RW params
        self.numColMuxed = param.numColMuxed
        self.activityRowRead = 1.0 # % of rows that are activated in read, in real scene should be calculated from real data
        self.activityRowWrite = 1.0 # % of rows that are activated in write
        # self.numReadCellPerOperationNeuro = _numCol
        self.numWriteCellPerOperationNeuro = numCol
        self.adderBit = math.ceil(math.log(self.numRow, 2)) + self.param.cellBit
        self.numAdder = self.numCol/param.synapseBit
        
        self.unitWireRes = 1
        
        self.cell = MemCell(param)
        self.cell.resMemCellAvg = 1/(1/(param.resistanceOn + self.cell.resCellAccess ) * self.numRow/2.0 + 1/(param.resistanceOff + self.cell.resCellAccess )* self.numRow/2.0) * self.numRow
        
        # init components
        self.wlDecoder = RowDecoder(numAddrRow = math.ceil(math.log(self.numRow, 2)), MUX = False, 
                                    param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.wlDecoderDriver = DecoderDriver(numOutput = self.numRow, numLoad = self.numCol,
                                             param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.wlNewDecoderDriver = WLNewDecoderDriver(numWLRow = self.numRow,
                                                     param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.mux = Mux(numInput = math.ceil(self.numCol/self.numColMuxed), numSelection = self.numColMuxed,
                       param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.muxDecoder = RowDecoder(numAddrRow = math.ceil(math.log(self.numColMuxed, 2)), MUX = True,
                                        param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.multilevelSenseAmp = MultilevelSenseAmp(numCol = self.numCol/self.numColMuxed, levelOutput = param.levelOutput,
                                                     param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.adder = Adder(numBit = self.adderBit-1, numAdder = self.numCol/self.numColMuxed,
                           param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.dff = DFF(numDff = self.adderBit*self.numAdder,
                       param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.slSwitchMatrix = SwitchMatrix(mode = ROW_MODE, numOutput=self.numRow,  
                                           param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.shiftAddWeight = ShiftAdd(numUnit = self.numAdder, numAdderBit = self.adderBit, numReadPulse = (param.synapseBit-1)*self.param.cellBit+1,
                                       param=self.param, tech=self.tech, gate_params=self.gate_params)
        # still need to convert from NeuroSIM, (cpp to python)
        # muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false)
        
        
        if self.cell.MemCellType == SRAM:
            raise NotImplementedError
            # self.lengthRow = self.numCol * self.cell.widthInFeatureSize * tech.featureSize
            # self.lengthCol = self.numRow * self.cell.heightInFeatureSize * tech.featureSize
            # param.arraywidthunit = self.cell.widthInFeatureSize * tech.featureSize
            # param.arrayheight = self.numRow * self.cell.heightInFeatureSize * tech.featureSize
        
        
        elif self.cell.MemCellType == RRAM:
            cellHeight = self.cell.heightInFeatureSize
            cellWidth = self.cell.widthInFeatureSize
            # 1T1R cell
            self.lengthRow = self.numCol * cellWidth * tech.featureSize
            self.lengthCol = self.numRow * cellHeight * tech.featureSize
            param.arraywidthunit = cellWidth * tech.featureSize
            param.arrayheight = self.numRow * cellHeight * tech.featureSize
        
        self.capRow1 = self.lengthRow * 0.2e-15/1e-6 # BL for 1T1R, WL for Cross-point and SRAM
        self.capRow2 = self.lengthRow * 0.2e-15/1e-6 # WL for 1T1R
        self.capCol = self.lengthCol * 0.2e-15/1e-6
        
        self.resRow = self.lengthRow * param.Metal1_unitwireresis
        self.resCol = self.lengthCol * param.Metal0_unitwireresis
        
        # if self.cell.MemCellType == SRAM:
        #     # firstly calculate the CMOS resistance and capacitance
        #     resCellAccess = CalculateOnResistance(param.widthAccessCMOS*tech.featureSize, NMOS, param.temperature, tech)
        #     capCellAccess = CalculateDrainCap(param.widthAccessCMOS*tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT*tech.featureSize, tech)
        
    def CalculateArea(self):
        if self.cell.MemCellType == RRAM:
            # array only
            self.heightArray = self.lengthCol
            self.widthArray = self.lengthRow
            self.areaArray = self.heightArray * self.widthArray
            
            # peripheral circuits
            self.wlDecoder.CalculateArea()
            self.wlNewDecoderDriver.CalculateArea() # 1T1R, cmos access
            self.mux.CalculateArea()
            self.muxDecoder.CalculateArea()
            self.multilevelSenseAmp.CalculateArea(self.widthArray)
            self.adder.CalculateArea()
            self.dff.CalculateArea()
            self.slSwitchMatrix.CalculateArea()
            self.shiftAddWeight.CalculateArea()
            # others, need to convert from NeuroSIM
            
            self.height = self.slSwitchMatrix.height + self.heightArray \
                        + self.mux.height + self.muxDecoder.height \
                        + self.multilevelSenseAmp.height \
                        + self.adder.height + self.dff.height \
                        + self.shiftAddWeight.height
            self.width = max(self.widthArray + self.wlDecoder.width + self.wlNewDecoderDriver.width, self.adder.width)
            self.area = self.height * self.width
            
            self.usedArea = self.areaArray + self.wlDecoder.area + self.wlNewDecoderDriver.area + self.multilevelSenseAmp.area + self.adder.area + self.dff.area + self.mux.area + self.slSwitchMatrix.area + self.shiftAddWeight.area
            assert self.usedArea <= self.area, "Error: Array area is smaller than the area of all components"
            self.emptyArea = self.area - self.usedArea

            
    def CalculateLatency(self):
        # from NeuroSim.main.cpp, the CalculateclkFreq is true, so we choose this branch
        readLatency, writeLatency = 0,0
        readLatencyADC = 0
        
        if self.cell.MemCellType == RRAM:
            
            # conventionalSequential, 
            
            self.capBL = self.lengthCol * 0.2e-15/1e-6
            self.colRamp = 0
            tau = self.capCol * self.cell.resMemCellAvg
            self.colRamp = horowitz(tau, 0, 1e20)['rampOutput']
            self.colDelay = tau * 0.2 # assume the 15~20% voltage drop is enough for sensing
            
            self.numWriteOperationPerRow = math.ceil(self.numCol * self.activityRowWrite) / self.numWriteCellPerOperationNeuro
            
            # CalculateclkFreq: true
            self.wlDecoder.CalculateLatency(self.capRow1, 0, 
                                            self.resRow, self.numCol, 
                                            1, 2*self.numWriteOperationPerRow*self.numRow*self.activityRowWrite)
            # CMOS access
            self.wlNewDecoderDriver.CalculateLatency(self.capRow1, self.capRow1,
                                                    self.resRow, 1,
                                                    2*self.numWriteOperationPerRow*self.numRow*self.activityRowWrite)
            
            self.mux.CalculateLatency(numRead=1)
            self.muxDecoder.CalculateLatency(capLoad1=self.mux.capTgGateN*ceil(self.numCol/self.numColMuxed),
                                             capLoad2=self.mux.capTgGateP*ceil(self.numCol/self.numColMuxed),
                                             resLoad=0,
                                             colnum=0,
                                             numRead=1, numWrite=0)
            self.multilevelSenseAmp.CalculateLatency(currentMode=True, numColMuxed=1, numRead=1)
            self.dff.CalculateLatency(1)
            
            readLatency += self.wlDecoder.readLatency + self.wlNewDecoderDriver.readLatency # hide mux + muxdecoder + amp
            readLatency += self.colDelay
            readLatency += self.multilevelSenseAmp.readLatency
            # TODO: SA encoder.readLatency
        
        self.readLatency = readLatency
            
    
    def CalculatePower(self):
        
        self.readDynamicEnergy, self.writeDynamicEnergy, self.readDynamicEnergyArray = 0,0,0
        
        if self.cell.MemCellType == RRAM:
            numReadCells = math.ceil(self.numCol / self.param.numColMuxed)
            numWriteCells = math.ceil(self.numCol)
            numWriteOperationPerRow = 1
            capBL = self.lengthCol * 0.2e-15/1e-6
            
            # readDynamicEnergy *****
            # Array
            readDynamicEnergyArray = 0
            readDynamicEnergyArray += capBL * (self.param.readVoltage ** 2) * numReadCells # Selected BLs activityColWrite
            readDynamicEnergyArray += self.capRow2 * self.tech.vdd * self.tech.vdd; # Selected WL
            readDynamicEnergyArray *= self.numRow * self.activityRowRead * self.param.numColMuxed

            # Pheripheral circuits
            readDynamicEnergy = 0
            self.wlDecoder.CalculatePower(self.numRow*self.numColMuxed, 2*self.numRow)
            readDynamicEnergy += self.wlDecoder.readDynamicEnergy
            self.wlNewDecoderDriver.CalculatePower(numRead = self.numRow*self.activityRowRead*self.numColMuxed, 
                                                    numWrite = 2*numWriteOperationPerRow*self.numRow*self.activityRowWrite)
            self.mux.CalculatePower(self.numColMuxed)
            self.muxDecoder.CalculatePower(self.numColMuxed, 1)
            self.multilevelSenseAmp.CalculatePower(self.numRow*self.activityRowRead)
            self.adder.CalculatePower(_numRead = self.numColMuxed*self.numRow*self.activityRowRead, 
                                      _numAdderPerOperation = numReadCells)
            self.dff.CalculatePower(_numRead = self.numColMuxed*self.numRow*self.activityRowRead,
                                    numDffPerOperation = numReadCells*(math.ceil(math.log2(self.numRow))/2 + self.param.cellBit))
            self.slSwitchMatrix.CalculatePower(self.numRow, self.numCol, self.activityRowRead, self.activityRowWrite)
            self.shiftAddWeight.CalculatePower((self.param.synapseBit-1) * math.ceil(self.numColMuxed/self.param.synapseBit)),
            # others: muxdecoder ...
            
            readDynamicEnergy = self.wlDecoder.readDynamicEnergy
            readDynamicEnergy += self.wlNewDecoderDriver.readDynamicEnergy
            readDynamicEnergy += self.mux.readDynamicEnergy
            readDynamicEnergy += self.muxDecoder.readDynamicEnergy
            readDynamicEnergy += self.multilevelSenseAmp.readDynamicEnergy
            readDynamicEnergy += self.adder.readDynamicEnergy
            readDynamicEnergy += self.dff.readDynamicEnergy
            # readDynamicEnergy += self.slSwitchMatrix.readDynamicEnergy # no readDynamicEnergy
            readDynamicEnergy += self.shiftAddWeight.readDynamicEnergy
            # others: muxdecoder ...
            
            # writeDynamicEnergy (skip) *****
            writeDynamicEnergy = 0
            
            # leakage
            leakage = 0
            leakage += self.wlDecoder.leakage
            leakage += self.wlNewDecoderDriver.leakage
            leakage += self.dff.leakage
            leakage += self.slSwitchMatrix.leakage
            # mux, senseamp no leakage
            # others: muxdecoder, ...
        
        elif self.cell.MemCellType == SRAM:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.readDynamicEnergy = readDynamicEnergy
        self.readDynamicEnergyArray = readDynamicEnergyArray
        self.writeDynamicEnergy = writeDynamicEnergy
        self.leakage = leakage
            
    def printInfo(self):
        def format_percentage(value, total):
            """计算百分比并保留三位小数，如果 total 为零则返回 0.000%"""
            return f"{(value / total * 100):.3f}%" if total != 0 else "0.000%"

        print(f"Summary of Array:")
        # Area, height, width
        print(f"Area: {self.area*1e12:.3f}μm2, Height: {self.height*1e6:.3f}μm, Width: {self.width*1e6:.3f}μm")
        print(f"usedArea: {self.usedArea*1e12:.3f}μm2 ({format_percentage(self.usedArea, self.area)})")
        print(f"Breakdown of Each Component:")
        print(f"\tMemoryCrossbar: {self.areaArray*1e12:.3f}μm2, ({format_percentage(self.areaArray, self.area)})")
        print(f"\tWLDecoder: {self.wlDecoder.area*1e12:.3f}μm2, ({format_percentage(self.wlDecoder.area, self.area)})")
        print(f"\tWLDecoderDriver: {self.wlNewDecoderDriver.area*1e12:.3f}μm2, ({format_percentage(self.wlNewDecoderDriver.area, self.area)})")
        print(f"\tMux: {self.mux.area*1e12:.3f}μm2, ({format_percentage(self.mux.area, self.area)})")
        print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.area*1e12:.3f}μm2, ({format_percentage(self.multilevelSenseAmp.area, self.area)})")
        print(f"\tAdder: {self.adder.area*1e12:.3f}μm2, ({format_percentage(self.adder.area, self.area)})")
        print(f"\tDFF: {self.dff.area*1e12:.3f}μm2, ({format_percentage(self.dff.area, self.area)})")
        print(f"\tSwitchMatrix: {self.slSwitchMatrix.area*1e12:.3f}μm2, ({format_percentage(self.slSwitchMatrix.area, self.area)})")
        print("\n")
        
        # Latency
        print(f"Read Latency: {self.readLatency*1e9:.3f}ns")
        print(f"Breakdown of Each Component:")
        print(f"\tWLDecoder: {self.wlDecoder.readLatency*1e9:.3f}ns, ({format_percentage(self.wlDecoder.readLatency, self.readLatency)})")
        print(f"\tWLDecoderDriver: {self.wlNewDecoderDriver.readLatency*1e9:.3f}ns, ({format_percentage(self.wlNewDecoderDriver.readLatency, self.readLatency)})")
        print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.readLatency*1e9:.3f}ns, ({format_percentage(self.multilevelSenseAmp.readLatency, self.readLatency)})")
        print("\n")

        # Power
        print(f"Read Dynamic Energy: {self.readDynamicEnergy*1e9:.3f}nJ")
        print(f"Breakdown of Each Component:")
        print(f"\tWLDecoder: {self.wlDecoder.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.wlDecoder.readDynamicEnergy, self.readDynamicEnergy)})")
        print(f"\tWLDecoderDriver: {self.wlNewDecoderDriver.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.wlNewDecoderDriver.readDynamicEnergy, self.readDynamicEnergy)})")
        print(f"\tMux: {self.mux.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.mux.readDynamicEnergy, self.readDynamicEnergy)})")
        print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.multilevelSenseAmp.readDynamicEnergy, self.readDynamicEnergy)})")
        print(f"\tAdder: {self.adder.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.adder.readDynamicEnergy, self.readDynamicEnergy)})")
        print(f"\tDFF: {self.dff.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.dff.readDynamicEnergy, self.readDynamicEnergy)})")
        print("\n")

if __name__ == '__main__':
    
    from gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(22, FormulaBindings.DeviceRoadmap.LSTP, FormulaBindings.TransistorType.conventional)
    param = FormulaBindings.Param()
    param.memcelltype = RRAM
    gate_params = compute_gate_params(param, tech)
    
    # test
    array = Array(128, 128, param, tech, gate_params)
    
    array.CalculateArea()
    array.CalculateLatency()
    array.CalculatePower()
    array.printInfo()
    
    
    