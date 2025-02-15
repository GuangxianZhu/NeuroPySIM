import math
from MemCell import MemCell
from constant import *
from from_neurosim.build import FormulaBindings
from Gate_calculator import horowitz

from Decoder import RowDecoder
from NewSwitchMatrix import NewSwitchMatrix
from Mux import Mux
from MultilevelSenseAmp import MultilevelSenseAmp
from MultilevelSAEncoder import MultilevelSAEncoder
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
        self.totalNumWritePulse = 0 # modified in tile layer
        self.numReadCellPerOperationNeuro = numCol
        self.numWriteCellPerOperationNeuro = numCol
        self.adderBit = math.ceil(math.log(self.numRow, 2)) + self.param.cellBit
        self.numAdder = self.numCol/param.synapseBit
        
        self.unitWireRes = 1
        
        self.cell = MemCell(param)
        # self.cell.resMemCellAvg = 1/(1/(self.cell.resMemCellOn ) * self.numRow/2.0 + 1/(self.cell.resMemCellOff)* self.numRow/2.0) * self.numRow
        # Use a single average resistance value in system simulations to represent the mix of low-resistance and high-resistance cells throughout the array
        self.cell.resMemCellAvg = 2 * self.cell.resMemCellOn * self.cell.resMemCellOff / (self.cell.resMemCellOn + self.cell.resMemCellOff)

        # init components: conventionalParallel Mode in NeuroSim
        # 1. digital crossbar
        self.wlNewSwitchMatrix = NewSwitchMatrix(numOutput = self.numRow, activityRowRead = self.activityRowRead,
                                                param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.slSwitchMatrix = SwitchMatrix(mode = COL_MODE, numOutput=self.numCol, 
                                           param=self.param, tech=self.tech, gate_params=self.gate_params)
        
        self.mux = Mux(numInput = ceil(self.numCol/self.numColMuxed), numSelection = self.numColMuxed,
                       param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.muxDecoder = RowDecoder(numAddrRow = ceil(log2(self.numColMuxed)), MUX = True,
                                    param=self.param, tech=self.tech, gate_params=self.gate_params)
        
        self.multilevelSenseAmp = MultilevelSenseAmp(numCol=self.numCol/self.numColMuxed, 
                                                     levelOutput = param.levelOutput,
                                                     numReadCellPerOperationNeuro=self.numReadCellPerOperationNeuro,
                                                     parallel=False,
                                                     currentMode=True,
                                                     param=self.param, tech=self.tech, gate_params=self.gate_params)
        self.multilevelSAEncoder = MultilevelSAEncoder(numlevel = param.levelOutput, numEncoder = self.numCol/self.numColMuxed,
                                                    param=self.param, tech=self.tech, gate_params=self.gate_params)
        
        self.shiftAddWeight = ShiftAdd(numUnit = self.numAdder, numAdderBit = self.adderBit, numReadPulse = (param.synapseBit-1)*self.param.cellBit+1,
                                       param=self.param, tech=self.tech, gate_params=self.gate_params)
        
        
        if self.cell.MemCellType == SRAM:
            raise NotImplementedError
            # self.lengthRow = self.numCol * self.cell.widthInFeatureSize * tech.featureSize
            # self.lengthCol = self.numRow * self.cell.heightInFeatureSize * tech.featureSize
        
        
        elif self.cell.MemCellType == RRAM:
            cellHeight = self.cell.heightInFeatureSize
            cellWidth = self.cell.widthInFeatureSize
            # 1T1R cell
            self.lengthRow = self.numCol * cellWidth * tech.featureSize
            self.lengthCol = self.numRow * cellHeight * tech.featureSize
        
        self.capRow1 = self.lengthRow * 0.2e-15/1e-6 # BL for 1T1R, WL for Cross-point and SRAM
        self.capRow2 = self.lengthRow * 0.2e-15/1e-6 # WL for 1T1R
        self.capCol = self.lengthCol * 0.2e-15/1e-6
        
        self.resRow = self.lengthRow * param.Metal1_unitwireresis
        self.resCol = self.lengthCol * param.Metal0_unitwireresis
        
        
    def CalculateArea(self):
        
        if self.cell.MemCellType == RRAM:
            
            if self.param.RRAM_mode == RRAM_DIGITAL:
                # crossbar memory array
                self.heightArray = self.lengthCol
                self.widthArray = self.lengthRow
                self.areaArray = self.heightArray * self.widthArray
                
                # peripheral circuits
                self.wlNewSwitchMatrix.CalculateArea()
                self.slSwitchMatrix.CalculateArea()
                self.mux.CalculateArea()
                self.muxDecoder.CalculateArea()
                self.multilevelSenseAmp.CalculateArea(self.widthArray)
                self.multilevelSAEncoder.CalculateArea()
                self.shiftAddWeight.CalculateArea()
                
                self.height = self.slSwitchMatrix.height + self.heightArray \
                            + self.mux.height + self.muxDecoder.height \
                            + self.multilevelSenseAmp.height + self.multilevelSAEncoder.height \
                            + self.shiftAddWeight.height
                self.width = self.widthArray + self.wlNewSwitchMatrix.width
                self.area = self.height * self.width
                
                self.usedArea = self.areaArray + self.wlNewSwitchMatrix.area + self.mux.area + self.slSwitchMatrix.area + self.shiftAddWeight.area
                assert self.usedArea <= self.area, "Error: Array area is smaller than the area of all components"
                self.emptyArea = self.area - self.usedArea
                
            elif self.param.RRAM_mode == RRAM_ANALOG:
                # crossbar memory array
                self.heightArray = self.lengthCol
                self.widthArray = self.lengthRow
                self.areaArray = self.heightArray * self.widthArray
                
                # peripheral circuits
                self.wlNewSwitchMatrix.CalculateArea()
                self.slSwitchMatrix.CalculateArea()
                self.multilevelSenseAmp.CalculateArea(self.widthArray)
                self.multilevelSAEncoder.CalculateArea()
                
                self.height = self.slSwitchMatrix.height + self.heightArray \
                            + self.multilevelSenseAmp.height + self.multilevelSAEncoder.height
                self.width = self.widthArray + self.wlNewSwitchMatrix.width
                self.area = self.height * self.width
                
                self.usedArea = self.areaArray + self.wlNewSwitchMatrix.area + self.multilevelSenseAmp.area + self.slSwitchMatrix.area
                assert self.usedArea <= self.area, "Error: Array area is smaller than the area of all components"
                self.emptyArea = self.area - self.usedArea
                

            
    def CalculateLatency(self, columnResistance):
        PM = self.param
        readLatency, writeLatency = 0,0
        readLatencyADC = 0
        
        if self.cell.MemCellType == RRAM:
            
            if PM.RRAM_mode == RRAM_DIGITAL:
            
                self.capBL = self.lengthCol * 0.2e-15/1e-6
                tau = self.capCol * self.cell.resMemCellAvg
                self.colRamp = horowitz(tau, 0, 1e20)['rampOutput']
                self.colDelay = tau * 0.2 # assume the 15~20% voltage drop is enough for sensing
                
                self.numWriteOperationPerRow = ceil(self.numCol * self.activityRowWrite) / self.numWriteCellPerOperationNeuro
                
                self.wlNewSwitchMatrix.CalculateLatency(capLoad=self.capRow2, resLoad=self.resRow,
                                                        numRead=self.numColMuxed,
                                                        numWrite=2*self.numWriteOperationPerRow*self.numRow*self.activityRowWrite)
                self.slSwitchMatrix.CalculateLatency(capLoad=self.capCol, resLoad=self.resCol,
                                                    numRead=0,
                                                    numWrite=2*self.numWriteOperationPerRow*self.numRow*self.activityRowWrite)
                
                self.mux.CalculateLatency(capLoad=0, numRead=1)
                self.muxDecoder.CalculateLatency(capLoad1=self.mux.capTgGateN*ceil(self.numCol/self.numColMuxed),
                                                capLoad2=self.mux.capTgGateP*ceil(self.numCol/self.numColMuxed),
                                                numRead=self.numColMuxed, numWrite=0)
                self.multilevelSenseAmp.CalculateLatency(columnResistance=columnResistance,
                                                         numColMuxed=1, numRead=1)
                self.multilevelSAEncoder.CalculateLatency(numRead=self.numColMuxed)
                
                self.shiftAddWeight.CalculateLatency(numRead=self.numColMuxed)
                
                # readLatency
                readLatency += self.wlNewSwitchMatrix.readLatency
                readLatency += self.colDelay
                readLatency += self.mux.readLatency
                readLatency += self.muxDecoder.readLatency
                readLatency += self.multilevelSenseAmp.readLatency + self.multilevelSAEncoder.readLatency
                readLatency += self.shiftAddWeight.readLatency
                
                # writeLatency
                writeLatency += self.totalNumWritePulse*PM.writePulseWidth
                writeLatency += max(self.wlNewSwitchMatrix.writeLatency, self.slSwitchMatrix.writeLatency)
        
            elif PM.RRAM_mode == RRAM_ANALOG:
                
                self.capBL = self.lengthCol * 0.2e-15/1e-6
                self.colRamp = 0
                tau = self.capCol * self.cell.resMemCellAvg
                self.colRamp = horowitz(tau, 0, 1e20)['rampOutput']
                self.colDelay = tau * 0.2 # assume the 15~20% voltage drop is enough for sensing
                
                self.numWriteOperationPerRow = ceil(self.numCol * self.activityRowWrite) / self.numWriteCellPerOperationNeuro
                
                self.wlNewSwitchMatrix.CalculateLatency(capLoad=self.capRow2, resLoad=self.resRow,
                                                        numRead=self.numColMuxed,
                                                        numWrite=2*self.numWriteOperationPerRow*self.numRow*self.activityRowWrite)
                self.slSwitchMatrix.CalculateLatency(capLoad=self.capCol, resLoad=self.resCol,
                    numRead=0, numWrite = 2*self.numWriteOperationPerRow*self.numRow*self.activityRowWrite)
                
                self.multilevelSenseAmp.CalculateLatency(columnResistance=columnResistance,
                                                         numColMuxed=1, numRead=1)
                self.multilevelSAEncoder.CalculateLatency(numRead=self.numColMuxed)
                                
                # readLatency
                readLatency += self.wlNewSwitchMatrix.readLatency
                readLatency += self.colDelay
                readLatency += self.multilevelSenseAmp.readLatency + self.multilevelSAEncoder.readLatency
                
                # writeLatency
                writeLatency += self.totalNumWritePulse*PM.writePulseWidth
                writeLatency += max(self.wlNewSwitchMatrix.writeLatency, self.slSwitchMatrix.writeLatency)
        
        
        self.readLatency = readLatency
        self.writeLatency = writeLatency
        
    
    def CalReadDynamicEnergyArray(self, activity, numReadCells):
        
        capBL = self.lengthCol * 0.2e-15/1e-6
        
        readDynamicEnergyArray = 0
        
        readDynamicEnergyArray += capBL * (self.param.readVoltage ** 2) * numReadCells # Selected BLs activityColWrite
        readDynamicEnergyArray += self.capRow2 * self.tech.vdd * self.tech.vdd
        readDynamicEnergyArray *= self.numRow * activity * self.param.numColMuxed
        
        return readDynamicEnergyArray
        
    
    def CalculatePower(self):
        PM = self.param
        
        self.readDynamicEnergy, self.writeDynamicEnergy, self.readDynamicEnergyArray = 0,0,0
        numReadCells = math.ceil(self.numCol / self.param.numColMuxed)
        numWriteOperationPerRow = 1
        
        if self.cell.MemCellType == RRAM:
            
            if PM.RRAM_mode == RRAM_DIGITAL:
            
                # WriteDynamicEnergy *****
                # Array: Calculate in tile level
                
                # ReadDynamicEnergy *****
                # Array
                readDynamicEnergyArray = self.CalReadDynamicEnergyArray(self.activityRowRead, numReadCells)

                # Pheripheral circuits
                readDynamicEnergy = 0
                self.wlNewSwitchMatrix.CalculatePower(numRead=self.numColMuxed,
                                                    numWrite=2*numWriteOperationPerRow*self.numRow*self.activityRowWrite,
                                                    activityRowRead=self.activityRowRead)
                self.slSwitchMatrix.CalculatePower(self.numRow, self.numCol, self.numWriteCellPerOperationNeuro,
                                                   self.activityRowRead, self.activityRowWrite)
                
                self.mux.CalculatePower(self.numColMuxed)
                self.muxDecoder.CalculatePower(self.numColMuxed, 1)
                
                self.multilevelSenseAmp.CalculatePower(numRead=self.numRow*self.activityRowRead)
                self.multilevelSAEncoder.CalculatePower(numRead=self.numColMuxed)
                
                self.shiftAddWeight.CalculatePower((self.param.synapseBit-1) * math.ceil(self.numColMuxed/self.param.synapseBit)),
                
                readDynamicEnergy = readDynamicEnergyArray
                readDynamicEnergy += self.wlNewSwitchMatrix.readDynamicEnergy
                readDynamicEnergy += self.slSwitchMatrix.readDynamicEnergy
                readDynamicEnergy += self.mux.readDynamicEnergy
                readDynamicEnergy += self.muxDecoder.readDynamicEnergy
                readDynamicEnergy += self.multilevelSenseAmp.readDynamicEnergy
                readDynamicEnergy += self.shiftAddWeight.readDynamicEnergy
                            
                # leakage
                leakage = 0
                leakage += self.wlNewSwitchMatrix.leakage
                leakage += self.slSwitchMatrix.leakage
                leakage += self.mux.leakage
                leakage += self.muxDecoder.leakage
                leakage += self.multilevelSenseAmp.leakage
                leakage += self.multilevelSAEncoder.leakage
                leakage += self.shiftAddWeight.leakage
                
            elif PM.RRAM_mode == RRAM_ANALOG:
                
                # WriteDynamicEnergy *****
                # Array: Calculate in tile level
                
                # ReadDynamicEnergy *****
                # Array
                readDynamicEnergyArray = self.CalReadDynamicEnergyArray(self.activityRowRead, numReadCells)

                # Pheripheral circuits
                readDynamicEnergy = 0
                self.wlNewSwitchMatrix.CalculatePower(numRead=self.numColMuxed,
                                                    numWrite=2*numWriteOperationPerRow*self.numRow*self.activityRowWrite,
                                                    activityRowRead=self.activityRowRead)
                self.slSwitchMatrix.CalculatePower(self.numRow, self.numCol, self.numWriteCellPerOperationNeuro,
                                                   self.activityRowRead, self.activityRowWrite)
                
                self.multilevelSenseAmp.CalculatePower(numRead=self.numRow*self.activityRowRead)
                self.multilevelSAEncoder.CalculatePower(numRead=self.numColMuxed)
                
                readDynamicEnergy = self.wlNewSwitchMatrix.readDynamicEnergy
                readDynamicEnergy += self.multilevelSenseAmp.readDynamicEnergy
                readDynamicEnergy += self.multilevelSAEncoder.readDynamicEnergy
                            
                # leakage
                leakage = 0
                leakage += self.wlNewSwitchMatrix.leakage
                leakage += self.slSwitchMatrix.leakage
                leakage += self.multilevelSenseAmp.leakage
                leakage += self.multilevelSAEncoder.leakage
        
        else:
            raise NotImplementedError
        
        self.readDynamicEnergy = readDynamicEnergy + readDynamicEnergyArray
        self.leakage = leakage
            
            
    def printInfo(self):
        def format_percentage(value, total):
            return f"{(value / total * 100):.3f}%" if total != 0 else "0.000%"

        if self.param.RRAM_mode == RRAM_DIGITAL:
        
            print(f"Summary of Array:")
            # Area, height, width
            print(f"Area: {self.area*1e12:.3f}μm2, Height: {self.height*1e6:.3f}μm, Width: {self.width*1e6:.3f}μm")
            print(f"usedArea: {self.usedArea*1e12:.3f}μm2 ({format_percentage(self.usedArea, self.area)})")
            print(f"Breakdown of Each Component:")
            print(f"\tMemCrossbar: {self.areaArray*1e12:.3f}μm2, ({format_percentage(self.areaArray, self.area)})")
            print(f"\twlNewSwitchMatrix: {self.wlNewSwitchMatrix.area*1e12:.3f}μm2, ({format_percentage(self.wlNewSwitchMatrix.area, self.area)})")
            print(f"\tSwitchMatrix: {self.slSwitchMatrix.area*1e12:.3f}μm2, ({format_percentage(self.slSwitchMatrix.area, self.area)})")
            print(f"\tMux: {self.mux.area*1e12:.3f}μm2, ({format_percentage(self.mux.area, self.area)})")
            print(f"\tMuxDecoder: {self.muxDecoder.area*1e12:.3f}μm2, ({format_percentage(self.muxDecoder.area, self.area)})")
            print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.area*1e12:.3f}μm2, ({format_percentage(self.multilevelSenseAmp.area, self.area)})")
            print(f"\tMultilevelSAEncoder: {self.multilevelSAEncoder.area*1e12:.3f}μm2, ({format_percentage(self.multilevelSAEncoder.area, self.area)})")
            print(f"\tShiftAdd: {self.shiftAddWeight.area*1e12:.3f}μm2, ({format_percentage(self.shiftAddWeight.area, self.area)})")
            print("\n")
            
            # Latency
            print(f"Read Latency: {self.readLatency*1e9:.3f}ns")
            print(f"Breakdown of Each Component:")
            print(f"\tMemCrossbar: {self.colDelay*1e9:.3f}ns, ({format_percentage(self.colDelay, self.readLatency)})")
            print(f"\twlNewSwitchMatrix: {self.wlNewSwitchMatrix.readLatency*1e9:.3f}ns, ({format_percentage(self.wlNewSwitchMatrix.readLatency, self.readLatency)})")
            print(f"\tMux: {self.mux.readLatency*1e9:.3f}ns, ({format_percentage(self.mux.readLatency, self.readLatency)})")
            print(f"\tMuxDecoder: {self.muxDecoder.readLatency*1e9:.3f}ns, ({format_percentage(self.muxDecoder.readLatency, self.readLatency)})")
            print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.readLatency*1e9:.3f}ns, ({format_percentage(self.multilevelSenseAmp.readLatency, self.readLatency)})")
            print(f"\tMultilevelSAEncoder: {self.multilevelSAEncoder.readLatency*1e9:.3f}ns, ({format_percentage(self.multilevelSAEncoder.readLatency, self.readLatency)})")
            print(f"\tShiftAdd: {self.shiftAddWeight.readLatency*1e9:.3f}ns, ({format_percentage(self.shiftAddWeight.readLatency, self.readLatency)})")
            print("\n")
            
            print(f"Write Latency: {self.writeLatency*1e9:.3f}ns")
            print("\n")

            # Energy
            print(f"Read Dynamic Energy: {self.readDynamicEnergy*1e9:.3f}nJ")
            print(f"Write Dynamic Energy: {self.writeDynamicEnergy*1e9:.3f}nJ")
            print(f"Leakage: {self.leakage*1e9:.3f}nJ")
            print(f"Breakdown of Each Component:")
            print(f"\tMemCrossbar: {self.readDynamicEnergyArray*1e9:.3f}nJ, ({format_percentage(self.readDynamicEnergyArray, self.readDynamicEnergy)})")
            print(f"\twlNewSwitchMatrix: {self.wlNewSwitchMatrix.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.wlNewSwitchMatrix.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tSwitchMtx: {self.slSwitchMatrix.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.slSwitchMatrix.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tMux: {self.mux.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.mux.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tMuxDecoder: {self.muxDecoder.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.muxDecoder.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.multilevelSenseAmp.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tMultilevelSAEncoder: {self.multilevelSAEncoder.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.multilevelSAEncoder.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tShiftAdd: {self.shiftAddWeight.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.shiftAddWeight.readDynamicEnergy, self.readDynamicEnergy)})")
            print("\n")
            
        elif self.param.RRAM_mode == RRAM_ANALOG:
            
            print(f"Summary of Array:")
            # Area, height, width
            print(f"Area: {self.area*1e12:.3f}μm2, Height: {self.height*1e6:.3f}μm, Width: {self.width*1e6:.3f}μm")
            print(f"usedArea: {self.usedArea*1e12:.3f}μm2 ({format_percentage(self.usedArea, self.area)})")
            print(f"Breakdown of Each Component:")
            print(f"\tMemCrossbar: {self.areaArray*1e12:.3f}μm2, ({format_percentage(self.areaArray, self.area)})")
            print(f"\twlNewSwitchMatrix: {self.wlNewSwitchMatrix.area*1e12:.3f}μm2, ({format_percentage(self.wlNewSwitchMatrix.area, self.area)})")
            print(f"\tSwitchMatrix: {self.slSwitchMatrix.area*1e12:.3f}μm2, ({format_percentage(self.slSwitchMatrix.area, self.area)})")
            print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.area*1e12:.3f}μm2, ({format_percentage(self.multilevelSenseAmp.area, self.area)})")
            print(f"\tMultilevelSAEncoder: {self.multilevelSAEncoder.area*1e12:.3f}μm2, ({format_percentage(self.multilevelSAEncoder.area, self.area)})")
            print("\n")
            
            # Latency
            print(f"Read Latency: {self.readLatency*1e9:.3f}ns")
            print(f"Breakdown of Each Component:")
            print(f"\tMemCrossbar: {self.colDelay*1e9:.3f}ns, ({format_percentage(self.colDelay, self.readLatency)})")
            print(f"\twlNewSwitchMatrix: {self.wlNewSwitchMatrix.readLatency*1e9:.3f}ns, ({format_percentage(self.wlNewSwitchMatrix.readLatency, self.readLatency)})")
            print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.readLatency*1e9:.3f}ns, ({format_percentage(self.multilevelSenseAmp.readLatency, self.readLatency)})")
            print(f"\tMultilevelSAEncoder: {self.multilevelSAEncoder.readLatency*1e9:.3f}ns, ({format_percentage(self.multilevelSAEncoder.readLatency, self.readLatency)})")
            print("\n")
            
            # Energy
            print(f"Read Dynamic Energy: {self.readDynamicEnergy*1e9:.3f}nJ")
            print(f"Write Dynamic Energy: {self.writeDynamicEnergy*1e9:.3f}nJ")
            print(f"Leakage: {self.leakage*1e9:.3f}nJ")
            print(f"Breakdown of Each Component:")
            print(f"\tMemCrossbar: {self.readDynamicEnergyArray*1e9:.3f}nJ, ({format_percentage(self.readDynamicEnergyArray, self.readDynamicEnergy)})")
            print(f"\twlNewSwitchMatrix: {self.wlNewSwitchMatrix.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.wlNewSwitchMatrix.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tSwitchMtx: {self.slSwitchMatrix.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.slSwitchMatrix.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tMultilevelSenseAmp: {self.multilevelSenseAmp.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.multilevelSenseAmp.readDynamicEnergy, self.readDynamicEnergy)})")
            print(f"\tMultilevelSAEncoder: {self.multilevelSAEncoder.readDynamicEnergy*1e9:.3f}nJ, ({format_percentage(self.multilevelSAEncoder.readDynamicEnergy, self.readDynamicEnergy)})")
            print("\n")
            
        
if __name__ == '__main__':
    
    from Gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(22, FormulaBindings.DeviceRoadmap.LSTP, 
                    FormulaBindings.TransistorType.conventional)
    
    from Param import Param
    param = Param()
    
    param.memcelltype = RRAM
    param.RRAM_mode = RRAM_DIGITAL
    param.numRowSubArray = 128
    param.numColSubArray = 128
    param.synapseBit = 8 
    param.numColMuxed = 8
    param.cellBit = 1
    param.technode = 22
    param.temp = 300
    param.levelOutput = 32
    
    gate_params = compute_gate_params(param, tech)
    
    # test
    array = Array(128, 128, param, tech, gate_params)
    
    array.CalculateArea()
    colResis = [1e3] * 128
    array.totalNumWritePulse = 15
    array.CalculateLatency(colResis)
    array.CalculatePower()
    array.printInfo()
    
    
    