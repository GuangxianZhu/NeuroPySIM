
import math
from constant import *
from from_neurosim.build import FormulaBindings
from MemCell import MemCell

class CurrentSenseAmp:
    def __init__(self, numCol, param, tech, gate_params):
        self.numCol = numCol
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
    
    def CalculateUnitArea(self):
        self.areaUnit = (self.gate_params["hNmos"]*self.gate_params["wNmos"])*48 + (self.gate_params["hPmos"]*self.gate_params["wPmos"])*40
    
    def CalculateArea(self, widthArray):
        self.area = self.areaUnit * self.numCol
        self.width = widthArray * self.numCol
        self.height = self.areaUnit/widthArray
    
    def CalculateLatency(self, _columnResistance:list, _numColMuxed, _numRead):
        Group = self.numCol/_numColMuxed
        LatencyCol = 1e-9
        self.readLatency = LatencyCol * _numRead
    
    def CalculatePower(self, _columnResistance:list, _numRead):
        readDynamicEnergy=0
        for j in len(_columnResistance):
            T_Col = 1e-9
            columnRes = _columnResistance[j] * 0.5/self.param.readVoltage
            P_Col = 1.0*1e-6 + 0.061085*math.exp(-2.303*math.log10(columnRes))
            readDynamicEnergy += T_Col*P_Col
        readDynamicEnergy *= _numRead
        self.readDynamicEnergy = readDynamicEnergy


class MultilevelSenseAmp:
    def __init__(self, numCol, levelOutput, param, tech, gate_params):
        
        self.numCol = numCol
        self.levelOutput = levelOutput
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.cell = MemCell(self.param)
        self.numReadCellPerOperationNeuro = self.param.numColSubArray
        
        self.Rref = []
        for i in range(self.levelOutput):
            if self.cell.MemCellType == SRAM:
                raise NotImplementedError
            
            elif self.cell.MemCellType == RRAM:
                if param.operationmode == 2: # conventionalParallel (Use several multi-bit RRAM as one synapse)
                    R_start = param.resistanceOn / param.numRowParallel
                    G_start = 1/R_start
                    G_index = 0 - G_start
                    
                    R_this = 1/ (G_start + i*G_index/param.levelOutput)
                    self.Rref.append(R_this)
                else:
                    R_start = param.resistanceOn
                    R_end = param.resistanceOff
                    G_start = 1/R_start
                    G_end = 1/R_end
                    G_index = (G_end - G_start)
                    R_this = 1/(G_start + i*G_index/param.levelOutput)
                    self.Rref.append(R_this)
        
        self.currentSenseAmp = CurrentSenseAmp(numCol=(self.levelOutput-1)*self.numCol, param=param, tech=tech, gate_params=gate_params)
        self.gatecap_senseamp_P = self.gate_params["capPmosInput"]
        self.junctioncap_senseamp_P = self.gate_params["capPmosOutput"]
        self.gatecap_senseamp_N = self.gate_params["capNmosInput"]
        self.junctioncap_senseamp_N = self.gate_params["capNmosOutput"]
        
        self.initialized = True
        
    def CalculateArea(self, widthArray):
        area = 0
        area += ((self.gate_params["hNmos"]*self.gate_params["wNmos"])*9)*(self.levelOutput-1)*self.numCol
        area += self.param.arrayheight * self.param.arraywidthunit *(self.param.levelOutput-1)*self.numCol/self.param.dumcolshared
        self.width = widthArray
        self.height = area / self.width
        self.area = area
            
            
    def CalculateLatency(self, currentMode, numColMuxed, numRead):
        latency_minrefcon = self.ColumnLatency_Table(self.Rref[-1])
        latency_maxrefcon = self.ColumnLatency_Table(self.Rref[1])
        LatencyCol = max(latency_minrefcon, latency_maxrefcon)
        if currentMode: 
            readLatency = LatencyCol*numColMuxed
        else:
            readLatency = (1e-9)*numColMuxed
        readLatency *= numRead
        self.readLatency = readLatency
        
            
    def CalculatePower(self, _numRead):
        # my model
        colCapPerCol = 1e-14
        senseAmpCapPerCol = 5e-15
        refOverheadPerLevel = 1e-13
        coefficient = 1.0
        
        E_colSwitch = colCapPerCol * self.numCol * (self.levelOutput - 1) * (self.param.readVoltage ** 2)
        E_senseAmp = senseAmpCapPerCol * self.numCol * (self.levelOutput - 1) * (self.param.readVoltage ** 2)
        E_ref = refOverheadPerLevel * (self.levelOutput - 1)
        E_single_read = E_colSwitch + E_senseAmp + E_ref
        E_single_read *= coefficient
        E_total = E_single_read * _numRead
        self.readDynamicEnergy = E_total
    
    def ColumnLatency_Table(self, Res):
        """
        in cpp (45nm):
        if (Res<695) {
						latency = -2.66629E-10* log(x) + 3.07867E-09 ;
					}
					else if ((Res>=695) && (Res<4832)) {
						latency = -6.27624E-11* log(x) + 1.78932E-09; 
					}
					else {
						latency = 1.43578E-14* x  + 1.46251E-09 ; 
					}
					refcap=9.21E-15;
					dC=0.9E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
        """
        x = Res
        if Res<695:
            latency = -2.66629E-10* math.log(x) + 3.07867E-09
        elif (Res>=695) and (Res<4832):
            latency = -6.27624E-11* math.log(x) + 1.78932E-09
        else:
            latency = 1.43578E-14* x  + 1.46251E-09
        return latency