from from_neurosim.build import FormulaBindings
from constant import *

param = FormulaBindings.Param()

def sqrt(x):
    return x ** 0.5

class MemCell:
    def __init__(self, param):
        
        self.MemCellType = param.memcelltype
        self.featureSize = param.featuresize
        
        if self.MemCellType == SRAM:
            self.widthInFeatureSize = param.widthInFeatureSizeSRAM
            self.heightInFeatureSize = param.heightInFeatureSizeSRAM
            raise NotImplementedError
        
        elif self.MemCellType == RRAM:
            self.accessType = param.accesstype
            
            self.widthInFeatureSize = param.widthInFeatureSize1T1R
            self.heightInFeatureSize = param.heightInFeatureSize1T1R
            
            self.resistanceOn = param.resistanceOn # Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
            self.resistanceOff = param.resistanceOff # Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
            self.resistanceAvg = (param.resistanceOn + param.resistanceOff) / 2 # Average resistance (for energy estimation)
            
            self.readVoltage = param.readVoltage  # On-chip read voltage for memory cell
            self.readPulseWidth = param.readPulseWidth
            self.accessVoltage = param.accessVoltage # Gate voltage for the transistor in 1T1R
            self.resistanceAccess = param.resistanceAccess
            self.maxNumLevelLTP = param.maxNumLevelLTP # Maximum number of conductance states during LTP or weight increase
            self.maxNumLevelLTD = param.maxNumLevelLTD # LTD or weight decrease
            self.writeVoltageLTP = param.writeVoltage
            self.writeVoltageLTD = param.writeVoltage
            self.writeVoltage = sqrt(self.writeVoltageLTP*self.writeVoltageLTP + self.writeVoltageLTD*self.writeVoltageLTD)   # Use an average value of write voltage for NeuroSim
            self.writePulseWidthLTP = param.writePulseWidth
            self.writePulseWidthLTD = param.writePulseWidth
            self.writePulseWidth = (self.writePulseWidthLTP + self.writePulseWidthLTD) / 2
            self.nonlinearIV = param.nonlinearIV
            self.nonlinearity = param.nonlinearity
            
            self.minSenseVoltage = param.minSenseVoltage
            
            
            self.resCellAccess = param.resistanceOn * IR_DROP_TOLERANCE
            self.resMemCellOn = self.resCellAccess + param.resistanceOn # calculate single memory cell resistance_ON
            self.resMemCellOff = self.resCellAccess + param.resistanceOff # calculate single memory cell resistance_OFF
            self.resMemCellAvg = self.resCellAccess + self.resistanceAvg # calculate single memory cell resistance_AVG
            
            
            
            
            
            
            
            
            