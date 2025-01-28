from from_neurosim.build import FormulaBindings
from constant import *

param = FormulaBindings.Param()

class MemCell:
    def __init__(self, param):
        self.MemCellType = param.memcelltype
        
        if self.MemCellType == SRAM:
            self.widthInFeatureSize = param.widthInFeatureSizeSRAM
            self.heightInFeatureSize = param.heightInFeatureSizeSRAM
        
        elif self.MemCellType == RRAM:
            self.accessType = param.accesstype
            
            self.widthInFeatureSize = param.widthInFeatureSize1T1R
            self.heightInFeatureSize = param.heightInFeatureSize1T1R
            
            self.resCellAccess = param.resistanceOn * IR_DROP_TOLERANCE
            self.resMemCellOn = self.resCellAccess + param.resistanceOn
            self.resMemCellOff = self.resCellAccess + param.resistanceOff
            
        self.resMemCellAvg = -1