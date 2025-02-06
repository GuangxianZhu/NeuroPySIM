from from_neurosim.build import FormulaBindings
from constant import *

class Param(FormulaBindings.Param):
    def __setattr__(self, name, value):
        
        # First call the parent class's __setattr__ method to complete the assignment
        super().__setattr__(name, value)
        
        # When set synapseBit, update numColMuxed
        if name == 'synapseBit':
            super().__setattr__('numColMuxed', value)
            super().__setattr__('numColPerSynapse', value)
            
        # resistanceOn and maxConductance
        if name == "resistanceOn":
            try:
                super().__setattr__("maxConductance", 1.0/value)
                super().__setattr__("resistanceAccess", 1.0/value*IR_DROP_TOLERANCE)
            except Exception as e:
                print(f"Error updating maxConductance: {e}")
        
        if name == "resistanceOff":
            try:
                super().__setattr__("minConductance", 1.0/value)
            except Exception as e:
                print(f"Error updating minConductance: {e}")
            
            
            
            
            
            
            
            
            
            