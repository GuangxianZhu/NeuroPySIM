import math
from constant import *
from gate_calculator import horowitz, CalculateTransconductance
from from_neurosim.build import FormulaBindings

from Adder import Adder

def ceil(x):
    return int(math.ceil(x))

def log2(x):
    return math.log(x, 2)

class AdderTree:
    def __init__(self, numSubcoreRow, numAdderBit, numAdderTree,
                 param, tech, gate_params):
        self.numSubcoreRow = numSubcoreRow
        self.numAdderBit = numAdderBit
        self.numAdderTree = numAdderTree
        
        self.numStage = ceil(log2(numSubcoreRow)) # # of stage of the adder tree, used for CalculateLatency ...
        
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.adder = None
        
    def CalculateArea(self):
        self.numAdderEachStage = 0
        self.numBitEachStage = self.numAdderBit
        self.numAdderEachTree = 0
        I = ceil(log2(self.numSubcoreRow))
        J = self.numSubcoreRow
        """
        while (i != 0) {  // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			numAdderEachTree += numBitEachStage*numAdderEachStage;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
		}
        """
        for i in range(I):
            numAdderEachStage = ceil(J/2)
            self.numAdderEachTree += self.numBitEachStage * numAdderEachStage
            self.numBitEachStage += 1
            J = ceil(J/2)
        
        self.adder = Adder(self.numAdderEachTree, self.numAdderTree, 
                           self.param, self.tech, self.gate_params)
        
        self.adder.CalculateArea()
        self.height = self.adder.height
        self.width = self.adder.width
        self.area = self.adder.area
        
        self.adder = None
        
    def CalculateLatency(self, numRead, numUnitAdd:int, capLoad):
        
        if numUnitAdd ==0:
            I = ceil(log2(self.numSubcoreRow))
            J = self.numSubcoreRow
            
        else:
            I = ceil(log2(numUnitAdd))
            J = numUnitAdd
    
        self.numAdderEachStage = ceil(J/2)
        self.adder = Adder(self.numBitEachStage, self.numAdderEachStage, 
                           self.param, self.tech, self.gate_params)
        self.adder.CalculateLatency(capLoad, numRead)
        self.readLatency = self.adder.readLatency
        self.numBitEachStage += 1
        J = ceil(J/2)
        I -= 1
        
        """
        if (i>0) {
			while (i != 0) {   // calculate the total # of full adder in each Adder Tree
				numAdderEachStage = ceil(j/2);
				adder.Initialize(2, numAdderEachStage, clkFreq);   
				adder.CalculateLatency(1e20, _capLoad, 1);
				readLatency += adder.readLatency;
				numBitEachStage += 1;
				j = ceil(j/2);
				i -= 1;
				
				adder.initialized = false;
			}
		}
        """
        
        while I > 0:
            numAdderEachStage = ceil(J/2)
            self.adder = Adder(self.numBitEachStage, numAdderEachStage, 
                               self.param, self.tech, self.gate_params)
            self.adder.CalculateLatency(capLoad, numRead)
            self.readLatency += self.adder.readLatency
            self.numBitEachStage += 1
            J = ceil(J/2)
            I -= 1
            self.adder = None
            
        self.readLatency *= numRead
    
    def CalculatePower(self, numRead, numUnitAdd:int):
        
        self.readDynamicEnergy = 0
        self.leakage = 0
        
        if numUnitAdd ==0:
            I = ceil(log2(self.numSubcoreRow))
            J = self.numSubcoreRow
            
        else:
            I = ceil(log2(numUnitAdd))
            J = numUnitAdd
    
        self.numAdderEachStage = 0
        self.numBitEachStage = self.numAdderBit
        self.numAdderEachTree = 0
        
        for i in range(I):
            self.numAdderEachStage = ceil(J/2)
            self.adder = Adder(self.numBitEachStage, self.numAdderEachStage, 
                               self.param, self.tech, self.gate_params)
            self.adder.CalculatePower(1, self.numAdderEachStage)
            self.readDynamicEnergy += self.adder.readDynamicEnergy
            self.leakage += self.adder.leakage
            self.numBitEachStage += 1
            J = ceil(J/2)
        
        self.readDynamicEnergy *= self.numAdderTree
        self.readDynamicEnergy *= numRead
        self.leakage *= self.numAdderTree
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    