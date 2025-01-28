import math
from constant import *
from gate_calculator import horowitz, CalculateTransconductance
from from_neurosim.build import FormulaBindings

class Adder:
    def __init__(self, numBit, numAdder, param, tech, gate_params):
        self.numBit = numBit
        self.numAdder = numAdder
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.hNand = self.gate_params["hNand"]
        self.wNand = self.gate_params["wNand"]
        self.resNandN = self.gate_params["resNandN"]
        self.resNandP = self.gate_params["resNandP"]
        self.capNandInput = self.gate_params["capNandInput"]
        self.capNandOutput = self.gate_params["capNandOutput"]
        self.widthNandN = self.gate_params["widthNandN"]
        self.widthNandP = self.gate_params["widthNandP"]
        
    def CalculateArea(self):
        hAdder = self.hNand
        wAdder = self.wNand * 9 * self.numBit
        self.width = wAdder * self.numAdder
        self.height = hAdder
        self.area = self.width * self.height
    
    def CalculateLatency(self, _capLoad, _numRead):
        
        rampInput = 1e20
        ramp = {}
        ramp[0] = rampInput
        
        # Calibration data pattern is A=1111111..., B=1000000... and Cin=1
        resPullDown = self.resNandN * 2
        resPullUp = self.resNandP
        trDown = resPullDown * (self.capNandOutput + self.capNandInput * 3)
        trUp = resPullUp * (self.capNandOutput + self.capNandInput * 2)
        betaDown = 1 / (resPullDown * CalculateTransconductance(self.widthNandN, NMOS, self.tech))
        betaUp = 1 / (resPullUp * CalculateTransconductance(self.widthNandP, PMOS, self.tech))
        # 1st
        readLatency = horowitz(trDown, betaDown, ramp[0])['result']
        ramp[1] = horowitz(trDown, betaDown, ramp[0])['rampOutput']
        
        # 2nd
        readLatency += horowitz(trUp, betaUp, ramp[1])['result']
        ramp[2] = horowitz(trUp, betaUp, ramp[1])['rampOutput']
        
        # 3rd
        readLatencyIntermediate = horowitz(trDown, betaDown, ramp[2])['result']
        ramp[3] = horowitz(trDown, betaDown, ramp[2])['rampOutput']
        
        # 4th
        readLatencyIntermediate += horowitz(trUp, betaUp, ramp[3])['result']
        ramp[4] = horowitz(trUp, betaUp, ramp[3])['rampOutput']
        
        if self.numBit > 2:
            readLatency += readLatencyIntermediate * (self.numBit - 2)
        
        # 5th
        readLatency += horowitz(trDown, betaDown, ramp[4])['result']
        ramp[5] = horowitz(trDown, betaDown, ramp[4])['rampOutput']
        
        # 6th
        readLatency += horowitz(trUp, betaUp, ramp[5])['result']
        ramp[6] = horowitz(trUp, betaUp, ramp[5])['rampOutput']
        
        # 7th
        readLatency += horowitz(trDown, betaDown, ramp[6])['result']
        ramp[7] = horowitz(trDown, betaDown, ramp[6])['rampOutput']
        
        readLatency *= _numRead
        self.rampOutput = ramp[7]
        
        self.readLatency = readLatency
        
    def CalculatePower(self, _numRead, _numAdderPerOperation):
        self.leakage = 0
        self.readDynamicEnergy = 0
        
        # 1st stage
        readDynamicEnergy = (self.capNandInput * 6) * self.tech.vdd * self.tech.vdd # Input of 1 and 2 and Cin
        readDynamicEnergy += (self.capNandOutput * 2) * self.tech.vdd * self.tech.vdd # Output of S[0] and 5
        # Second and later stages
        readDynamicEnergy += (self.capNandInput * 7) * self.tech.vdd * self.tech.vdd * (self.numBit-1)
        readDynamicEnergy += (self.capNandOutput * 3) * self.tech.vdd * self.tech.vdd * (self.numBit-1)
        
        # Hidden transition
		# First stage
        readDynamicEnergy += (self.capNandOutput + self.capNandInput) * self.tech.vdd * self.tech.vdd * 2 # #2 and #3
        readDynamicEnergy += (self.capNandOutput + self.capNandInput * 2) * self.tech.vdd * self.tech.vdd;	# #4
        readDynamicEnergy += (self.capNandOutput + self.capNandInput * 3) * self.tech.vdd * self.tech.vdd;	# #5
        readDynamicEnergy += (self.capNandOutput + self.capNandInput) * self.tech.vdd * self.tech.vdd;		# #6
        # Second and later stages
        readDynamicEnergy += (self.capNandOutput + self.capNandInput * 3) * self.tech.vdd * self.tech.vdd * (self.numBit-1);	# # 1
        readDynamicEnergy += (self.capNandOutput + self.capNandInput) * self.tech.vdd * self.tech.vdd * (self.numBit-1);		# # 3
        readDynamicEnergy += (self.capNandOutput + self.capNandInput) * self.tech.vdd * self.tech.vdd * 2 * (self.numBit-1);	# #6 and #7
	
        readDynamicEnergy *= min(_numAdderPerOperation, self.numAdder) * _numRead
        
        self.readDynamicEnergy = readDynamicEnergy
        
if __name__ == "__main__":
    
    from gate_calculator import compute_gate_params
    
    tech = FormulaBindings.Technology()
    tech.Initialize(22, FormulaBindings.DeviceRoadmap.LSTP, FormulaBindings.TransistorType.conventional)
    param = FormulaBindings.Param()
    param.memcelltype = RRAM
    gate_params = compute_gate_params(param, tech)
    
    adder = Adder(4, 64, param, tech, gate_params)
    adder.CalculateArea()
    adder.CalculateLatency(45*0.2e-15/1e-6, 128)
    adder.CalculatePower(128, 64)
    
    print(adder.area, adder.readLatency, adder.readDynamicEnergy)