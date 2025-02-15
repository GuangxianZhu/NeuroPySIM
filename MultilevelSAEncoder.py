import math
from constant import *
from from_neurosim.build import FormulaBindings
from MemCell import MemCell
from Gate_calculator import CalculateGateArea, CalculateOnResistance, CalculateGateCapacitance, CalculateTransconductance, horowitz, CalculateGateLeakage

def ceil(x):
    return math.ceil(x)

def log2(x):
    return math.log2(x)


class MultilevelSAEncoder:
    def __init__(self, numlevel, numEncoder, param, tech, gate_params):
        
        self.numlevel = numlevel    # number of levels from MultilevelSA
        self.numEncoder = numEncoder # number of encoder needed
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.numInput = ceil(numlevel/2)    # num of NAND gate in encoder
        self.numGate = ceil(log2(numlevel)) # num of NAND gate in encoder
        
    def CalculateArea(self):
        hNand, wNand = self.gate_params["hNand"], self.gate_params["wNand"]
        hInv, wInv = self.gate_params["hInv"], self.gate_params["wInv"]
        # large Nand
        widthNandN = self.gate_params["widthNandN"]
        widthNandP = self.gate_params["widthNandP"]
        hNandLg = CalculateGateArea(NAND, self.numInput, widthNandN, widthNandP, MAX_TRANSISTOR_HEIGHT * self.tech.featureSize, self.tech)['height']
        wNandLg = CalculateGateArea(NAND, self.numInput, widthNandN, widthNandP, MAX_TRANSISTOR_HEIGHT * self.tech.featureSize, self.tech)['width']
        
        wEncoder = 2*wInv + wNand + wNandLg
        hEncoder = max( (self.numlevel-1)*hInv, (self.numlevel-1)*hNand )
        
        self.height = hEncoder * self.numEncoder
        self.width = wEncoder
        self.area = self.height * self.width
        
        # large Nand cap
        self.capNandLgInput = CalculateGateCapacitance(NAND, self.numInput, widthNandN, widthNandP, hNandLg, self.tech)['capInput']
        self.capNandLgOutput = CalculateGateCapacitance(NAND, self.numInput, widthNandN, widthNandP, hNandLg, self.tech)['capOutput']
    
    """
void MultilevelSAEncoder::CalculateLatency(double _rampInput, double numRead){
	if (!initialized) {
		cout << "[MultilevelSAEncoder] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		rampInput = _rampInput;
		double tr;		/* time constant */
		double gm;		/* transconductance */
		double beta;	/* for horowitz calculation */
		double resPullUp, resPullDown;
		double readLatencyIntermediate = 0;
		double ramp[10];
		
		ramp[0] = rampInput;

		// 1st INV to NAND2
		resPullDown = CalculateOnResistance(widthInvN, NMOS, inputParameter.temperature, tech) * 2;
		tr = resPullDown * (capInvOutput + capNandInput * 2);
		gm = CalculateTransconductance(widthNandN, NMOS, tech);
		beta = 1 / (resPullDown * gm);
		readLatency += horowitz(tr, beta, ramp[0], &ramp[1]);
		
		// 2nd NAND2 to Large NAND
		resPullUp = CalculateOnResistance(widthNandP, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * (capNandOutput + capNandLgInput * numInput);
		gm = CalculateTransconductance(widthNandP, PMOS, tech);
		beta = 1 / (resPullUp * gm);
		readLatency += horowitz(tr, beta, ramp[1], &ramp[2]);
		
		// 3rd large NAND to INV
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;
		tr = resPullDown * (capNandLgOutput + capInvInput);
		gm = CalculateTransconductance(widthNandN, NMOS, tech);
		beta = 1 / (resPullDown * gm);
		readLatencyIntermediate += horowitz(tr, beta, ramp[2], &ramp[3]);

		// 4th INV
		resPullUp = CalculateOnResistance(widthInvP, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * capInvOutput;
		gm = CalculateTransconductance(widthNandP, PMOS, tech);
		beta = 1 / (resPullUp * gm);
		readLatencyIntermediate += horowitz(tr, beta, ramp[3], &ramp[4]);
		
		readLatency *= numRead;
		rampOutput = ramp[4];
	}
}
    """
    
    def CalculateLatency(self, numRead):
        self.rampInput = 1e20
        self.ramp = [0]*10
        self.ramp[0] = self.rampInput
        
        
        widthInvN = self.gate_params["widthInvN"]
        capInvOutput = self.gate_params["capInvOutput"]
        capNandInput = self.gate_params["capNandInput"]
        capNandOutput = self.gate_params["capNandOutput"]
        capNandLgInput = self.capNandLgInput
        capNandLgOutput = self.capNandLgOutput
        
        GP = self.gate_params
        
        
        # 1st INV to NAND2
        resPullDown = CalculateOnResistance(widthInvN, NMOS, self.param.temp, self.tech) * 2
        tr = resPullDown * (capInvOutput + capNandInput * 2)
        gm = CalculateTransconductance(GP["widthNandN"], NMOS, self.tech)
        beta = 1 / (resPullDown * gm)
        self.readLatency = horowitz(tr, beta, self.ramp[0])['result']
        self.ramp[1] = horowitz(tr, beta, self.ramp[0])['rampOutput']
        
        # 2nd NAND2 to Large NAND
        resPullUp = CalculateOnResistance(GP["widthNandP"], PMOS, self.param.temp, self.tech)
        tr = resPullUp * (capNandOutput + capNandLgInput * self.numInput)
        gm = CalculateTransconductance(GP["widthNandP"], PMOS, self.tech)
        beta = 1 / (resPullUp * gm)
        self.readLatency += horowitz(tr, beta, self.ramp[1])['result']
        self.ramp[2] = horowitz(tr, beta, self.ramp[1])['rampOutput']
        
        # 3rd large NAND to INV
        resPullDown = CalculateOnResistance(GP["widthNandN"], NMOS, self.param.temp, self.tech) * 2
        tr = resPullDown * (capNandLgOutput + capInvOutput)
        gm = CalculateTransconductance(GP["widthNandN"], NMOS, self.tech)
        beta = 1 / (resPullDown * gm)
        readLatencyIntermediate = horowitz(tr, beta, self.ramp[2])['result']
        self.ramp[3] = horowitz(tr, beta, self.ramp[2])['rampOutput']
        
        # 4th INV
        resPullUp = CalculateOnResistance(GP["widthInvP"], PMOS, self.param.temp, self.tech)
        tr = resPullUp * capInvOutput
        gm = CalculateTransconductance(GP["widthNandP"], PMOS, self.tech)
        beta = 1 / (resPullUp * gm)
        readLatencyIntermediate += horowitz(tr, beta, self.ramp[3])['result']
        
        self.readLatency *= numRead
        self.rampOutput = self.ramp[4]
        
    def CalculatePower(self, numRead):
        """
        readDynamicEnergy = 0;
		leakage = 0;

		leakage = CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * (numLevel+numGate) * numEncoder
		          + CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, inputParameter.temperature, tech) * tech.vdd * (numLevel+numGate) * numEncoder
				  + CalculateGateLeakage(NAND, numInput, widthNandN, widthNandP, inputParameter.temperature, tech) * tech.vdd * numGate * numEncoder;
		
		readDynamicEnergy += (capInvInput + capInvOutput) * tech.vdd * tech.vdd * (numLevel+numGate) * numEncoder;
		readDynamicEnergy += (capNandInput + capNandOutput) * tech.vdd * tech.vdd * (numLevel+numGate) * numEncoder;
		readDynamicEnergy += (capNandLgInput + capNandLgOutput) * tech.vdd * tech.vdd * numGate * numEncoder;
		readDynamicEnergy *= numRead;
        """
        
        self.leakage = 0
        self.readDynamicEnergy = 0
        
        GP = self.gate_params
        
        self.leakage = CalculateGateLeakage(INV, 1, GP["widthInvN"], GP["widthInvP"], self.param.temp, self.tech) * self.tech.vdd * (self.numlevel+self.numGate) * self.numEncoder
        self.leakage += CalculateGateLeakage(NAND, 2, GP["widthNandN"], GP["widthNandP"], self.param.temp, self.tech) * self.tech.vdd * (self.numlevel+self.numGate) * self.numEncoder
        self.leakage += CalculateGateLeakage(NAND, self.numInput, GP["widthNandN"], GP["widthNandP"], self.param.temp, self.tech) * self.tech.vdd * self.numGate * self.numEncoder
        
        self.readDynamicEnergy += (GP["capInvInput"] + GP["capInvOutput"]) * self.tech.vdd * self.tech.vdd * (self.numlevel+self.numGate) * self.numEncoder
        self.readDynamicEnergy += (GP["capNandInput"] + GP["capNandOutput"]) * self.tech.vdd * self.tech.vdd * (self.numlevel+self.numGate) * self.numEncoder
        self.readDynamicEnergy += (self.capNandLgInput + self.capNandLgOutput) * self.tech.vdd * self.tech.vdd * self.numGate * self.numEncoder
        self.readDynamicEnergy *= numRead