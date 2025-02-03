import math
from constant import *
from MemCell import MemCell
from from_neurosim.build import FormulaBindings
from gate_calculator import CalculateGateLeakage, CalculateGateArea, CalculateGateCapacitance, CalculateOnResistance, horowitz, CalculateTransconductance

class WLNewDecoderDriver:
    def __init__(self, numWLRow, param, tech, gate_params: dict):
        self.numWLRow = numWLRow
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
    def CalculateArea(self):
        hNand, wNand = self.gate_params['hNand'], self.gate_params['wNand']
        hInv, wInv = self.gate_params['hInv'], self.gate_params['wInv']
        hTg, wTg = self.gate_params['hTg'], self.gate_params['wTg']
        
        self.height = max( max(hInv, hNand), hTg) * self.numWLRow
        self.width = 3 * wNand + wInv + 2 * wTg
        self.area = self.height * self.width
        
    def CalculateLatency(self, capLoad1, capLoad2, resLoad, 
                         numRead, numWrite):
        """
        // 1st stage: NAND2
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;      // pulldown 2 NMOS in series
		trnand = resPullDown * (capNandOutput + capInvInput);          // connect to INV
		gmnand = CalculateTransconductance(widthNandN, NMOS, tech);  
		betanand = 1 / (resPullDown * gmnand);
		readLatency += horowitz(trnand, betanand, rampInput, NULL);
		writeLatency += horowitz(trnand, betanand, rampInput, NULL);
		
		// 2ed stage: INV
		resPullUp = CalculateOnResistance(widthInvP, PMOS, inputParameter.temperature, tech);
		trinv = resPullUp * (capInvOutput + 2 * capNandInput);       // connect to 2 NAND2 gate
		gminv = CalculateTransconductance(widthNandP, PMOS, tech);  
		betainv = 1 / (resPullUp * gminv);
		readLatency += horowitz(trinv, betainv, rampInput, NULL);
		writeLatency += horowitz(trinv, betainv, rampInput, NULL);
		
		// 3ed stage: NAND2
		resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;      
		trnand = resPullDown * (capNandOutput + capTgGateP + capTgGateN);      // connect to 2 transmission gates
		gmnand = CalculateTransconductance(widthNandN, NMOS, tech);  
		betanand = 1 / (resPullDown * gmnand);
		readLatency += horowitz(trnand, betanand, rampInput, NULL);
		writeLatency += horowitz(trnand, betanand, rampInput, NULL);
		
		// 4th stage: TG
		capOutput = 2 * capTgDrain;      
		trtg = resTg * (capOutput + capLoad) + resLoad * capLoad / 2;        // elmore delay model
		readLatency += horowitz(trtg, 0, 1e20, &rampOutput);	// get from chargeLatency in the original SubArray.cpp
		writeLatency += horowitz(trtg, 0, 1e20, &rampOutput);
		
		readLatency *= numRead;
		writeLatency *= numWrite;
        """
        
        DT = self.gate_params
        
        rampInput = 1e20
        resPullDown = CalculateOnResistance(DT['widthNandN'], NMOS, self.param.temp, self.tech) * 2
        trnand = resPullDown * (DT['capNandOutput'] + DT['capInvInput']) # connect to INV
        gmnand = CalculateTransconductance(DT['widthNandN'], NMOS, self.tech)
        betanand = 1 / (resPullDown * gmnand)
        self.readLatency = horowitz(trnand, betanand, rampInput)['result']
        self.writeLatency = self.readLatency
        
        resPullUp = CalculateOnResistance(DT['widthInvP'], PMOS, self.param.temp, self.tech)
        trinv = resPullUp * (DT['capInvOutput'] + 2 * DT['capNandInput'])
        gminv = CalculateTransconductance(DT['widthNandP'], PMOS, self.tech)
        betainv = 1 / (resPullUp * gminv)
        self.readLatency += horowitz(trinv, betainv, rampInput)['result']
        self.writeLatency += self.readLatency
        
        resPullDown = CalculateOnResistance(DT['widthNandN'], NMOS, self.param.temp, self.tech) * 2
        trnand = resPullDown * (DT['capNandOutput'] + DT['capTgGateP'] + DT['capTgGateN'])
        gmnand = CalculateTransconductance(DT['widthNandN'], NMOS, self.tech)
        betanand = 1 / (resPullDown * gmnand)
        self.readLatency += horowitz(trnand, betanand, rampInput)['result']
        self.writeLatency += self.readLatency
        
        capOutput = 2 * DT['capTgDrain']
        trtg = DT['resTg'] * (capOutput + capLoad1) + resLoad * capLoad1 / 2
        self.readLatency += horowitz(trtg, 0, 1e20)['result']
        self.writeLatency += self.readLatency
        
        self.readLatency *= numRead
        self.writeLatency *= numWrite
        
    def CalculatePower(self, numRead, numWrite):
        """
        // Leakage power
		// NAND2
		leakage += CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, inputParameter.temperature, tech) * tech.vdd * numWLRow * 2;
		// INV
		leakage += CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * numWLRow * 2;
		// assuming no leakge in TG
		
		// Read dynamic energy (only one row activated)
		readDynamicEnergy += capNandInput * tech.vdd * tech.vdd;                           // NAND2 input charging ( 0 to 1 )
		readDynamicEnergy += (capInvOutput + capTgGateN) * tech.vdd * tech.vdd;            // INV output charging ( 0 to 1 )
		readDynamicEnergy += (capNandOutput + capTgGateN + capTgGateP) * tech.vdd * tech.vdd;               // NAND2 output charging ( 0 to 1 )
		readDynamicEnergy += capTgDrain * cell.readVoltage * cell.readVoltage;         // TG gate energy
		readDynamicEnergy *= numRead;          // multiply reading operation times
		
		// Write dynamic energy (only one row activated)
		writeDynamicEnergy += capNandInput * tech.vdd * tech.vdd;
		writeDynamicEnergy += (capInvOutput + capTgGateN) * tech.vdd * tech.vdd;
		writeDynamicEnergy += (capNandOutput + capTgGateN + capTgGateP) * tech.vdd * tech.vdd;     
		writeDynamicEnergy += capTgDrain * cell.writeVoltage * cell.writeVoltage;    
		writeDynamicEnergy *= numWrite;
        """
        self.leakage, self.readDynamicEnergy, self.writeDynamicEnergy = 0,0,0
        
        DT = self.gate_params
        
        # Leakage power
        self.leakage += CalculateGateLeakage(NAND, 2, DT['widthNandN'], DT['widthNandP'], self.param.temp, self.tech) * self.tech.vdd * self.numWLRow * 2
        self.leakage += CalculateGateLeakage(INV, 1, DT['widthInvN'], DT['widthInvP'], self.param.temp, self.tech) * self.tech.vdd * self.numWLRow * 2
        
        # Read dynamic energy
        self.readDynamicEnergy += DT['capNandInput'] * self.tech.vdd * self.tech.vdd
        self.readDynamicEnergy += (DT['capInvOutput'] + DT['capTgGateN']) * self.tech.vdd * self.tech.vdd
        self.readDynamicEnergy += (DT['capNandOutput'] + DT['capTgGateN'] + DT['capTgGateP']) * self.tech.vdd * self.tech.vdd
        self.readDynamicEnergy += DT['capTgDrain'] * self.tech.vdd * self.tech.vdd
        self.readDynamicEnergy *= numRead
        
        # Write dynamic energy
        self.writeDynamicEnergy += DT['capNandInput'] * self.tech.vdd * self.tech.vdd
        self.writeDynamicEnergy += (DT['capInvOutput'] + DT['capTgGateN']) * self.tech.vdd * self.tech.vdd
        self.writeDynamicEnergy += (DT['capNandOutput'] + DT['capTgGateN'] + DT['capTgGateP']) * self.tech.vdd * self.tech.vdd
        self.writeDynamicEnergy += DT['capTgDrain'] * self.tech.vdd * self.tech.vdd
        self.writeDynamicEnergy *= numWrite
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        