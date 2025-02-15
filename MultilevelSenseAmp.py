
import math
from constant import *
from from_neurosim.build import FormulaBindings
from MemCell import MemCell

class MultilevelSenseAmp:
    def __init__(self, numCol, levelOutput, numReadCellPerOperationNeuro, parallel, currentMode,
                 param, tech, gate_params):
        
        self.numCol = numCol
        self.levelOutput = levelOutput
        self.numReadCellPerOperationNeuro = numReadCellPerOperationNeuro
        self.parallel = parallel
        self.currentMode = currentMode
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.cell = MemCell(self.param)
        self.numReadCellPerOperationNeuro = self.param.numColSubArray
        
        self.Rref = []
        for i in range(self.levelOutput):
            
            if parallel: # conventionalParallel (Use several multi-bit RRAM as one synapse)
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
        
        self.initialized = True
        
    def CalculateArea(self, widthArray):
        GP = self.gate_params
        PM = self.param
        hNmos, wNmos = GP["hNmos"], GP["wNmos"]
        hPmos, wPmos = GP["hPmos"], GP["wPmos"]
        
        self.area = ((hNmos*wNmos*48) + (hPmos*wPmos*24)) * (self.levelOutput-1) * self.numCol
        self.width = widthArray
        self.height = self.area/self.width
          
    def CalculateLatency(self, columnResistance:list, numColMuxed, numRead):
        self.readLatency = 0
        LatencyCol = 0
        for j in range(len(columnResistance)):
            T_Col = self.GetColumnLatency(columnResistance[j])
            if columnResistance[j] == columnResistance[j]:
                LatencyCol = max(LatencyCol, T_Col)
            else:
                LatencyCol = LatencyCol
            if LatencyCol < 1e-9:
                LatencyCol = 1e-9
            elif LatencyCol > 10e-9:
                LatencyCol = 10e-9
        if self.currentMode:
            self.readLatency = LatencyCol*numColMuxed
            self.readLatency *= numRead
        else:
            self.readLatency = (1e-9)*numColMuxed
            self.readLatency *= numRead
        
        
    def CalculatePower(self, numRead):
        self.leakage = 0
        self.readDynamicEnergy = 0
        
        LatencyCol = 0
        for j in range(len(self.Rref)):
            T_Col = self.GetColumnLatency(self.Rref[j])
            if self.Rref[j] == self.Rref[j]:
                LatencyCol = max(LatencyCol, T_Col)
            else:
                LatencyCol = LatencyCol
            if LatencyCol < 1e-9:
                LatencyCol = 1e-9
            elif LatencyCol > 10e-9:
                LatencyCol = 10e-9
        
        for i in range(len(self.Rref)):
            P_Col = self.GetColumnPower(self.Rref[i])
            if self.currentMode:
                self.readDynamicEnergy += max(P_Col*LatencyCol, 0)
            else:
                self.readDynamicEnergy += max(P_Col*1e-9, 0)
        self.readDynamicEnergy *= numRead
    
    def GetColumnLatency(self, columnRes):
        Column_Latency = 0
        up_bound = 3
        mid_bound = 1.1
        low_bound = 0.9
        T_max = 0
        
        columnRes *= 0.5/self.param.readVoltage
        if (1/columnRes == 0) or (columnRes == 0):
            Column_Latency = 0
        else:
            if self.param.deviceroadmap == 1:
                Column_Latency = 1e-9
            else:
                if self.param.technode == 130:
                    T_max = (0.2679*math.log(columnRes/1000)+0.0478)*1e-9
                    for i in range(1, self.levelOutput-1):
                        ratio = self.Rref[i]/columnRes
                        T = 0
                        if (ratio >= 20) or (ratio <= 0.05):
                            T = 1e-9
                        else:
                            if ratio <= low_bound:
                                T = T_max * (3.915*(ratio**3)-5.3996*(ratio**2)+2.4653*ratio+0.3856)
                            elif (mid_bound <= ratio <= up_bound):
                                T = T_max * (0.0004*(ratio**4)-0.0087*(ratio**3)+0.0742*(ratio**2)-0.2725*ratio+1.2211)
                            elif ratio > up_bound:
                                T = T_max * (0.0004*(ratio**4)-0.0087*(ratio**3)+0.0742*(ratio**2)-0.2725*ratio+1.2211)
                            else:
                                T = T_max
                        Column_Latency = max(Column_Latency, T)
                elif self.param.technode == 90:
                    T_max = (0.0586*math.log(columnRes/1000)+1.41)*1e-9
                    for i in range(1, self.levelOutput-1):
                        ratio = self.Rref[i]/columnRes
                        T = 0
                        if (ratio >= 20) or (ratio <= 0.05):
                            T = 1e-9
                        else:
                            if ratio <= low_bound:
                                T = T_max * (3.726*(ratio**3)-5.651*(ratio**2)+2.8249*ratio+0.3574)
                            elif (mid_bound <= ratio <= up_bound):
                                T = T_max * (0.0000008*(ratio**4)-0.00007*(ratio**3)+0.0017*(ratio**2)-0.0188*ratio+0.9835)
                            elif ratio > up_bound:
                                T = T_max * (0.0000008*(ratio**4)-0.00007*(ratio**3)+0.0017*(ratio**2)-0.0188*ratio+0.9835)
                            else:
                                T = T_max
                        Column_Latency = max(Column_Latency, T)
                elif self.param.technode == 65:
                    T_max = (0.1239*math.log(columnRes/1000)+0.6642)*1e-9
                    for i in range(1, self.levelOutput-1):
                        ratio = self.Rref[i]/columnRes
                        T = 0
                        if (ratio >= 20) or (ratio <= 0.05):
                            T = 1e-9
                        else:
                            if ratio <= low_bound:
                                T = T_max * (1.3899*(ratio**3)-2.6913*(ratio**2)+2.0483*ratio+0.3202)
                            elif (mid_bound <= ratio <= up_bound):
                                T = T_max * (0.0036*(ratio**4)-0.0363*(ratio**3)+0.1043*(ratio**2)-0.0346*ratio+1.0512)
                            elif ratio > up_bound:
                                T = T_max * (0.0036*(ratio**4)-0.0363*(ratio**3)+0.1043*(ratio**2)-0.0346*ratio+1.0512)
                            else:
                                T = T_max
                        Column_Latency = max(Column_Latency, T)
                elif self.param.technode == 45 or self.param.technode == 32:
                    T_max = (0.0714*math.log(columnRes/1000)+0.7651)*1e-9
                    for i in range(1, self.levelOutput-1):
                        ratio = self.Rref[i]/columnRes
                        T = 0
                        if (ratio >= 20) or (ratio <= 0.05):
                            T = 1e-9
                        else:
                            if ratio <= low_bound:
                                T = T_max * (3.7949*(ratio**3)-5.6685*(ratio**2)+2.6492*ratio+0.4807)
                            elif (mid_bound <= ratio <= up_bound):
                                T = T_max * (0.000001*(ratio**4)-0.00006*(ratio**3)+0.0001*(ratio**2)-0.0171*ratio+1.0057)
                            elif ratio > up_bound:
                                T = T_max * (0.000001*(ratio**4)-0.00006*(ratio**3)+0.0001*(ratio**2)-0.0171*ratio+1.0057)
                            else:
                                T = T_max
                        Column_Latency = max(Column_Latency, T)
                else:
                    Column_Latency = 1e-9
        return Column_Latency
    
    """
    double MultilevelSenseAmp::GetColumnPower(double columnRes) {
	double Column_Power = 0;
	// in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
	// we should modify the equivalent columnRes
	columnRes *= 0.5/param->readVoltage;
	if (currentMode) {
		if ((double) 1/columnRes == 0) { 
			Column_Power = 1e-6;
		} else if (columnRes == 0) {
			Column_Power = 0;
		} else {
			if (param->deviceroadmap == 1) {  // HP
				if (param->technode == 130) {
					Column_Power = 19.898*(levelOutput-1)*1e-6;
					Column_Power += 0.17452*exp(-2.367*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 13.09*(levelOutput-1)*1e-6;
					Column_Power += 0.14900*exp(-2.345*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 9.9579*(levelOutput-1)*1e-6;
					Column_Power += 0.1083*exp(-2.321*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 7.7017*(levelOutput-1)*1e-6;
					Column_Power += 0.0754*exp(-2.296*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 3.9648*(levelOutput-1)*1e-6;
					Column_Power += 0.079*exp(-2.313*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 1.8939*(levelOutput-1)*1e-6;
					Column_Power += 0.073*exp(-2.311*log10(columnRes));
				} else if (param->technode == 14){  
					Column_Power = 1.2*(levelOutput-1)*1e-6;
					Column_Power += 0.0584*exp(-2.311*log10(columnRes));
				} else if (param->technode == 10){  
					Column_Power = 0.8*(levelOutput-1)*1e-6;
					Column_Power += 0.0318*exp(-2.311*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.5*(levelOutput-1)*1e-6;
					Column_Power += 0.0210*exp(-2.311*log10(columnRes));
				}
			} else {                         // LP
				if (param->technode == 130) {
					Column_Power = 18.09*(levelOutput-1)*1e-6;
					Column_Power += 0.1380*exp(-2.303*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 12.612*(levelOutput-1)*1e-6;
					Column_Power += 0.1023*exp(-2.303*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 8.4147*(levelOutput-1)*1e-6;
					Column_Power += 0.0972*exp(-2.303*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 6.3162*(levelOutput-1)*1e-6;
					Column_Power += 0.075*exp(-2.303*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 3.0875*(levelOutput-1)*1e-6;
					Column_Power += 0.0649*exp(-2.297*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 1.7*(levelOutput-1)*1e-6;
					Column_Power += 0.0631*exp(-2.303*log10(columnRes));
				} else if (param->technode == 14){   
					Column_Power = 1.0*(levelOutput-1)*1e-6;
					Column_Power += 0.0508*exp(-2.303*log10(columnRes));
				} else if (param->technode == 10){   
					Column_Power = 0.55*(levelOutput-1)*1e-6;
					Column_Power += 0.0315*exp(-2.303*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.35*(levelOutput-1)*1e-6;
					Column_Power += 0.0235*exp(-2.303*log10(columnRes));
				}
			}
		}
		
	} else {
		if ((double) 1/columnRes == 0) { 
			Column_Power = 1e-6;
		} else if (columnRes == 0) {
			Column_Power = 0;
		} else {
			if (param->deviceroadmap == 1) {  // HP
				if (param->technode == 130) {
					Column_Power = 27.84*(levelOutput-1)*1e-6;
					Column_Power += 0.207452*exp(-2.367*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 22.2*(levelOutput-1)*1e-6;
					Column_Power += 0.164900*exp(-2.345*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 13.058*(levelOutput-1)*1e-6;
					Column_Power += 0.128483*exp(-2.321*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 8.162*(levelOutput-1)*1e-6;
					Column_Power += 0.097754*exp(-2.296*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 4.76*(levelOutput-1)*1e-6;
					Column_Power += 0.083709*exp(-2.313*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 2.373*(levelOutput-1)*1e-6;
					Column_Power += 0.084273*exp(-2.311*log10(columnRes));
				} else if (param->technode == 14){  
					Column_Power = 1.467*(levelOutput-1)*1e-6;
					Column_Power += 0.060584*exp(-2.311*log10(columnRes));
				} else if (param->technode == 10){  
					Column_Power = 0.9077*(levelOutput-1)*1e-6;
					Column_Power += 0.049418*exp(-2.311*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.5614*(levelOutput-1)*1e-6;
					Column_Power += 0.040310*exp(-2.311*log10(columnRes));
				}
			} else {                         // LP
				if (param->technode == 130) {
					Column_Power = 23.4*(levelOutput-1)*1e-6;
					Column_Power += 0.169380*exp(-2.303*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 14.42*(levelOutput-1)*1e-6;
					Column_Power += 0.144323*exp(-2.303*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 10.18*(levelOutput-1)*1e-6;
					Column_Power += 0.121272*exp(-2.303*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 7.062*(levelOutput-1)*1e-6;
					Column_Power += 0.100225*exp(-2.303*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 3.692*(levelOutput-1)*1e-6;
					Column_Power += 0.079449*exp(-2.297*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 1.866*(levelOutput-1)*1e-6;
					Column_Power += 0.072341*exp(-2.303*log10(columnRes));
				} else if (param->technode == 14){   
					Column_Power = 1.126*(levelOutput-1)*1e-6;
					Column_Power += 0.061085*exp(-2.303*log10(columnRes));
				} else if (param->technode == 10){   
					Column_Power = 0.6917*(levelOutput-1)*1e-6;
					Column_Power += 0.051580*exp(-2.303*log10(columnRes));
				} else {   // 7nm
					Column_Power = 0.4211*(levelOutput-1)*1e-6;
					Column_Power += 0.043555*exp(-2.303*log10(columnRes));
				}
			}
		}
	}
	Column_Power *= (1+1.3e-3*(param->temp-300));
	return Column_Power;
}
    """
    def GetColumnPower(self, columnRes):
        Column_Power = 0
        # in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
        # we should modify the equivalent columnRes
        columnRes *= 0.5/self.param.readVoltage
        if self.currentMode:
            if (1/columnRes == 0):
                Column_Power = 1e-6
            elif columnRes == 0:
                Column_Power = 0
            else:
                if self.param.deviceroadmap == 1:  # HP
                    if self.param.technode == 130:
                        Column_Power = 19.898*(self.levelOutput-1)*1e-6
                        Column_Power += 0.17452*math.exp(-2.367*math.log10(columnRes))
                    elif self.param.technode == 90:
                        Column_Power = 13.09*(self.levelOutput-1)*1e-6
                        Column_Power += 0.14900*math.exp(-2.345*math.log10(columnRes))
                    elif self.param.technode == 65:
                        Column_Power = 9.9579*(self.levelOutput-1)*1e-6
                        Column_Power += 0.1083*math.exp(-2.321*math.log10(columnRes))
                    elif self.param.technode == 45:
                        Column_Power = 7.7017*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0754*math.exp(-2.296*math.log10(columnRes))
                    elif self.param.technode == 32:
                        Column_Power = 3.9648*(self.levelOutput-1)*1e-6
                        Column_Power += 0.079*math.exp(-2.313*math.log10(columnRes))
                    elif self.param.technode == 22:
                        Column_Power = 1.8939*(self.levelOutput-1)*1e-6
                        Column_Power += 0.073*math.exp(-2.311*math.log10(columnRes))
                    elif self.param.technode == 14:
                        Column_Power = 1.2*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0584*math.exp(-2.311*math.log10(columnRes))
                    elif self.param.technode == 10:
                        Column_Power = 0.8*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0318*math.exp(-2.311*math.log10(columnRes))
                    else:  # 7nm
                        Column_Power = 0.5*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0210*math.exp(-2.311*math.log10(columnRes))
                else:  # LP
                    if self.param.technode == 130:
                        Column_Power = 18.09*(self.levelOutput-1)*1e-6
                        Column_Power += 0.1380*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 90:
                        Column_Power = 12.612*(self.levelOutput-1)*1e-6
                        Column_Power += 0.1023*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 65:
                        Column_Power = 8.4147*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0972*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 45:
                        Column_Power = 6.3162*(self.levelOutput-1)*1e-6
                        Column_Power += 0.075*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 32:
                        Column_Power = 3.0875*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0649*math.exp(-2.297*math.log10(columnRes))
                    elif self.param.technode == 22:
                        Column_Power = 1.7*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0631*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 14:
                        Column_Power = 1.0*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0508*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 10:
                        Column_Power = 0.55*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0315*math.exp(-2.303*math.log10(columnRes))
                    else:  # 7nm
                        Column_Power = 0.35*(self.levelOutput-1)*1e-6
                        Column_Power += 0.0235*math.exp(-2.303*math.log10(columnRes))
        else:
            if (1/columnRes == 0):
                Column_Power = 1e-6
            elif columnRes == 0:
                Column_Power = 0
            else:
                if self.param.deviceroadmap == 1:
                    if self.param.technode == 130:
                        Column_Power = 27.84*(self.levelOutput-1)*1e-6
                        Column_Power += 0.207452*math.exp(-2.367*math.log10(columnRes))
                    elif self.param.technode == 90:
                        Column_Power = 22.2*(self.levelOutput-1)*1e-6
                        Column_Power += 0.164900*math.exp(-2.345*math.log10(columnRes))
                    elif self.param.technode == 65:
                        Column_Power = 13.058*(self.levelOutput-1)*1e-6
                        Column_Power += 0.128483*math.exp(-2.321*math.log10(columnRes))
                    elif self.param.technode == 45:
                        Column_Power = 8.162*(self.levelOutput-1)*1e-6
                        Column_Power += 0.097754*math.exp(-2.296*math.log10(columnRes))
                    elif self.param.technode == 32:
                        Column_Power = 4.76*(self.levelOutput-1)*1e-6
                        Column_Power += 0.083709*math.exp(-2.313*math.log10(columnRes))
                    elif self.param.technode == 22:
                        Column_Power = 2.373*(self.levelOutput-1)*1e-6
                        Column_Power += 0.084273*math.exp(-2.311*math.log10(columnRes))
                    elif self.param.technode == 14:
                        Column_Power = 1.467*(self.levelOutput-1)*1e-6
                        Column_Power += 0.060584*math.exp(-2.311*math.log10(columnRes))
                    elif self.param.technode == 10:
                        Column_Power = 0.9077*(self.levelOutput-1)*1e-6
                        Column_Power += 0.049418*math.exp(-2.311*math.log10(columnRes))
                    else:
                        Column_Power = 0.5614*(self.levelOutput-1)*1e-6
                        Column_Power += 0.040310*math.exp(-2.311*math.log10(columnRes))
                else:
                    if self.param.technode == 130:
                        Column_Power = 23.4*(self.levelOutput-1)*1e-6
                        Column_Power += 0.169380*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 90:
                        Column_Power = 14.42*(self.levelOutput-1)*1e-6
                        Column_Power += 0.144323*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 65:
                        Column_Power = 10.18*(self.levelOutput-1)*1e-6
                        Column_Power += 0.121272*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 45:
                        Column_Power = 7.062*(self.levelOutput-1)*1e-6
                        Column_Power += 0.100225*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 32:
                        Column_Power = 3.692*(self.levelOutput-1)*1e-6
                        Column_Power += 0.079449*math.exp(-2.297*math.log10(columnRes))
                    elif self.param.technode == 22:
                        Column_Power = 1.866*(self.levelOutput-1)*1e-6
                        Column_Power += 0.072341*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 14:
                        Column_Power = 1.126*(self.levelOutput-1)*1e-6
                        Column_Power += 0.061085*math.exp(-2.303*math.log10(columnRes))
                    elif self.param.technode == 10:
                        Column_Power = 0.6917*(self.levelOutput-1)*1e-6
                        Column_Power += 0.051580*math.exp(-2.303*math.log10(columnRes))
                    else:
                        Column_Power = 0.4211*(self.levelOutput-1)*1e-6
                        Column_Power += 0.043555*math.exp(-2.303*math.log10(columnRes))
        Column_Power *= (1+1.3e-3*(self.param.temp-300))
        return Column_Power
            
        
        
    
if __name__ == "__main__":
    # test area
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
    
    from Gate_calculator import compute_gate_params
    gate_params = compute_gate_params(param, tech)
    
    amp = MultilevelSenseAmp(128, 32, 128, True, True, param, tech, gate_params)
    amp.CalculateArea(100)
    print(amp.area)
    colresis = [100, 200, 100]
    amp.CalculateLatency(colresis, 8, 128)
    print(amp.readLatency)
    amp.CalculatePower(128)
    print(amp.readDynamicEnergy)
      
    
    