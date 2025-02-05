import math
import numpy as np

from constant import *
from Array import Array
from AdderTree import AdderTree

def ceil(x):
    return int(math.ceil(x))
def log2(x):
    return math.log(x, 2)

class Tile:
    def __init__(self, stageNum_row, stageNum_col, num_subarray_row, num_subarray_col, subarray_row_size, subarray_col_size,
                 param, tech, gate_params):
        self.stageNum_row = stageNum_row  # num of stage in Y direction
        self.stageNum_col = stageNum_col  # num of stage in X direction
        self.num_subarray_row = num_subarray_row  # num of array in Y direction
        self.num_subarray_col = num_subarray_col  # num of array in X direction
        self.subarray_row_size = subarray_row_size  # size of array in Y direction
        self.subarray_col_size = subarray_col_size  # size of array in X direction
        # self.container = np.zeros((stageNum_row, stageNum_col, 
        #                            num_subarray_row, num_subarray_col, 
        #                            subarray_row_size, subarray_col_size), dtype=np.int8)
        # self.container_old = np.zeros((stageNum_row, stageNum_col,
        #                             num_subarray_row, num_subarray_col,
        #                             subarray_row_size, subarray_col_size), dtype=np.int8)
        
        self.utilization = 0.0
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        #
        self.numBitSubarrayOutput = ceil(log2(subarray_row_size)) \
            + param.cellBit \
            + param.synapseBit \
            + (param.synapseBit - 1)*param.cellBit + 1
        self.numAdderTree = num_subarray_row * (subarray_col_size/param.synapseBit)
        
        self.array = Array(numRow=param.numRowSubArray, numCol=param.numColSubArray,
                           param=param, tech=tech, gate_params=gate_params)
        
        
        
        self.adderTree = AdderTree(numSubcoreRow=stageNum_row, numAdderBit=self.numBitSubarrayOutput,
                                   numAdderTree=self.numAdderTree, 
                                   param=param, tech=tech, gate_params=gate_params)
        
        
        
    def CalculateArea(self):
        
        self.array.CalculateArea()
        arrayH = self.array.height
        arrayW = self.array.width
        self.height = arrayH * self.num_subarray_row
        self.width = arrayW * self.num_subarray_col
        
        self.adderTree.CalculateArea()
        self.height += self.adderTree.height
        # self.width += self.adderTree.width # same as array width
        
        
        self.area = self.height * self.width
        
        
        
        array_usedArea = self.array.usedArea
        array_emptyArea = self.array.emptyArea
        self.usedArea = array_usedArea * self.num_subarray_row * self.num_subarray_col
        self.emptyArea = array_emptyArea * self.num_subarray_row * self.num_subarray_col
        
    def CalculateLatency(self, speedUp:int):
        
        if self.weight_cpoied==False or self.input_copied==False:
            raise ValueError("cannot cal latency, Weight or input matrix is not copied to the Tile.")
        
        # data related latency
        # I. array part
        # self.Latency = 0
        # for stagei in range(self.stageNum_row):
        #     for stagej in range(self.stageNum_col):
        #         for i in range(self.num_subarray_row):
        #             for j in range(self.num_subarray_col):
        #                 subarray_input = self.input_container[stagei][stagej][i][j] # (k, l)
        #                 subarray_weight = self.container[stagei][stagej][i][j] # (k, m)
                        
        #                 for k in range(subarray_input.shape[1]):
        #                     input_col = subarray_input[:, k]
        #                     columnResistance = get_column_resistance_np(input_col, subarray_weight)
        #                     self.array.CalculateLatency(columnResistance)
        #                     self.Latency += self.array.readLatency
        
        self.readLatency, self.writeLatency = 0, 0
        subarray_input = self.input_container[0][0][0][0] #8b

        self.input_bitlen = subarray_input.shape[1]
        self.num_nbit_inputs = self.input_bitlen / self.param.synapseBit # 8b/8
        
        self.array.CalculateLatency()
        self.readLatency += self.array.readLatency * self.input_bitlen
        self.writeLatency += self.array.writeLatency
        
        # II. peripheral part
        # skip for now (buffer, relu, etc.)
        self.adderTree.CalculateLatency(self.num_nbit_inputs, 
                                        self.num_subarray_row, 0)
        self.readLatency += self.adderTree.readLatency
        
        
        self.readLatency *= self.stageNum_row * self.stageNum_col * self.num_subarray_row * self.num_subarray_col
        self.readLatency /= speedUp
        
        # writeLatency no need speedup
        
        
    
    def CalculatePower(self):
        if self.weight_cpoied==False or self.input_copied==False:
            raise ValueError("cannot cal latency, Weight or input matrix is not copied to the Tile.")
        
        self.readDynamicEnergy = 0
        self.writeDynamicEnergy = 0
        self.leakage = 0
        calTimes = self.stageNum_row * self.stageNum_col * self.num_subarray_row * self.num_subarray_col
        
        # Write Dynamic Energy (SUM all stages, all subarrays in this func)
        # self.GetWriteUpdateEstimation() # slow, not recommended
        self.GetWriteUpdateEstimation_speedUp() # speedup using numpy OPs
                
        # DEBUG: check pulses
        # print(f"setPulse: {self.totalNumSetWritePulse}, {self.totalNumSetWritePulse/self.totalNumWritePulse:.2f}")
        # print(f"resetPulse: {self.totalNumResetWritePulse}, {self.totalNumResetWritePulse/self.totalNumWritePulse:.2f}")
        
        # Read Dynamic Energy
        
        # SLOW cal: all stage and all subarray's read energy
        for stagei in range(self.stageNum_row):
            for stagej in range(self.stageNum_col):
                for i in range(self.num_subarray_row):
                    for j in range(self.num_subarray_col):
                        
                        subarray_input = self.input_container[stagei][stagej][i][j]
                        
                        nonzero_counts = np.count_nonzero(subarray_input, axis=0)
                        activityRowReads = nonzero_counts / self.subarray_row_size    # 得到每一列的活动率

                        # 然后遍历每一列的活动率进行功率计算
                        for activity in activityRowReads:
                            self.array.activityRowRead = activity
                            self.array.CalculatePower()  # 计算后内部会更新 self.array.readDynamicEnergy
                            self.readDynamicEnergy += self.array.readDynamicEnergy
                    
        # FAST cal: assume all columns have the same activity rate
        # activity = 0.75
        # self.array.activityRowRead = activity
        # self.array.CalculatePower()
        # self.readDynamicEnergy += self.array.readDynamicEnergy * calTimes * self.num_nbit_inputs
        
        self.adderTree.CalculatePower(self.num_nbit_inputs, self.num_subarray_row) # for n bits
        self.readDynamicEnergy += self.adderTree.readDynamicEnergy * calTimes
        self.leakage += self.adderTree.leakage * calTimes
    

    def GetWriteUpdateEstimation(self):
        # from NeuroSIM 2.0
        
        PM = self.param
        totalNumWritePulse, numWritePulseAVG, activityColWrite, activityRowWrite = 0, 0, 0, 0
        totalNumSetWritePulse, totalNumResetWritePulse = 0, 0
        maxNumWritePulse = max(PM.maxNumLevelLTP, PM.maxNumLevelLTD)
        self.minDeltaConductance = (PM.maxConductance-PM.minConductance)/maxNumWritePulse
        
        minG, maxG = PM.minConductance, PM.maxConductance
        
        # cal all stage and all subarray's write energy
        for stagei in range(self.stageNum_row):
            for stagej in range(self.stageNum_col):
                for arri in range(self.num_subarray_row):
                    for arrj in range(self.num_subarray_col):
                        
                        subarray_weight = self.container[stagei][stagej][arri][arrj]
                        subarray_weight_old = self.container_old[stagei][stagej][arri][arrj]
                        
                        if self.param.memcelltype == RRAM:
                            # map 0->minG, 1->maxG, (for empty cells)
                            subarray_weight = np.where(subarray_weight==0, minG, subarray_weight)
                            subarray_weight = np.where(subarray_weight==1, maxG, subarray_weight)
                            subarray_weight_old = np.where(subarray_weight_old==0, minG, subarray_weight_old)
                            subarray_weight_old = np.where(subarray_weight_old==1, maxG, subarray_weight_old)
                        
                        numSelectedRowSet, numSelectedRowReset = 0, 0
                        numSelectedColSet, numSelectedColReset = 0, 0
                        for i in range(subarray_weight.shape[0]): # sweep col for a row
                            
                            numSet, numReset = 0, 0
                            numSetWritePulse, numResetWritePulse = 0, 0
                            rowSelected = False
                            
                            for j in range(subarray_weight.shape[1]):
                                if self.param.memcelltype == RRAM:
                                    
                                    # determine if a pulse is needed
                                    if abs(subarray_weight[i][j] - subarray_weight_old[i][j]) >= self.minDeltaConductance:
                                        rowSelected = True
                                        
                                        if subarray_weight[i][j] > subarray_weight_old[i][j]: # LTP
                                            numSet += 1
                                            thisPulse = ceil(abs(subarray_weight[i][j] - subarray_weight_old[i][j]) / self.minDeltaConductance)
                                            numSetWritePulse = max(numSetWritePulse, thisPulse)
                                            # energy in each cell
                                            V = PM.writeVoltage
                                            R = abs(1/subarray_weight[i][j] + 1/subarray_weight_old[i][j]) / 2
                                            pulse_wd = PM.writePulseWidth # write pulse width
                                            self.writeDynamicEnergy += V * (V / R * pulse_wd) * thisPulse
                                
                                        else: # LTD
                                            numReset += 1
                                            thisPulse = ceil(abs(subarray_weight[i][j] - subarray_weight_old[i][j]) / self.minDeltaConductance)
                                            numResetWritePulse = max(numResetWritePulse, thisPulse)
                                            # energy in each cell
                                            V = PM.writeVoltage
                                            R = abs(1/subarray_weight[i][j] + 1/subarray_weight_old[i][j]) / 2
                                            pulse_wd = PM.writePulseWidth # write pulse width
                                            self.writeDynamicEnergy += V * (V / R * pulse_wd) * thisPulse

                                    else: # no update
                                        numSet += 0
                                        numReset += 0
                                
                                else: # SRAM
                                    raise ValueError("SRAM or others are not supported for now.")
                            
                            if rowSelected and numSet>0: # if set happens in this row
                                numSelectedRowSet += 1
                            elif rowSelected and numReset>0: # if reset happens in this row
                                numSelectedRowReset += 1
                            else:
                                numSelectedRowSet += 0
                                numSelectedRowReset += 0
                            
                            numSelectedColSet += numSet
                            numSelectedColReset += numReset
                            totalNumSetWritePulse += numSetWritePulse
                            totalNumResetWritePulse += numResetWritePulse
                            
                        """
                        // get average num of selected column for set and reset
                        numSelectedColSet = numSelectedRowSet==0? 0:ceil(numSelectedColSet/numSelectedRowSet);
                        numSelectedColReset = numSelectedRowReset==0? 0:ceil(numSelectedColReset/numSelectedRowReset);
                            
                        *totalNumWritePulse = totalNumResetWritePulse + totalNumSetWritePulse;
                        *numWritePulseAVG = (*totalNumWritePulse)/(MAX(1, (numSelectedRowSet+numSelectedRowReset)/2.0));
                        *activityColWrite = ((numSelectedColSet+numSelectedColReset)/2.0)/newMemory[0].size();
                        *activityRowWrite = ((numSelectedRowSet+numSelectedRowReset)/2.0)/newMemory.size();	
                        """
                        # get average num of selected column for set and reset
                        if numSelectedRowSet == 0:
                            numSelectedColSet = 0
                        else:
                            numSelectedColSet = ceil(numSelectedColSet/numSelectedRowSet)
                            
                        if numSelectedRowReset == 0:
                            numSelectedColReset = 0
                        else:
                            numSelectedColReset = ceil(numSelectedColReset/numSelectedRowReset)
                            
                        totalNumWritePulse += totalNumResetWritePulse + totalNumSetWritePulse
                        numWritePulseAVG += totalNumWritePulse / max(1, (numSelectedRowSet + numSelectedRowReset) / 2.0)
                        activityColWrite += ((numSelectedColSet + numSelectedColReset) / 2.0) / subarray_weight.shape[1]
                        activityRowWrite += ((numSelectedRowSet + numSelectedRowReset) / 2.0) / subarray_weight.shape[0]
                        

    def GetWriteUpdateEstimation_speedUp(self):
        # from NeuroSIM 2.0
        
        PM = self.param
        maxNumWritePulse = max(PM.maxNumLevelLTP, PM.maxNumLevelLTD)
        self.minDeltaConductance = (PM.maxConductance-PM.minConductance)/maxNumWritePulse
        
        minG, maxG = PM.minConductance, PM.maxConductance
        
        self.numSelectedRowSet, self.numSelectedRowReset = 0, 0
        self.totalSelectedColSet, self.totalSelectedColReset = 0, 0
        self.totalNumSetWritePulse, self.totalNumResetWritePulse = 0, 0
        self.totalNumWritePulse = 0
        
        self.writeDynamicEnergy = 0
        
        # cal all stage and all subarray's write energy
        for stagei in range(self.stageNum_row):
            for stagej in range(self.stageNum_col):
                for arri in range(self.num_subarray_row):
                    for arrj in range(self.num_subarray_col):
                        
                        if self.param.memcelltype == RRAM:
                        
                            subarray_weight = self.container[stagei][stagej][arri][arrj]
                            subarray_weight_old = self.container_old[stagei][stagej][arri][arrj]
                            
                            # map 0->minG, 1->maxG, (for empty cells)
                            subarray_weight = np.where(subarray_weight==0, minG, subarray_weight)
                            subarray_weight = np.where(subarray_weight==1, maxG, subarray_weight)
                            subarray_weight_old = np.where(subarray_weight_old==0, minG, subarray_weight_old)
                            subarray_weight_old = np.where(subarray_weight_old==1, maxG, subarray_weight_old)

                            
                            delta = subarray_weight - subarray_weight_old
                            abs_delta = np.abs(delta)
                            
                            # debug info
                            # print("subarray_weight sample:", subarray_weight[:5, :5])
                            # print("subarray_weight_old sample:", subarray_weight_old[:5, :5])
                            # print("delta sample:", delta[:5, :5])
                            # print("abs_delta sample:", abs_delta[:5, :5])
                            
                            update_mask = abs_delta >= self.minDeltaConductance
                            
                            pulse_counts = np.ceil(abs_delta / self.minDeltaConductance)
                            
                            LTP_mask = (delta > 0) & update_mask
                            LTD_mask = (delta < 0) & update_mask
                            # LTP_pulses = pulse_counts[LTP_mask]
                            # LTD_pulses = pulse_counts[LTD_mask]
                            
                            V = PM.writeVoltage
                            pulse_wd = PM.writePulseWidth
                            
                            R = np.abs(1/subarray_weight + 1/subarray_weight_old) / 2
                            
                            energy = np.zeros_like(subarray_weight, dtype=float)
                            energy[update_mask] = V * ((V / R[update_mask]) * pulse_wd) * pulse_counts[update_mask]
                            
                            energy_LTP = np.sum(energy[LTP_mask])
                            energy_LTD = np.sum(energy[LTD_mask])
                            self.writeDynamicEnergy += energy_LTP + energy_LTD
                            
                            # 统计行和列上的更新数量
                            # 对于每一行，判断是否存在 LTP 或 LTD 更新
                            rows_with_LTP = np.any(LTP_mask, axis=1)  # 返回一个布尔数组，每个元素表示该行是否有 LTP
                            rows_with_LTD = np.any(LTD_mask, axis=1)  # 同理
                            
                            self.numSelectedRowSet += np.sum(rows_with_LTP)
                            self.numSelectedRowReset += np.sum(rows_with_LTD)
                            
                            # 对于列上更新计数（这里假设你希望统计更新次数）
                            numSelectedColSet = np.sum(LTP_mask, axis=0)  # 这会返回一个数组，表示每一列的 LTP 更新数
                            numSelectedColReset = np.sum(LTD_mask, axis=0)
                            
                            self.totalSelectedColSet += np.sum(numSelectedColSet)
                            self.totalSelectedColReset += np.sum(numSelectedColReset)

                            # 为了计算每一行的最大脉冲数，只对更新的元素计算
                            # 初始化一个数组来存放每行的最大脉冲数
                            row_max_LTP_pulses = np.zeros(subarray_weight.shape[0])
                            row_max_LTD_pulses = np.zeros(subarray_weight.shape[0])

                            for i in range(subarray_weight.shape[0]):
                                # 针对每一行，提取对应行的脉冲数和更新标记
                                row_LTP = pulse_counts[i, :][LTP_mask[i, :]]
                                row_LTD = pulse_counts[i, :][LTD_mask[i, :]]
                                if row_LTP.size > 0:
                                    row_max_LTP_pulses[i] = np.max(row_LTP)
                                if row_LTD.size > 0:
                                    row_max_LTD_pulses[i] = np.max(row_LTD)

                            localSetPulse = np.sum(row_max_LTP_pulses)
                            localResetPulse = np.sum(row_max_LTD_pulses)
                            self.totalNumSetWritePulse += localSetPulse
                            self.totalNumResetWritePulse += localResetPulse
                            self.totalNumWritePulse += (localSetPulse + localResetPulse)

                        else:
                            raise ValueError("SRAM or others are not supported for now.")
        

    def copy_weight(self, matrix):
        self.container = np.zeros((self.stageNum_row, self.stageNum_col, 
                                   self.num_subarray_row, self.num_subarray_col, 
                                   self.subarray_row_size, self.subarray_col_size))
        
        print("copying weight matrix to the Tile...")
        total_capacity_rows = self.stageNum_row * self.num_subarray_row * self.subarray_row_size
        total_capacity_cols = self.stageNum_col * self.num_subarray_col * self.subarray_col_size
        
        maxConductance = self.param.maxConductance
        minConductance = self.param.minConductance
        if self.param.memcelltype == RRAM: # RRAM, analog value, map 0->minConductance, 1->maxConductance
            # linear mapping
            matrix = (matrix+1)/2*(maxConductance-minConductance)+minConductance
            # check if any value is 0
            if np.count_nonzero(matrix==0):
                raise ValueError("0 conductance value is not allowed in RRAM.")

        if matrix.shape[0] > total_capacity_rows or matrix.shape[1] > total_capacity_cols:
            raise ValueError("Matrix size exceeds the capacity of a Tile.")

        row_limit = min(matrix.shape[0], self.stageNum_row * self.num_subarray_row * self.subarray_row_size)
        col_limit = min(matrix.shape[1], self.stageNum_col * self.num_subarray_col * self.subarray_col_size)

        # Calculate the size of each stage
        stage_row_size = self.num_subarray_row * self.subarray_row_size
        stage_col_size = self.num_subarray_col * self.subarray_col_size

        # Fill in the 6D container
        for i in range(row_limit):
            for j in range(col_limit):
                stage_row = i // stage_row_size
                stage_col = j // stage_col_size
                subarray_row = (i % stage_row_size) // self.subarray_row_size
                subarray_row_index = i % self.subarray_row_size
                subarray_col = (j % stage_col_size) // self.subarray_col_size
                subarray_col_index = j % self.subarray_col_size
                # Assign the value
                self.container[stage_row][stage_col][subarray_row][subarray_col][subarray_row_index][subarray_col_index] = matrix[i][j]

        self.utilization = (row_limit * col_limit) / \
                            (self.stageNum_row * self.num_subarray_row * self.subarray_row_size * self.stageNum_col * self.num_subarray_col * self.subarray_col_size)

        self.weight_cpoied = True

    def copy_weight_old(self, matrix):
        self.container_old = np.zeros((self.stageNum_row, self.stageNum_col,
                                    self.num_subarray_row, self.num_subarray_col,
                                    self.subarray_row_size, self.subarray_col_size))
        print("copying weight matrix (OLD) to the Tile...")
        total_capacity_rows = self.stageNum_row * self.num_subarray_row * self.subarray_row_size
        total_capacity_cols = self.stageNum_col * self.num_subarray_col * self.subarray_col_size
        
        maxConductance = self.param.maxConductance
        minConductance = self.param.minConductance
        if self.param.memcelltype == RRAM: # RRAM, analog value, map 0->minConductance, 1->maxConductance
            # linear mapping
            matrix = (matrix+1)/2*(maxConductance-minConductance)+minConductance
            # check if any value is 0
            if np.count_nonzero(matrix==0):
                raise ValueError("0 conductance value is not allowed in RRAM.")

        if matrix.shape[0] > total_capacity_rows or matrix.shape[1] > total_capacity_cols:
            raise ValueError("Matrix size exceeds the capacity of a Tile.")

        row_limit = min(matrix.shape[0], self.stageNum_row * self.num_subarray_row * self.subarray_row_size)
        col_limit = min(matrix.shape[1], self.stageNum_col * self.num_subarray_col * self.subarray_col_size)

        # Calculate the size of each stage
        stage_row_size = self.num_subarray_row * self.subarray_row_size
        stage_col_size = self.num_subarray_col * self.subarray_col_size

        # Fill in the 6D container
        for i in range(row_limit):
            for j in range(col_limit):
                stage_row = i // stage_row_size
                stage_col = j // stage_col_size
                subarray_row = (i % stage_row_size) // self.subarray_row_size
                subarray_row_index = i % self.subarray_row_size
                subarray_col = (j % stage_col_size) // self.subarray_col_size
                subarray_col_index = j % self.subarray_col_size
                # Assign the value
                self.container_old[stage_row][stage_col][subarray_row][subarray_col][subarray_row_index][subarray_col_index] = matrix[i][j]

        self.utilization = (row_limit * col_limit) / \
                            (self.stageNum_row * self.num_subarray_row * self.subarray_row_size * self.stageNum_col * self.num_subarray_col * self.subarray_col_size)

        self.weight_cpoied = True

    def copy_input(self, input_matrix):
        # 6 dim input container
        self.input_container = np.zeros((self.stageNum_row, self.stageNum_col, self.num_subarray_row, self.num_subarray_col,
                                         self.subarray_row_size, input_matrix.shape[1]), dtype=np.int8)

        # Fill in the 6D input container
        for ii in range(self.stageNum_row):
            for jj in range(self.stageNum_col):
                for kk in range(self.num_subarray_row):
                    for ll in range(self.num_subarray_col):
                        row_index = ii * self.num_subarray_row * self.subarray_row_size + kk * self.subarray_row_size
                        for i in range(min(self.subarray_row_size, input_matrix.shape[0] - row_index)):
                            for j in range(input_matrix.shape[1]):
                                self.input_container[ii][jj][kk][ll][i][j] = input_matrix[row_index + i][j]

        self.input_copied = True

    def save_weight_container_bin(self, filename):
        with open(filename, 'wb') as outfile:
            # Iterate over the 6D container and write each row to the file
            for stage_row in self.container:
                for stage_col in stage_row:
                    for subarray_row in stage_col:
                        for subarray_col in subarray_row:
                            for row in subarray_col:
                                outfile.write(row.tobytes())

    def save_input_container_bin(self, filename):
        with open(filename, 'wb') as outfile:
            # Iterate over the 6D input container and write each row to the file
            for stage_row in self.input_container:
                for stage_col in stage_row:
                    for subarray_row in stage_col:
                        for subarray_col in subarray_row:
                            for row in subarray_col:
                                outfile.write(row.tobytes())




