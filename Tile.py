import math
import numpy as np
from tqdm import tqdm

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
        
        self.utilization = 0.0
        self.param = param
        self.tech = tech
        self.gate_params = gate_params
        
        self.numBitSubarrayOutput = ceil(log2(subarray_row_size)) \
            + param.cellBit \
            + param.synapseBit \
            + (param.synapseBit - 1)*param.cellBit + 1
        self.numAdderTree = num_subarray_row * (subarray_col_size/param.synapseBit)
        
        self.totalNumWritePulse = -1
        
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
        
        # I. array part
        self.readLatency, self.writeLatency = 0, 0
        subarray_input = self.input_container[0][0][0][0] # 8b

        self.input_bitlen = subarray_input.shape[1]
        self.num_nbit_inputs = self.input_bitlen / self.param.synapseBit # 8b/8
        
        # Write Dynamic Energy (SUM all stages, all subarrays in this func),
        # calculate here because we need numpulses for write latency
        self.GetWriteUpdateEstimation_speedUp()
        
        if self.totalNumWritePulse == -1: raise ValueError("totalNumWritePulse is not calculated.")
        self.array.totalNumWritePulse = self.totalNumWritePulse
        
        #########################################
        # Cycle-accurate latency calculation (SLOW)
        #########################################
        for stagei in range(self.stageNum_row):
            for stagej in range(self.stageNum_col):
                for i in range(self.num_subarray_row):
                    for j in range(self.num_subarray_col):
                        
                        ############################################
                        # Slow calculation
                        ############################################
                        # crossBarInput = self.input_container[stagei][stagej][i][j]
                        # crossBarWeight = self.container[stagei][stagej][i][j]
                        
                        # for k in tqdm(range(crossBarInput.shape[1])):
                        #     inputVec = crossBarInput[:, k]
                        #     columnResistance = self.GetColumnResistance(inputVec, crossBarWeight)
                        #     self.array.CalculateLatency(columnResistance)
                        #     self.readLatency += self.array.readLatency
                        
                        ############################################
                        # Vectorized calculation
                        ############################################
                        crossBarInput = self.input_container[stagei][stagej][i][j]  # shape: (num_rows, num_k)
                        crossBarWeight = self.container[stagei][stagej][i][j]         # shape: (num_rows, num_cols)
                        
                        # Vectorized calculation: get resistances for all input vectors at once.
                        # The returned array has shape (num_cols, num_k) where each column corresponds to a given input vector.
                        columnResistance_all = self.GetColumnResistance_vectorized(crossBarInput, crossBarWeight)
                        
                        # If your CalculateLatency can process multiple sets of column resistances at once, you can vectorize further.
                        # Otherwise, loop over the computed resistances for each input vector.
                        for colRes in columnResistance_all.T:  # iterate over each input vector's result (shape: (num_cols,))
                            self.array.CalculateLatency(colRes)
                            self.readLatency += self.array.readLatency
        
        ##############################
        # Fast estimate
        ##############################
        # self.array.CalculateLatency(columnResistance)
        # self.readLatency += self.array.readLatency * self.input_bitlen
        
        # II. peripheral part
        # skip for now (buffer, relu, etc.)
        self.adderTree.CalculateLatency(self.num_nbit_inputs, self.num_subarray_row, 0)
        # self.readLatency += self.adderTree.readLatency
        
        
        # self.readLatency *= self.stageNum_row * self.stageNum_col * self.num_subarray_row * self.num_subarray_col
        self.readLatency /= speedUp
        
        # writeLatency no need speedup
        self.writeLatency += self.array.writeLatency
        
        
    def CalculatePower(self):
        if self.weight_cpoied==False or self.input_copied==False:
            raise ValueError("cannot cal latency, Weight or input matrix is not copied to the Tile.")
        
        self.readDynamicEnergy = 0
        # self.writeDynamicEnergy = 0
        self.leakage = 0
        calTimes = self.stageNum_row * self.stageNum_col * self.num_subarray_row * self.num_subarray_col
                
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
                        activityRowReads = nonzero_counts / self.subarray_row_size    # each col's activity rate 

                        for activity in activityRowReads:
                            self.array.activityRowRead = activity
                            self.array.CalculatePower()
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
        
        self.writeDynamicEnergy = 0
        
        # cal all stage and all subarray's write energy
        for stagei in range(self.stageNum_row):
            for stagej in range(self.stageNum_col):
                for arri in range(self.num_subarray_row):
                    for arrj in range(self.num_subarray_col):
                        
                        if self.param.memcelltype == RRAM:
                        
                            subarray_weight = self.container[stagei][stagej][arri][arrj]
                            subarray_weight_old = self.container_old[stagei][stagej][arri][arrj]

                            delta = subarray_weight - subarray_weight_old
                            abs_delta = np.abs(delta)
                            
                            update_mask = abs_delta >= self.minDeltaConductance
                            
                            pulse_counts = np.ceil(abs_delta / self.minDeltaConductance)
                            
                            LTP_mask = (delta > 0) & update_mask
                            LTD_mask = (delta < 0) & update_mask
                            
                            V = PM.writeVoltage
                            pulse_wd = PM.writePulseWidth
                            
                            R = np.abs(1/subarray_weight + 1/subarray_weight_old) / 2
                            
                            energy = np.zeros_like(subarray_weight, dtype=float)
                            energy[update_mask] = V * ((V / R[update_mask]) * pulse_wd) * pulse_counts[update_mask]
                            
                            energy_LTP = np.sum(energy[LTP_mask])
                            energy_LTD = np.sum(energy[LTD_mask])
                            self.writeDynamicEnergy += energy_LTP + energy_LTD
                            
                            # Count updates on rows and columns
                            # For each row, determine if there is an LTP or LTD update
                            rows_with_LTP = np.any(LTP_mask, axis=1)  # Returns a Boolean array, each element indicates whether the line has LTP
                            rows_with_LTD = np.any(LTD_mask, axis=1)  # same as above
                            
                            self.numSelectedRowSet += np.sum(rows_with_LTP)
                            self.numSelectedRowReset += np.sum(rows_with_LTD)
                            
                            # For update counts on columns (assuming you want to count updates)
                            numSelectedColSet = np.sum(LTP_mask, axis=0)  # This returns an array with the number of LTP updates for each column.
                            numSelectedColReset = np.sum(LTD_mask, axis=0)
                            
                            self.totalSelectedColSet += np.sum(numSelectedColSet)
                            self.totalSelectedColReset += np.sum(numSelectedColReset)

                            # To calculate the maximum number of pulses per row, only the updated elements are calculated
                            # Initialize an array to store the maximum number of pulses per row
                            row_max_LTP_pulses = np.zeros(subarray_weight.shape[0])
                            row_max_LTD_pulses = np.zeros(subarray_weight.shape[0])

                            for i in range(subarray_weight.shape[0]):
                                # For each row, extract the pulse number and update flag of the corresponding row
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
        
    
    def GetColumnResistance(self, inputVec: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Compute the column resistances for an RRAM array with parallel read.
        
        Parameters:
            inputVec: 1D NumPy array of binary activations (0 or 1) with length equal to the number of rows.
            weight:   2D NumPy array (shape: [num_rows, num_cols]) representing the device conductances.
        
        Returns:
            A 1D NumPy array containing the resistance for each column.
        """
        numrows, numcols = weight.shape
        conductance_list = []
        resistance_list = []
        
        for col_id in range(numcols):
            col_cond = 0.0
            for row_id in range(numrows):
                if inputVec[row_id] == 1:
                    # Compute the intrinsic device resistance (inverse of device conductance)
                    device_resistance = 1.0 / weight[row_id, col_id]
                    # Total resistance includes device resistance, row and column wire resistances,
                    # and the access resistance.
                    total_wire_res = (
                        device_resistance +
                        (col_id + 1) * self.param.wireResistanceRow +
                        (numrows - row_id) * self.param.wireResistanceCol +
                        self.array.cell.resCellAccess
                    )
                    # Sum the contribution from this activated cell (conductance = 1/total resistance)
                    col_cond += 1.0 / total_wire_res
            
            conductance_list.append(col_cond)
        
        # Convert column conductance to resistance (avoid division by zero)
        for cond in conductance_list:
            if cond != 0:
                resistance_list.append(1.0 / cond)
            else:
                resistance_list.append(np.inf)
        
        return np.array(resistance_list)
        
    
    def GetColumnResistance_vectorized(self, inputMat: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of column resistances for an RRAM array with parallel read.
        
        Parameters:
            inputMat: 2D NumPy array of binary inputs with shape (num_rows, num_k),
                    where each column is an input vector.
            weight:   2D NumPy array of device conductances with shape (num_rows, num_cols).
        
        Returns:
            A 2D NumPy array of column resistances with shape (num_cols, num_k).
            (Each column corresponds to the resistance of each crossbar column for one input vector.)
        """
        num_rows, num_cols = weight.shape
        _, num_k = inputMat.shape

        # Create index arrays for broadcasting
        rows = np.arange(num_rows).reshape(num_rows, 1)   # shape: (num_rows, 1)
        cols = np.arange(num_cols).reshape(1, num_cols)     # shape: (1, num_cols)

        # Compute the intrinsic device resistance for each cell
        device_resistance = 1.0 / weight  # shape: (num_rows, num_cols)

        # Wire resistances:
        wire_row_offset = (cols + 1) * self.param.wireResistanceRow   # shape: (1, num_cols)
        wire_col_offset = (num_rows - rows) * self.param.wireResistanceCol  # shape: (num_rows, 1)

        # Total resistance seen by each cell
        cell_total_res = device_resistance + wire_row_offset + wire_col_offset + self.array.cell.resCellAccess  
        # shape: (num_rows, num_cols)

        # Expand cell_total_res to handle multiple input vectors:
        # We'll compute contributions for each input vector (k)
        # Input mask: for each cell, for each input vector, check if it is activated.
        input_mask = inputMat[:, np.newaxis, :]  # shape: (num_rows, 1, num_k)

        # Compute contribution (conductance) for each cell for each input vector:
        # If activated (==1), contribution is 1 / cell_total_res; else 0.
        # We expand cell_total_res to shape (num_rows, num_cols, 1)
        contributions = np.where(input_mask == 1, 1.0 / cell_total_res[:, :, np.newaxis], 0.0)
        # contributions shape: (num_rows, num_cols, num_k)

        # Sum contributions over rows to get the total column conductance for each input vector.
        column_conductance = np.sum(contributions, axis=0)  # shape: (num_cols, num_k)

        # Convert conductance to resistance. Avoid division by zero:
        # column_resistance = np.where(column_conductance != 0, 1.0 / column_conductance, np.inf)
        column_resistance = np.full(column_conductance.shape, np.inf)
        np.divide(1.0, column_conductance, out=column_resistance, where=column_conductance!=0)

        return column_resistance  # shape: (num_cols, num_k)


    def copy_weight(self, matrix):
        self.container = np.zeros((self.stageNum_row, self.stageNum_col, 
                                   self.num_subarray_row, self.num_subarray_col, 
                                   self.subarray_row_size, self.subarray_col_size))
        
        # print("copying weight matrix to the Tile...")
        total_capacity_rows = self.stageNum_row * self.num_subarray_row * self.subarray_row_size
        total_capacity_cols = self.stageNum_col * self.num_subarray_col * self.subarray_col_size

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

        # linear mapping weight to conductance
        maxConductance = self.param.maxConductance
        minConductance = self.param.minConductance
        self.container = (self.container+1)/2*(maxConductance-minConductance)+minConductance
        
        self.weight_cpoied = True

    def copy_weight_old(self, matrix):
        self.container_old = np.zeros((self.stageNum_row, self.stageNum_col,
                                    self.num_subarray_row, self.num_subarray_col,
                                    self.subarray_row_size, self.subarray_col_size))
        # print("copying weight matrix (OLD) to the Tile...")
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

        maxConductance = self.param.maxConductance
        minConductance = self.param.minConductance
        self.container_old = (self.container_old+1)/2*(maxConductance-minConductance)+minConductance
        
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




