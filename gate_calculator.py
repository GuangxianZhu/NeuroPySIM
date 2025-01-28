
from from_neurosim.build import FormulaBindings
from constant import *
from typing import Dict

def compute_gate_params(param, tech) -> Dict[str, float]:
    
    gate_params = {}
    widthNmos = MIN_NMOS_SIZE * tech.featureSize
    widthPmos = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize
    hNmos = CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT , tech)['height']
    wNmos = CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT , tech)['width']
    aNmos = CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT , tech)['area']
    hPmos = CalculateGateArea(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT , tech)['height']
    wPmos = CalculateGateArea(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT , tech)['width']
    aPmos = CalculateGateArea(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT , tech)['area']
    capNmosInput = CalculateGateCapacitance(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech)['capInput']
    capNmosOutput = CalculateGateCapacitance(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech)['capOutput']
    capPmosInput = CalculateGateCapacitance(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech)['capInput']
    capPmosOutput = CalculateGateCapacitance(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech)['capOutput']
    gate_params['widthNmos'], gate_params['widthPmos'] = widthNmos, widthPmos
    gate_params['hNmos'], gate_params['wNmos'], gate_params['aNmos'] = hNmos, wNmos, aNmos
    gate_params['hPmos'], gate_params['wPmos'], gate_params['aPmos'] = hPmos, wPmos, aPmos
    gate_params['capNmosInput'], gate_params['capNmosOutput'] = capNmosInput, capNmosOutput
    gate_params['capPmosInput'], gate_params['capPmosOutput'] = capPmosInput, capPmosOutput
    
    # INV
    widthInvN = MIN_NMOS_SIZE * tech.featureSize
    widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize
    capInvInput = CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['capInput']
    capInvOutput = CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['capOutput']
    resInv = CalculateOnResistance(widthInvN, NMOS, param.temp, tech)
    hInv = CalculateGateArea(INV, 1, widthInvN, widthInvP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['height']
    wInv = CalculateGateArea(INV, 1, widthInvN, widthInvP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['width']
    aInv = CalculateGateArea(INV, 1, widthInvN, widthInvP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['area']
    leakageInv = CalculateGateLeakage(INV, 1, widthInvN, widthInvP, param.temp, tech) * tech.vdd
    gate_params['widthInvN'], gate_params['widthInvP'] = widthInvN, widthInvP
    gate_params['capInvInput'], gate_params['capInvOutput'] = capInvInput, capInvOutput
    gate_params['resInv'], gate_params['hInv'], gate_params['wInv'], gate_params['aInv'] = resInv, hInv, wInv, aInv
    gate_params['leakageInv'] = leakageInv
    
    # driver INV (MUX)
    widthDriverInvN = 2 * MIN_NMOS_SIZE * tech.featureSize * param.sizingfactor_MUX
    widthDriverInvP = 2 * tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * param.sizingfactor_MUX
    gate_params['widthDriverInvN'], gate_params['widthDriverInvP'] = widthDriverInvN, widthDriverInvP

    # NAND2
    widthNandN = 2 * MIN_NMOS_SIZE * tech.featureSize
    widthNandP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize
    capNandInput = CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['capInput']
    capNandOutput = CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['capOutput']
    resNandN = CalculateOnResistance(widthNandN, NMOS, param.temp, tech)
    resNandP = CalculateOnResistance(widthNandP, PMOS, param.temp, tech)
    hNand = CalculateGateArea(NAND, 2, widthNandN, widthNandP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['height']
    wNand = CalculateGateArea(NAND, 2, widthNandN, widthNandP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['width']
    aNand = CalculateGateArea(NAND, 2, widthNandN, widthNandP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['area']
    leakageNand = CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, param.temp, tech) * tech.vdd
    gate_params['widthNandN'], gate_params['widthNandP'] = widthNandN, widthNandP
    gate_params['capNandInput'], gate_params['capNandOutput'] = capNandInput, capNandOutput
    gate_params['resNandN'], gate_params['resNandP'] = resNandN, resNandP
    gate_params['hNand'], gate_params['wNand'], gate_params['aNand'] = hNand, wNand, aNand
    gate_params['leakageNand'] = leakageNand
    
    # NOR2
    widthNorN = MIN_NMOS_SIZE * tech.featureSize
    widthNorP = tech.pnSizeRatio * 2 * MIN_NMOS_SIZE * tech.featureSize
    capNorInput = CalculateGateCapacitance(NOR, 2, widthNorN, widthNorP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['capInput']
    capNorOutput = CalculateGateCapacitance(NOR, 2, widthNorN, widthNorP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['capOutput']
    gate_params['widthNorN'], gate_params['widthNorP'] = widthNorN, widthNorP
    gate_params['capNorInput'], gate_params['capNorOutput'] = capNorInput, capNorOutput

    # Tg
    widthTgN = MIN_NMOS_SIZE * tech.featureSize
    widthTgP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize
    resTg = 1/(1/CalculateOnResistance(widthTgN, NMOS, param.temp, tech) + 1/CalculateOnResistance(widthTgP, PMOS, param.temp, tech))
    hTg = CalculateGateArea(INV, 1, widthTgN, widthTgP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['height']
    wTg = CalculateGateArea(INV, 1, widthTgN, widthTgP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['width']
    aTg = CalculateGateArea(INV, 1, widthTgN, widthTgP, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech)['area']
    capTgInput = CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech)['capInput']
    capTgOutput = CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech)['capOutput']
    capTgDrain = capTgOutput
    capTgGateN = CalculateGateCap(widthTgN, tech)
    capTgGateP = CalculateGateCap(widthTgP, tech)
    gate_params['widthTgN'], gate_params['widthTgP'] = widthTgN, widthTgP
    gate_params['resTg'], gate_params['hTg'], gate_params['wTg'], gate_params['aTg'] = resTg, hTg, wTg, aTg
    gate_params['capTgInput'], gate_params['capTgOutput'], gate_params['capTgDrain'] = capTgInput, capTgOutput, capTgDrain
    gate_params['capTgGateN'], gate_params['capTgGateP'] = capTgGateN, capTgGateP

    # Cell
    minCellHeight = MAX_TRANSISTOR_HEIGHT * tech.featureSize
    minCellWidth = 2 * (POLY_WIDTH + MIN_GAP_BET_GATE_POLY) * tech.featureSize
    # RRAM
    resCellAccess = param.resistanceOn * IR_DROP_TOLERANCE
    gate_params['minCellHeight'], gate_params['minCellWidth'] = minCellHeight, minCellWidth
    gate_params['resCellAccess'] = resCellAccess
    
    return gate_params
    
    

# Function bindings
def horowitz(tr, beta, ramp_input)-> dict:
    return FormulaBindings.horowitz(tr, beta, ramp_input)

def CalculateGateArea(gateType:int, numInput:int, widthNMOS:float, widthPMOS:float,
                      heightTransistorRegion:float, tech) -> dict:
    # dict: [area, h, w]
    return FormulaBindings.CalculateGateArea(gateType, numInput, widthNMOS, widthPMOS, heightTransistorRegion, tech)

def CalculateGateCapacitance(gateType, numInput, widthNMOS, widthPMOS, heightTransistorRegion, tech):
    return FormulaBindings.CalculateGateCapacitance(gateType, numInput, widthNMOS, widthPMOS, heightTransistorRegion, tech)

def CalculateGateLeakage(gateType, numInput, widthNMOS, widthPMOS,
                                    temperature, tech):
    return FormulaBindings.CalculateGateLeakage(gateType, numInput, widthNMOS, widthPMOS, temperature, tech)

def CalculateOnResistance(width, type, temp, tech):
    return FormulaBindings.CalculateOnResistance(width, type, temp, tech)

def CalculateDrainCap(width, type, heightTransistorRegion, tech):
    return FormulaBindings.CalculateDrainCap(width, type, heightTransistorRegion, tech)

def CalculateGateCap(width, tech):
    return FormulaBindings.CalculateGateCap(width, tech)

def CalculateTransconductance(width, type, tech):
    return FormulaBindings.CalculateTransconductance(width, type, tech)

