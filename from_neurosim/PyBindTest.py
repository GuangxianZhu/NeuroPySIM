from build import FormulaBindings

# 创建一个 Technology 实例
tech = FormulaBindings.Technology()
tech.Initialize(45, FormulaBindings.DeviceRoadmap.LSTP, 
                FormulaBindings.TransistorType.conventional)  # Initialize with feature size=22nm, device roadmap=LSTP, transistor type=conventional
print(f"Feature size: {tech.featureSizeInNano}")
print(tech.vdd)

# test params.h
param = FormulaBindings.Param()
print(param.sizingfactor_MUX)

# 调用 CalculateGateCap, 直接计算栅极电容
width = 10.0  # 栅宽
gate_cap = FormulaBindings.CalculateGateCap(width, tech)
print(f"Gate Capacitance: {gate_cap:.4e} F")

# CalculateGateCapacitance, 复杂逻辑门的电容计算
gate_type = 0  # INV
num_input = 2
width_nmos = 1.5
width_pmos = 3.0
height_transistor_region = 10.0
result = FormulaBindings.CalculateGateCapacitance(
    gate_type, num_input, width_nmos, width_pmos, height_transistor_region, tech
)
print(f"Input Capacitance: {result['capInput']} F")
print(f"Output Capacitance: {result['capOutput']} F")

# 调用 CalculateGateArea
gate_type = 0  # INV
num_input = 2
width_nmos = 1.5
width_pmos = 3.0
height_transistor_region = 10.0

result = FormulaBindings.CalculateGateArea(
    gate_type, num_input, width_nmos, width_pmos, height_transistor_region, tech
)

print(f"Gate Area: {result['area']} µm²")
print(f"Gate Height: {result['height']} µm")
print(f"Gate Width: {result['width']} µm")

# leakage
temperature = 300.0
leakage = FormulaBindings.CalculateGateLeakage(
    gate_type, num_input, width_nmos, width_pmos, temperature, tech
)
print(f"Leakage: {leakage:.4e} A")

# horowitz
tr = 5e-9          # 传输延迟 (s)
beta = 0.5         # 默认值为 0.5
ramp_input = 1e9   # 输入斜率 (V/s)

result = FormulaBindings.horowitz(tr, beta, ramp_input)
delay = result['result']
ramp_output = result['rampOutput']
print(f"Delay: {delay:.6e} s")
print(f"Ramp Output: {ramp_output:.6e} V/s")


# 调用 CalculateTransconductance
width = 1.0  # 晶体管宽度 (µm)
type = FormulaBindings.TransistorType.conventional
gm = FormulaBindings.CalculateTransconductance(width, type, tech)
print(f"Transconductance (gm): {gm:.6e} S")

# 调用 CalculateOnResistance
resistance = FormulaBindings.CalculateOnResistance(width, type, temperature, tech)
print(f"On-Resistance: {resistance:.6e} Ω")

# 调用 CalculateDrainCap
type=0 # NMOS
drain_cap = FormulaBindings.CalculateDrainCap(width, type, height_transistor_region, tech)
print(f"Drain Capacitance: {drain_cap:.6e} F")
