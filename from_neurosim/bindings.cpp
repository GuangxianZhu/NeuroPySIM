#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "formula.h"
#include "Technology.h"
#include "Param.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(FormulaBindings, m) {
    m.doc() = "Bindings for formula.cpp";

    // Expose DeviceRoadmap enum
    py::enum_<DeviceRoadmap>(m, "DeviceRoadmap")
        .value("HP", HP)
        .value("LSTP", LSTP)
        .export_values();

    // Expose TransistorType enum
    py::enum_<TransistorType>(m, "TransistorType")
        .value("conventional", conventional)
        .value("FET_2D", FET_2D)
        .value("TFET", TFET)
        .export_values();

    // Expose CalculateGateCap function
    m.def("CalculateGateCap", &CalculateGateCap, "Calculate MOSFET gate capacitance");

    // Expose CalculateGateArea function
    m.def("CalculateGateArea", [](int gateType, int numInput, double widthNMOS, double widthPMOS,
                                  double heightTransistorRegion, Technology& tech) {
        double height, width;
        double area = CalculateGateArea(gateType, numInput, widthNMOS, widthPMOS,
                                        heightTransistorRegion, tech, &height, &width);
        return py::dict("area"_a = area, "height"_a = height, "width"_a = width);
    }, "Calculate layout area and dimensions of a logic gate");

    // Expose CalculateGateCapacitance
    m.def("CalculateGateCapacitance", [](int gateType, int numInput,
                                     double widthNMOS, double widthPMOS,
                                     double heightTransistorRegion, Technology& tech) {
    double capInput = 0, capOutput = 0;
        CalculateGateCapacitance(gateType, numInput, widthNMOS, widthPMOS, heightTransistorRegion, tech, &capInput, &capOutput);
        return py::dict("capInput"_a = capInput, "capOutput"_a = capOutput);
    }, "Calculate input and output gate capacitances");

    // Expose CalculateGateLeakage
    m.def("CalculateGateLeakage", [](int gateType, int numInput, double widthNMOS, double widthPMOS,
                                    double temperature, Technology& tech) {
        double leakage = CalculateGateLeakage(gateType, numInput, widthNMOS, widthPMOS, temperature, tech);
        return leakage;
    }, "Calculate the gate leakage current");

    // horowitz
    m.def("horowitz", [](double tr, double beta, double rampInput) {
        double rampOutput = 0.0; // 用于保存 rampOutput 的值
        double result = horowitz(tr, beta, rampInput, &rampOutput);
        return py::dict("result"_a = result, "rampOutput"_a = rampOutput);
    }, "Calculate the delay and ramp output using the Horowitz delay model");

    // CalculateTransconductance
    m.def("CalculateTransconductance", [](double width, int type, Technology& tech) {
        double gm = CalculateTransconductance(width, type, tech);
        return gm;
    }, "Calculate the transconductance based on width, transistor type, and technology parameters");

    // OnResistance
    m.def("CalculateOnResistance", [](double width, int type, double temperature, Technology& tech) {
        double resistance = CalculateOnResistance(width, type, temperature, tech);
        return resistance;
    }, "Calculate the on-resistance based on width, transistor type, temperature, and technology parameters");

    m.def("CalculateOnResistance_normal", [](double width, int type, double temperature, Technology& tech) {
        double resistance = CalculateOnResistance_normal(width, type, temperature, tech);
        return resistance;
    }, "Calculate the on-resistance based on width, transistor type, temperature, and technology parameters");

    m.def("CalculateDrainCap", [](double width, int type, double heightTransistorRegion, Technology& tech) {
        double drainCap = CalculateDrainCap(width, type, heightTransistorRegion, tech);
        return drainCap;
    }, "Calculate the drain capacitance based on width, transistor type, and transistor region height");


    // Expose Technology class
    py::class_<Technology>(m, "Technology")
        .def(py::init<>())
        // functions
        .def("Initialize", &Technology::Initialize)
        .def("PrintProperty", &Technology::PrintProperty)
        // member variables
        .def_readwrite("initialized", &Technology::initialized)
        .def_readwrite("featureSizeInNano", &Technology::featureSizeInNano)
        .def_readwrite("featureSize", &Technology::featureSize)
        .def_readwrite("RRAMFeatureSize", &Technology::RRAMFeatureSize)
        .def_readwrite("vdd", &Technology::vdd)
        .def_readwrite("vth", &Technology::vth)
        .def_readwrite("heightFin", &Technology::heightFin)
        .def_readwrite("widthFin", &Technology::widthFin)
        .def_readwrite("PitchFin", &Technology::PitchFin)
        .def_readwrite("phyGateLength", &Technology::phyGateLength)
        .def_readwrite("capIdealGate", &Technology::capIdealGate)
        .def_readwrite("capFringe", &Technology::capFringe)
        .def_readwrite("capJunction", &Technology::capJunction)
        .def_readwrite("capOverlap", &Technology::capOverlap)
        .def_readwrite("capSidewall", &Technology::capSidewall)
        .def_readwrite("capDrainToChannel", &Technology::capDrainToChannel)
        .def_readwrite("buildInPotential", &Technology::buildInPotential)
        .def_readwrite("pnSizeRatio", &Technology::pnSizeRatio)
        .def_readwrite("current_gmNmos", &Technology::current_gmNmos)
        .def_readwrite("current_gmPmos", &Technology::current_gmPmos)
        .def_readwrite("capPolywire", &Technology::capPolywire)
        .def_readwrite("max_sheet_num", &Technology::max_sheet_num)
        .def_readwrite("thickness_sheet", &Technology::thickness_sheet)
        .def_readwrite("width_sheet", &Technology::width_sheet)
        .def_readwrite("effective_width", &Technology::effective_width)
        .def_readwrite("max_fin_num", &Technology::max_fin_num)
        .def_readwrite("max_fin_per_GAA", &Technology::max_fin_per_GAA)
        .def_readwrite("gm_oncurrent", &Technology::gm_oncurrent)
        .def_readwrite("cap_draintotal", &Technology::cap_draintotal);

        
    // 绑定 Param 类
    py::class_<Param>(m, "Param")
        .def(py::init<>())
        .def_readwrite("operationmode", &Param::operationmode)
        .def_readwrite("operationmodeBack", &Param::operationmodeBack)
        .def_readwrite("memcelltype", &Param::memcelltype)
        .def_readwrite("accesstype", &Param::accesstype)
        .def_readwrite("transistortype", &Param::transistortype)
        .def_readwrite("deviceroadmap", &Param::deviceroadmap)
        .def_readwrite("heightInFeatureSizeSRAM", &Param::heightInFeatureSizeSRAM)
        .def_readwrite("widthInFeatureSizeSRAM", &Param::widthInFeatureSizeSRAM)
        .def_readwrite("widthSRAMCellNMOS", &Param::widthSRAMCellNMOS)
        .def_readwrite("widthSRAMCellPMOS", &Param::widthSRAMCellPMOS)
        .def_readwrite("widthAccessCMOS", &Param::widthAccessCMOS)
        .def_readwrite("minSenseVoltage", &Param::minSenseVoltage)
        .def_readwrite("heightInFeatureSize1T1R", &Param::heightInFeatureSize1T1R)
        .def_readwrite("widthInFeatureSize1T1R", &Param::widthInFeatureSize1T1R)
        .def_readwrite("heightInFeatureSizeCrossbar", &Param::heightInFeatureSizeCrossbar)
        .def_readwrite("widthInFeatureSizeCrossbar", &Param::widthInFeatureSizeCrossbar)
        .def_readwrite("relaxArrayCellHeight", &Param::relaxArrayCellHeight)
        .def_readwrite("relaxArrayCellWidth", &Param::relaxArrayCellWidth)
        .def_readwrite("globalBusType", &Param::globalBusType)
        .def_readwrite("globalBufferType", &Param::globalBufferType)
        .def_readwrite("tileBufferType", &Param::tileBufferType)
        .def_readwrite("peBufferType", &Param::peBufferType)
        .def_readwrite("chipActivation", &Param::chipActivation)
        .def_readwrite("reLu", &Param::reLu)
        .def_readwrite("novelMapping", &Param::novelMapping)
        .def_readwrite("pipeline", &Param::pipeline)
        .def_readwrite("SARADC", &Param::SARADC)
        .def_readwrite("currentMode", &Param::currentMode)
        .def_readwrite("validated", &Param::validated)
        .def_readwrite("synchronous", &Param::synchronous)
        .def_readwrite("globalBufferCoreSizeRow", &Param::globalBufferCoreSizeRow)
        .def_readwrite("globalBufferCoreSizeCol", &Param::globalBufferCoreSizeCol)
        .def_readwrite("tileBufferCoreSizeRow", &Param::tileBufferCoreSizeRow)
        .def_readwrite("tileBufferCoreSizeCol", &Param::tileBufferCoreSizeCol)
        .def_readwrite("clkFreq", &Param::clkFreq)
        .def_readwrite("featuresize", &Param::featuresize)
        .def_readwrite("readNoise", &Param::readNoise)
        .def_readwrite("resistanceOn", &Param::resistanceOn)
        .def_readwrite("resistanceOff", &Param::resistanceOff)
        .def_readwrite("maxConductance", &Param::maxConductance)
        .def_readwrite("minConductance", &Param::minConductance)
        .def_readwrite("temp", &Param::temp)
        .def_readwrite("technode", &Param::technode)
        .def_readwrite("wireWidth", &Param::wireWidth)
        .def_readwrite("multipleCells", &Param::multipleCells)
        .def_readwrite("maxNumLevelLTP", &Param::maxNumLevelLTP)
        .def_readwrite("maxNumLevelLTD", &Param::maxNumLevelLTD)
        .def_readwrite("readVoltage", &Param::readVoltage)
        .def_readwrite("readPulseWidth", &Param::readPulseWidth)
        .def_readwrite("writeVoltage", &Param::writeVoltage)
        .def_readwrite("accessVoltage", &Param::accessVoltage)
        .def_readwrite("resistanceAccess", &Param::resistanceAccess)
        .def_readwrite("nonlinearIV", &Param::nonlinearIV)
        .def_readwrite("nonlinearity", &Param::nonlinearity)
        .def_readwrite("writePulseWidth", &Param::writePulseWidth)
        .def_readwrite("numWritePulse", &Param::numWritePulse)
        .def_readwrite("globalBusDelayTolerance", &Param::globalBusDelayTolerance)
        .def_readwrite("localBusDelayTolerance", &Param::localBusDelayTolerance)
        .def_readwrite("treeFoldedRatio", &Param::treeFoldedRatio)
        .def_readwrite("maxGlobalBusWidth", &Param::maxGlobalBusWidth)
        .def_readwrite("algoWeightMax", &Param::algoWeightMax)
        .def_readwrite("algoWeightMin", &Param::algoWeightMin)
        .def_readwrite("neuro", &Param::neuro)
        .def_readwrite("multifunctional", &Param::multifunctional)
        .def_readwrite("parallelWrite", &Param::parallelWrite)
        .def_readwrite("parallelRead", &Param::parallelRead)
        .def_readwrite("numlut", &Param::numlut)
        .def_readwrite("numColMuxed", &Param::numColMuxed)
        .def_readwrite("numWriteColMuxed", &Param::numWriteColMuxed)
        .def_readwrite("levelOutput", &Param::levelOutput)
        .def_readwrite("avgWeightBit", &Param::avgWeightBit)
        .def_readwrite("numBitInput", &Param::numBitInput)
        .def_readwrite("numRowSubArray", &Param::numRowSubArray)
        .def_readwrite("numColSubArray", &Param::numColSubArray)
        .def_readwrite("cellBit", &Param::cellBit)
        .def_readwrite("synapseBit", &Param::synapseBit)
        .def_readwrite("speedUpDegree", &Param::speedUpDegree)
        .def_readwrite("XNORparallelMode", &Param::XNORparallelMode)
        .def_readwrite("XNORsequentialMode", &Param::XNORsequentialMode)
        .def_readwrite("BNNparallelMode", &Param::BNNparallelMode)
        .def_readwrite("BNNsequentialMode", &Param::BNNsequentialMode)
        .def_readwrite("conventionalParallel", &Param::conventionalParallel)
        .def_readwrite("conventionalSequential", &Param::conventionalSequential)
        .def_readwrite("numRowPerSynapse", &Param::numRowPerSynapse)
        .def_readwrite("numColPerSynapse", &Param::numColPerSynapse)
        .def_readwrite("AR", &Param::AR)
        .def_readwrite("Rho", &Param::Rho)
        .def_readwrite("wireLengthRow", &Param::wireLengthRow)
        .def_readwrite("wireLengthCol", &Param::wireLengthCol)
        .def_readwrite("unitLengthWireResistance", &Param::unitLengthWireResistance)
        .def_readwrite("wireResistanceRow", &Param::wireResistanceRow)
        .def_readwrite("wireResistanceCol", &Param::wireResistanceCol)
        .def_readwrite("alpha", &Param::alpha)
        .def_readwrite("beta", &Param::beta)
        .def_readwrite("gamma", &Param::gamma)
        .def_readwrite("delta", &Param::delta)
        .def_readwrite("epsilon", &Param::epsilon)
        .def_readwrite("zeta", &Param::zeta)
        .def_readwrite("speciallayout", &Param::speciallayout)
        .def_readwrite("unitcap", &Param::unitcap)
        .def_readwrite("unitres", &Param::unitres)
        .def_readwrite("drivecapin", &Param::drivecapin)
        .def_readwrite("dumcolshared", &Param::dumcolshared)
        .def_readwrite("columncap", &Param::columncap)
        .def_readwrite("arrayheight", &Param::arrayheight)
        .def_readwrite("arraywidthunit", &Param::arraywidthunit)
        .def_readwrite("resCellAccess", &Param::resCellAccess)
        .def_readwrite("inputtoggle", &Param::inputtoggle)
        .def_readwrite("outputtoggle", &Param::outputtoggle)
        .def_readwrite("ADClatency", &Param::ADClatency)
        .def_readwrite("rowdelay", &Param::rowdelay)
        .def_readwrite("muxdelay", &Param::muxdelay)
        .def_readwrite("technologynode", &Param::technologynode)
        .def_readwrite("numRowParallel", &Param::numRowParallel)
        .def_readwrite("totaltile_num", &Param::totaltile_num)
        .def_readwrite("sync_data_transfer", &Param::sync_data_transfer)

        .def_readwrite("sizingfactor_MUX", &Param::sizingfactor_MUX)
        .def_readwrite("sizingfactor_WLdecoder", &Param::sizingfactor_WLdecoder)
        .def_readwrite("newswitchmatrixsizeratio", &Param::newswitchmatrixsizeratio)
        .def_readwrite("switchmatrixsizeratio", &Param::switchmatrixsizeratio)
        .def_readwrite("buffernumber", &Param::buffernumber)
        .def_readwrite("buffersizeratio", &Param::buffersizeratio)

        .def_readwrite("Metal0", &Param::Metal0)
        .def_readwrite("Metal1", &Param::Metal1)
        .def_readwrite("AR_Metal0", &Param::AR_Metal0)
        .def_readwrite("AR_Metal1", &Param::AR_Metal1)
        .def_readwrite("Rho_Metal0", &Param::Rho_Metal0)
        .def_readwrite("Rho_Metal1", &Param::Rho_Metal1)
        .def_readwrite("Metal0_unitwireresis", &Param::Metal0_unitwireresis)
        .def_readwrite("Metal1_unitwireresis", &Param::Metal1_unitwireresis);
}
