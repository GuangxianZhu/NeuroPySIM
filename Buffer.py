import math
from MemCell import MemCell
from constant import *
from from_neurosim.build import FormulaBindings
from Gate_calculator import horowitz

from Decoder import RowDecoder
from DecoderDriver import DecoderDriver
from Mux import Mux
from MultilevelSenseAmp import MultilevelSenseAmp
from Adder import Adder
from DFF import DFF
from SwitchMatrix import SwitchMatrix
from ShiftAdd import ShiftAdd

class Buffer:
    def __init__(self, numBit, num_interface, SRAM:bool,
                 param, tech, gate_params):
        self.numBit = numBit