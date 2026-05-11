from etic.core.data_structure.pauli import PauliString
from etic.core.language.qec_gadget import LogicGadget


class PPM(LogicGadget):
    product_to_measure: PauliString
