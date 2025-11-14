REGISTRY = {}

from .sc2_decomposer import SC2Decomposer
REGISTRY["sc2"] = SC2Decomposer

from .cn_decomposer import MPEDecomposer
REGISTRY["grid_mpe"] = MPEDecomposer

from .sc2v2_decomposer import SC2V2Decomposer
REGISTRY["sc2v2"] = SC2V2Decomposer