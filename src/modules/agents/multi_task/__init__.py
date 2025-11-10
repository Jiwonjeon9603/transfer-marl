from .odis_agent import ODISAgent
from .updet_agent import UPDeTAgent
from .bc_agent import BCAgent
from .bcr_agent import BCRAgent
from .moe_agent import MoEAgent

REGISTRY = {}

REGISTRY["mt_odis"] = ODISAgent
REGISTRY["mt_updet"] = UPDeTAgent
REGISTRY["mt_bc"] = BCAgent
REGISTRY["mt_bcr"] = BCRAgent
REGISTRY["mt_moe"] = MoEAgent
