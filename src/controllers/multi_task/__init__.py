from .mt_odis_controller import ODISMAC
from .mt_updet_controller import UPDeTMAC
from .mt_bc_controller import BCMAC
from .mt_bcr_controller import BCRMAC
from .mt_stairs_controller import StairsMac
from .mt_boot_controller import BootMAC

REGISTRY = {}

REGISTRY["mt_odis_mac"] = ODISMAC
REGISTRY["mt_updet_mac"] = UPDeTMAC
REGISTRY["mt_bc_mac"] = BCMAC
REGISTRY["mt_bcr_mac"] = BCRMAC
REGISTRY["mt_stairs_mac"] = StairsMac
REGISTRY["mt_boot_mac"] = BootMAC