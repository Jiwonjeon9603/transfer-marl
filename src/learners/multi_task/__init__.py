from .odis_learner import ODISLearner
from .updet_learner import UPDeTLearner
from .bc_learner import BCLearner
from .stairs_learner import StairsLearner
from .boot_learner import BootLearner

REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["updet_learner"] = UPDeTLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["stairs_learner"] = StairsLearner
REGISTRY["boot_learner"] = BootLearner