from .odis_learner import ODISLearner
from .updet_learner import UPDeTLearner
from .bc_learner import BCLearner
from .stairs_learner import StairsLearner
from .updet_learner_bc import UPDeTLearnerBC


REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["updet_learner"] = UPDeTLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["stairs_learner"] = StairsLearner
REGISTRY["updet_learner_bc"] = UPDeTLearnerBC
