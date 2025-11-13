import os
import sys

from functools import partial
from .multiagentenv import MultiAgentEnv
from .grid_mpe import GridMPEEnv

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def __check_and_prepare_smac_kwargs(kwargs):
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


REGISTRY = {}
REGISTRY["grid_mpe"] = partial(env_fn, env=GridMPEEnv)

# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
