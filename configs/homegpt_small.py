from dataclasses import dataclass
from typing import Literal
from .base import Optimization
from .base import Config as BaseConfig

@dataclass
class Config(BaseConfig):
    config_name: str = 'homegpt_small'
    model_name: str = 'pythia-160m'

    num_of_devices: int = 1

