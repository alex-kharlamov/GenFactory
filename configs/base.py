from dataclasses import dataclass
from typing import Literal, Optional
from .data import Data

@dataclass
class Optimization:
    precision: Optional[Literal['bf16', 'fp32']] = None

    global_batch_size: int = 512
    learning_rate: float = 4e-4
    micro_batch_size: int = 8
    max_step: int = 715256 * 2
    warmup_steps: int = 2000

    weight_decay: float = 1e-1
    beta1:float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    min_lr: float = 4e-5


@dataclass
class Config:
    trainer: Literal['LLMTrainer'] = 'LLMTrainer'
    model_name: str = 'pythia-160m'

    optimization: Optimization = Optimization()
    data: Data = Data()

    num_of_devices: int = 1

    log_step_interval: int = 10
    eval_iters: int = 500
    save_step_interval: int = 5000
    eval_step_interval: int = 5000
    seed: int = 42


