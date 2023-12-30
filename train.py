import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch


from trainers.llm import LLMTrainer
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually


from lightning.pytorch import seed_everything



hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
# logger = step_csv_logger("out", name, flush_logs_every_n_steps=log_iter_interval)
# wandb_logger = WandbLogger()


# TODO fix lr scheduler
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # TODO add proper config loading from command line
    from configs.homegpt_small import Config

    config = Config()
    seed_everything(config.seed)
    trainer = LLMTrainer(config)
    # TODO add proper training resume handling
    trainer.fit()
