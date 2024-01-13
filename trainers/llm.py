import lightning as L
import torch
from models.internals.model_setups import Config as GPTConfig
import time
from models.GPT import GPT
from functools import partial
from models.utils import num_parameters
from configs.base import Config
from .base import Trainer as BaseTrainer
from data.dataloaders import create_dataloaders
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import os


class LLMTrainer(BaseTrainer):
    def __init__(self, config: Config):
        super().__init__(config=config)
        self.config = config

        self.model_config = GPTConfig.from_name(self.config.model_name)
        self.model = GPT(self.model_config, self.config.optimization)

        batch_size = self.config.optimization.global_batch_size // self.config.num_of_devices
        gradient_accumulation_steps = batch_size // self.config.optimization.micro_batch_size
        assert gradient_accumulation_steps > 0
        warmup_iters = self.config.optimization.warmup_steps * gradient_accumulation_steps

        max_iters = self.config.optimization.max_step * gradient_accumulation_steps
        lr_decay_iters = max_iters
        log_iter_interval = self.config.log_step_interval * gradient_accumulation_steps

        # TODO: implement proper dataloader state save\resume handling
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.config.data.checkpoint_dir, self.config.config_name),
            every_n_train_steps=self.config.save_step_interval // gradient_accumulation_steps,
            verbose=True,
            )

        self.tb_logger = TensorBoardLogger(self.config.data.log_dir, name=self.config.config_name)

        self.lightning_trainer = LightningTrainer(accumulate_grad_batches=gradient_accumulation_steps,
                                                  gradient_clip_val=self.config.optimization.grad_clip,
                                                  val_check_interval=self.config.eval_step_interval,
                                                  logger=self.tb_logger,
                                                  callbacks=[checkpoint_callback],
                                                  strategy=self.strategy,
                                                  limit_val_batches=self.config.eval_iters,
                                                  )
        self._create_dataloaders()

    def _create_dataloaders(self, ):
        self.train_dataloader, self.val_dataloader = create_dataloaders(
            batch_size=self.config.optimization.micro_batch_size,
            block_size=self.model_config.block_size,
            train_data_dir=self.config.data.dataset.train_data_dir,
            val_data_dir=self.config.data.dataset.val_data_dir,
            seed=self.config.seed,
        )
