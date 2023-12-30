from typing import Optional
from configs.base import Config
from lightning.fabric.strategies import XLAStrategy, FSDPStrategy
from lightning.pytorch import Trainer as seed_everything
from models.internals.lit_gpt import Block
from models.utils import get_default_supported_precision


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        tpu = False # TODO: add proper tpu handling
        self.precision = self.config.optimization.precision or get_default_supported_precision(training=True, tpu=tpu)

        if self.config.num_of_devices > 1:
            if tpu:
                # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
                self.devices = "auto"
                self.strategy = XLAStrategy(sync_module_states=False)
            else:
                self.strategy = FSDPStrategy(
                    auto_wrap_policy={Block},
                    activation_checkpointing_policy=None,
                    state_dict_type="full",
                    limit_all_gathers=True,
                    cpu_offload=False,
                )
        else:
            self.strategy = "auto"
        

    def _create_dataloaders(self, ):
        raise NotImplementedError


    def fit(self):
        assert self.train_dataloader is not None, "Please create dataloaders first" # TODO add proper assert
        assert self.val_dataloader is not None, "Please create dataloaders first"
        assert self.model is not None, "Please create model first"
        assert self.lightning_trainer is not None, "Please create lightning trainer first"
        self.lightning_trainer.fit(self.model, self.train_dataloader, self.val_dataloader)
