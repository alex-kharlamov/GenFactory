import lightning as L
import time
from functools import partial
from models.utils import num_parameters
from models.internals.lit_gpt import GPT as LitGPT
import torch
from configs.base import Optimization
from .utils import chunked_cross_entropy
from utils.speed_monitor import estimate_flops
from losses.cross_entropy import FusedCrossEntropyLoss
import math


class GPT(L.LightningModule):
    def __init__(self, config, optim_config: Optimization):
        super().__init__()
        self.config = config
        self.optim_config = optim_config

        self.dummy_world_size = 1 #TODO: add correct world size handling

        with torch.device("meta"):
            meta_model = LitGPT(self.config)
            # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
            # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
            # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
            estimated_flops = estimate_flops(meta_model) * self.optim_config.micro_batch_size
            print(f"Estimated TFLOPs: {estimated_flops * self.dummy_world_size / 1e12:.2f}")
            x = torch.randint(0, 1, (self.optim_config.micro_batch_size, self.config.block_size))
            # measured_flos run in meta. Will trigger fusedRMSNorm error
            #measured_flops = measure_flops(meta_model, x)
            #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
            del meta_model, x

        print(f"Loading model with {config.__dict__}")
        t0 = time.perf_counter()
        self.model = LitGPT(config)
        self.model.apply(partial(self.model._init_weights,
                                  n_layer=self.config.n_layer))
        print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        print(f"Total parameters {num_parameters(self.model):,}")

        self.loss = FusedCrossEntropyLoss()



    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        #TODO fix train_iter lr calculation
        # # determine and set the learning rate for this iteration
        # lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr


        input_ids = batch[:, 0 : self.config.block_size].contiguous()
        targets = batch[:, 1 : self.config.block_size + 1].contiguous()
        logits = self.model(input_ids)
        loss = self.loss(logits, targets)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_tokens", self.config.block_size * (batch_idx + 1) * input_ids.shape[0] * self.dummy_world_size / 1e9, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # TODO fix speed monitor
        # monitor.on_train_batch_end(
        #     state["iter_num"] * micro_batch_size,
        #     t1 - total_t0,
        #     # this assumes that device FLOPs are the same and that all devices have the same batch size
        #     fabric.world_size,
        #     state["step_count"],
        #     flops_per_batch=estimated_flops,
        #     lengths=total_lengths,
        #     train_loss = loss.item()
        # )
        return loss
        

    def validation_step(self, batch, batch_idx):
        input_ids = batch[:, 0 : self.config.block_size].contiguous()
        targets = batch[:, 1 : self.config.block_size + 1].contiguous()

        logits = self.model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
        
        self.log("val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_ppl", math.exp(loss.item()), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
        self.model.parameters(), lr=self.optim_config.learning_rate,
          weight_decay=self.optim_config.weight_decay,
            betas=(self.optim_config.beta1,
                   self.optim_config.beta2), foreach=False)
        scheduler = None
        return optimizer