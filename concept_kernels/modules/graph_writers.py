import json
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class GraphWriterBase:
    def __init__(self, config):
        self.config = config

    def write_scalar(self, name, value, step=None):
        raise NotImplementedError()


class TensorboardGraphWriter(GraphWriterBase):
    def __init__(self, config):
        super().__init__(config)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def write_scalar(self, name, value, step=None):
        self.writer.add_scalar(tag=name, scalar_value=value, global_step=step)
        # self.writer.flush()


class WandBGraphWriter(GraphWriterBase):
    def __init__(self, config):
        super().__init__(config)

    def write_scalar(self, name, value, step=None):
        raise NotImplementedError()
