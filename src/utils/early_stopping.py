from typing import Dict, Union, List
import numpy as np

import torch


class EarlyStopping:

    def __init__(self, patience: int, max_epochs: int = 10000):
        self.abort = False
        self.patience = patience
        self.model_state_dict = None
        self.model_params = None
        self.max_epochs = max_epochs
        self.curr_step = 0

        self.losses = list()

    def update(self,
               new_value,
               model: torch.nn.Module = None,
               model_params: Union[Dict, torch.Tensor, List] = None):

        self.losses.append(new_value)

        if self.curr_step <= self.patience or new_value <= np.mean(self.losses[-(self.patience + 1):-1]):
            if model is not None:
                self.model_state_dict = model.state_dict()
            if model_params is not None:
                self.model_params = model_params
        else:
            self.abort = True
        if self.curr_step is not None and self.curr_step >= self.max_epochs:
            self.abort = True

        self.curr_step = self.curr_step + 1

    def best_model_state_dict(self):
        return self.model_state_dict
