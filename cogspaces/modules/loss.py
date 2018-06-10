import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple


class MultiStudyLoss(nn.Module):
    def __init__(self, study_weights: Dict[str, float],
                 ) -> None:
        super().__init__()
        self.study_weights = study_weights

    def forward(self, preds: Dict[str, torch.FloatTensor],
                targets: Dict[str, torch.LongTensor]) \
            -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        loss = 0
        for study in preds:
            pred = preds[study]
            target = targets[study]
            this_loss = F.nll_loss(pred, target, size_average=True)
            loss += this_loss * self.study_weights[study]
        return loss