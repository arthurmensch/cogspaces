from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

import numpy as np


class CleverLR(object):
    def __init__(self, model, optimizer,
                 T_max=10,
                 gamma=2):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.model = model
        self.optimizer = optimizer
        self.lrs = list(map(lambda group: group['lr'],
                            optimizer.param_groups))
        self.best_lrs = self.lrs
        self.T_max = T_max
        self.gamma = gamma
        self.phase = 'ramp'
        self.best = 1e12

    def get_lr(self):
        if self.phase == 'ramp':
            return self.lrs
        else:
            return self.cosine_annealing.get_lr()

    def _update_lr(self):
        for param_group, lr in zip(self.optimizer.param_groups,
                                   self.get_lr()):
            param_group['lr'] = lr

    def step(self, metrics=None, at='epoch'):
        if self.phase == 'ramp' and at == 'batch':
            if metrics < self.best:
                self.best_lrs = self.lrs
                self.best = metrics
            if not np.isnan(metrics) and metrics < self.best * 2:
                self.lrs = [self.gamma * lr for lr in self.lrs]
                self._update_lr()
                print('LR', self.get_lr())

            else:
                print('Found learning rate, switching to cosine mode')
                self.lrs = [lr for lr in self.best_lrs]
                self._update_lr()
                self.model.reset_parameters()
                print('LR', self.get_lr())
                self.phase = 'cosine'
                self.cosine_annealing = CosineAnnealingLR(
                    self.optimizer, self.T_max, eta_min=min(self.lrs) * 1e-3,
                    last_epoch=-1)

        elif self.phase == 'cosine' and at == 'epoch':
            # lr is updated with cosine_annealing
            self.cosine_annealing.step()


class LinearRamp(ExponentialLR):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, gamma):
        self.num_bad_batch = 0
        self.best = 1e9
        super(LinearRamp, self).__init__(optimizer, gamma, -1)

    def step(self, metrics):
        current = metrics
        if current < metrics:
            self.best = current
            self.best_lrs = self.get_lrs()
            self.num_bad_batch = 0
        else:
            self.num_bad_epochs += 1
