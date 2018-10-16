import math

import torch
from torch import nn
from torch.nn import Parameter, functional as F

k1 = 0.63576
k2 = 1.87320
k3 = 1.48695


class DropoutLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=1e-8,
                 level='layer', var_penalty=0., adaptive=False,
                 sparsify=False):
        super().__init__(in_features, out_features, bias)

        self.p = p
        self.var_penalty = var_penalty

        if level == 'layer':
            self.log_alpha = Parameter(torch.Tensor(1, 1),
                                       requires_grad=adaptive)

        elif level == 'atom':
            self.log_alpha = Parameter(torch.Tensor(1, in_features),
                                       requires_grad=adaptive)
        elif level == 'coef':
            self.log_alpha = Parameter(torch.Tensor(out_features, in_features),
                                       requires_grad=adaptive)
        elif level == 'additive':
            assert adaptive
            self.log_sigma2 = Parameter(
                torch.Tensor(out_features, in_features),
                requires_grad=True)
        else:
            raise ValueError()

        self.sparsify = sparsify
        self.adaptive = adaptive
        self.level = level

        self.reset_dropout()

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'level'):
            self.reset_dropout()

    def reset_dropout(self):
        if self.p > 0:
            log_alpha = math.log(self.p) - math.log(1 - self.p)

            if self.level != 'additive':
                self.log_alpha.data.fill_(log_alpha)
            else:
                self.log_sigma2.data = log_alpha + torch.log(
                    self.weight.data ** 2 + 1e-8)

    def make_additive(self):
        assert self.level != 'additive'
        self.log_alpha.requires_grad = False
        self.level = 'additive'
        self.adaptive = True
        out_features, in_features = self.weight.shape
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features),
                                    requires_grad=True)
        self.log_sigma2.data = (self.log_alpha.expand(*self.weight.shape)
                                + torch.log(self.weight ** 2 + 1e-8)
                                ).detach()

        self.log_alpha.requires_grad = False

    def make_non_adaptive(self):
        assert self.level != 'additive'
        self.adaptive = False
        self.log_alpha.requires_grad = False

    def make_adaptive(self):
        assert self.level != 'additive'
        self.adaptive = True
        self.log_alpha.requires_grad = True

    def get_var_weight(self):
        if self.level == 'additive':
            return torch.exp(self.log_sigma2)
            # return self.sigma ** 2
        else:
            return torch.exp(self.log_alpha) * self.weight ** 2

    def get_log_alpha(self):
        if self.level == 'additive':
            return torch.clamp(
                self.log_sigma2 - torch.log(self.weight ** 2 + 1e-8), -8, 8)
        else:
            return torch.clamp(self.log_alpha, -8, 8)

    def get_dropout(self):
        return 1 / (1 + torch.exp(-self.get_log_alpha())).squeeze().detach()

    def forward(self, input):
        if self.training:
            if self.p == 0:
                return F.linear(input, self.weight, self.bias)
            if self.adaptive:
                output = F.linear(input, self.weight, self.bias)
                # Local reparemtrization trick: gaussian latent_dropout noise on input
                # <-> gaussian noise on output
                std = torch.sqrt(
                    F.linear(input ** 2, self.get_var_weight(), None) + 1e-8)
                eps = torch.randn_like(output, requires_grad=False)
                return output + std * eps
            else:
                eps = torch.randn_like(input, requires_grad=False)
                input = input * (
                        1 + torch.exp(.5 * self.get_log_alpha()) * eps)
                return F.linear(input, self.weight, self.bias)
        else:
            if self.sparsify:
                weight = self.sparse_weight
            else:
                weight = self.weight
            return F.linear(input, weight, self.bias)

    def penalty(self):
        if not self.adaptive or self.var_penalty == 0:
            return torch.tensor(0., device=self.weight.device,
                                dtype=torch.float)
        else:
            log_alpha = self.get_log_alpha()
            var_penalty = - k1 * (torch.sigmoid(k2 + k3 * log_alpha)
                                  - .5 * F.softplus(-log_alpha)
                                  - 1).expand(*self.weight.shape).sum()
            return var_penalty * self.var_penalty

    @property
    def density(self):
        return (self.sparse_weight != 0).float().mean().item()

    @property
    def sparse_weight(self):
        mask = self.get_log_alpha() > 3
        return self.weight.masked_fill(mask, 0)