from torch import nn
from torch.autograd import Function


class GradReverseFunc(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class GradientReversal(nn.Module):
    def forward(self, input):
        return GradReverseFunc.apply(input)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input