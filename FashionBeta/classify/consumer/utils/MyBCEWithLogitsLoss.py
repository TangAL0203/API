#-*-coding:utf-8-*-
from torch.autograd import Variable
from torch.nn.modules.module import Module

def binary_cross_entropy_with_logits(input, target, wpos=300.0, wneg=1.0, weight=None, size_average=True):
    r"""Function that measures Binary Cross Entropy between target and output
    logits.
    See :class:`~torch.nn.BCEWithLogitsLoss` for details.
    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``
    Examples::
         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()  #  NXC

    if weight is not None:
        loss = loss * weight

    mask = target*wpos+target.eq(0).float()*wneg
    loss = loss*mask
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


class BCEWithLogitsLoss(Module):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.
    This Binary Cross Entropy between the target and the output logits
    (no sigmoid applied) is:
    ..  :: loss(o, t) = - 1/n \sum_i (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))
    or in the case of the weight argument being specified:
    .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))
    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``
     Shape:
         - Input: :math:`(N, *)` where `*` means, any number of additional
           dimensions
         - Target: :math:`(N, *)`, same shape as the input
     Examples::
         >>> loss = nn.BCEWithLogitsLoss()
         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> output = loss(input, target)
         >>> output.backward()
    """
    def __init__(self, wpos=300.0, wneg=1./300.0, weight=None, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.wpos = wpos
        self.wneg = wneg

    def forward(self, input, target):
        if self.weight is not None:
            return binary_cross_entropy_with_logits(input, target, self.wpos, self.wneg, Variable(self.weight), self.size_average)
        else:
            return binary_cross_entropy_with_logits(input, target, self.wpos, self.wneg, size_average=self.size_average)
