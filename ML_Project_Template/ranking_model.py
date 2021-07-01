from numpy.core.numeric import NaN
import torch
from global_configs import INPUT_DIM

class FeedForwardSiamese(torch.nn.Module):
    def __init__(self, args):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FeedForwardSiamese, self).__init__()

        self.linear1 = torch.nn.Linear(INPUT_DIM, args.ff_hidden_dim)
        self.linear2 = torch.nn.Linear(args.ff_hidden_dim, 1)

    def forward(self, x1, x2):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        print(x1.shape)
        print(x2.shape)
        return NaN