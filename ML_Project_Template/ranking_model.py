from numpy.core.numeric import NaN
import torch
from global_configs import INPUT_DIM
from torch.nn import ReLU, Sigmoid

class FeedForwardSiamese(torch.nn.Module):
    def __init__(self, args):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FeedForwardSiamese, self).__init__()

        self.linear1 = torch.nn.Linear(INPUT_DIM, args.ff_hidden_dim)
        self.linear2 = torch.nn.Linear(args.ff_hidden_dim, 1)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.ff2 = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, args.ff_hidden_dim),
            ReLU(),
            torch.nn.Linear(args.ff_hidden_dim, 256),
            ReLU(),
            torch.nn.Linear(256, 1),
            ReLU()
        )
        self.ff1 = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, args.ff_hidden_dim),
            ReLU(),
            torch.nn.Linear(args.ff_hidden_dim, 1),
            ReLU()
        )

    def forward(self, x1, x2):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        '''
        x1l1_logit = self.linear1(x1)
        x1l1 = self.relu(x1l1_logit)
        x1l2_logit = self.linear2(x1l1)
        x1l2 = self.relu(x1l2_logit)

        x2l1_logit = self.linear1(x2)
        x2l1 = self.relu(x2l1_logit)
        x2l2_logit = self.linear2(x2l1)
        x2l2 = self.relu(x2l2_logit)

        y_pred = self.sigmoid(x1l2-x2l2)
        '''
        
        x1 = self.ff1(x1)
        x2 = self.ff1(x2)

        #y_pred = self.sigmoid(x1-x2)
        return x1, x2