from numpy.core.numeric import NaN
import torch
from global_configs import INPUT_DIM
from torch.nn import ReLU, Sigmoid
import numpy as np

def weights_init_normal(m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''

        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
            m.bias.data.fill_(0)

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
        self.ff1.apply(weights_init_normal)

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
        
        x1 = self.sigmoid(self.ff1(x1))
        x2 = self.sigmoid(self.ff1(x2))

        #y_pred = self.sigmoid(x1-x2)
        return x1, x2