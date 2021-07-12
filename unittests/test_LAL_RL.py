import unittest
from alipy.query_strategy.LAL_RL import Net
import torch

class Test_QueryStrategy(unittest.TestCase):
    def create_Net(self):
        net = Net(3,0)
        # net.fc1.bias = torch.nn.Parameter(torch.tensor([0]*10, dtype=torch.float))
        # net.fc1.weight = torch.nn.Parameter(torch.tensor([[1,1,1]] + [[0,0,0]]*9, dtype=torch.float))

        net.fc2.bias = torch.nn.Parameter(torch.tensor([0]*13, dtype=torch.float))

        print(net.fc1.weight.shape)

    def test_q_value(self):
        pass


t = Test_QueryStrategy()
t.create_Net()