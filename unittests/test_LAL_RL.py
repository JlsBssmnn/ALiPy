import unittest
from alipy.query_strategy.LAL_RL import Net
from alipy.query_strategy import QueryInstanceLAL_RL
import torch
import numpy as np
import sys

class Test_QueryStrategy(unittest.TestCase):
    def create_Net(self):
        net = Net(3,0)
        net.fc1.bias = torch.nn.Parameter(torch.tensor([0]*10, dtype=torch.float))
        net.fc1.weight = torch.nn.Parameter(torch.tensor([[1,1,1]] + [[0,0,0]]*9, dtype=torch.float))

        net.fc2.bias = torch.nn.Parameter(torch.tensor([0]*5, dtype=torch.float))
        net.fc2.weight = torch.nn.Parameter(torch.tensor([[1]+[0]*9+[-1]*3] + [[0]*13]*4, dtype=torch.float))

        net.fc3.weight = torch.nn.Parameter(torch.tensor([[1,0,0,0,0]], dtype=torch.float))

        return net

    def save_net(self, net):
        torch.save(net.state_dict(), sys.path[0] + "/net.pt")

    def sigmoid(self, num):
        return torch.sigmoid(torch.tensor(num)).item()

    def test_q_value(self):
        net = self.create_Net()
        for _ in range(10):
            state = torch.rand(3)
            action = torch.rand(3)

            expected_q_value = self.sigmoid(self.sigmoid(sum(state).item()) - sum(action).item())
            self.assertAlmostEqual(net(torch.cat((state,action)).reshape(1,6)).item(), expected_q_value)
        
        inputs = torch.tensor([
            [0.8,0.9,0.85,0,0.1,0.2], # hight q-value
            [0.3,0.5,0.55,0.45,0.6,0.3], # medium q-value
            [0.01,0.1,0,0.9,1.2,2] # low q-value
        ], dtype=torch.float)
        value1 = net(inputs[:1]).item()
        value2 = net(inputs[1:2]).item()
        value3 = net(inputs[2:3]).item()

        self.assertTrue(value3 < value2 < value1)

    
    def test_query_strat(self):
        data = np.array([[5,0],
                         [4,0],
                         [10,0],
                         [3,0],
                         [1,0],
                         [1,0.1],
                         [1,-1],
                         [-1,0]])
        strat = QueryInstanceLAL_RL(data, [0]*data.shape[0], sys.path[0]+"/net.pt")
        model = Model()

        np.random.seed(63) # choice of size 7 will produce [1,0,2]
        select_ind = strat.select([0], [1,2,3,4,5,6,7], model)
        self.assertEqual(select_ind[0], 4)

        np.random.seed(63) # choice of size 6 will produce [1,0,2]
        select_ind = strat.select([0], [1,2,3,5,6,7], model)
        self.assertEqual(select_ind[0], 5)

        np.random.seed(0) # choice of size 5 will produce [2,0,1]
        select_ind = strat.select([0], [1,2,3,6,7], model)
        self.assertEqual(select_ind[0], 6)

        np.random.seed(5) # choice of size 4 will produce [0,1,2]
        select_ind = strat.select([0], [1,2,3,7], model)
        self.assertEqual(select_ind[0], 7)

        
class Model:
    def predict_proba(self, data):
        result = []
        for x in data:
            if np.all(x == [5,0]):
                result.append(0.8)
            elif np.all(x == [4,0]):
                result.append(0.7)
            elif np.all(x == [10,0]):
                result.append(0.9)
            elif np.all(x == [3,0]):
                result.append(0.95)
            elif np.all(x == [1,0]):
                result.append(0)
            elif np.all(x == [1,0.1]):
                result.append(0.3)
            elif np.all(x == [1,-1]):
                result.append(0.6)
            elif np.all(x == [-1,0]):
                result.append(1)
        return np.array(result).reshape(data.shape[0],1)

if __name__ == "__main__":
    unittest.main()