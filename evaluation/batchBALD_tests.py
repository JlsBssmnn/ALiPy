from evaluation.evaluation import ExperimentRunner
import numpy as np
from alipy.query_strategy import QueryInstanceBatchBALD
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BayesianNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input

class Model:

    def __init__(self, net, batch_size=20):
        self.net = net
        self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss = nn.NLLLoss()
        self.batch_size = batch_size

    def fit(self, X, y, epochs=50, device=None):
        X = torch.from_numpy(X).reshape(-1,1,28,28).float().to(device)
        y = torch.from_numpy(y).long().to(device)
 
        for _ in range(epochs):
            for i in range(0, len(X), self.batch_size):
                self.optimizer.zero_grad()
                pre = self.net(X[i:i+self.batch_size])
                loss = self.loss(pre, y[i:i+self.batch_size])
                
                loss.backward()
                self.optimizer.step()

    def predict(self, X, batch_size=500, device=None):
        if len(X) <= batch_size:
            return self.predict_proba_batch(X, device).argmax(axis=1)
        
        pred = self.predict_proba_batch(X[:batch_size], device).argmax(axis=1)
        for i in range(batch_size, len(X), batch_size):
            pred = np.concatenate((pred, self.predict_proba_batch(X[i:i+batch_size], device).argmax(axis=1)))
        return pred

    def predict_proba(self, X, batch_size=500, device=None):
        if len(X) <= batch_size:
            return np.exp(self.predict_proba_batch(X, device))

        pred = self.predict_proba_batch(X[:batch_size], device)
        for i in range(batch_size, len(X), batch_size):
            pred = np.concatenate((pred, self.predict_proba_batch(X[i:i+batch_size], device)))
        return np.exp(pred)

    def predict_proba_batch(self, batch, device=None):
        return self.net(torch.from_numpy(batch).reshape(-1,1,28,28).float().to(device)).detach().cpu().numpy()


class query_strat:
    def __init__(self,K,num_samples):
        self.K = K
        self.num_samples = num_samples
    
    def select(self, label_ind, unlab_ind, batch_size, model_copy, query_strategy):
        return query_strategy.select(label_ind, unlab_ind, model_copy, batch_size, self.num_samples, self.K)

def test_MNIST(batchsize, path, saving_path, k, num_samples):
    """
    path: directory where X and y of the MNIST dataset are stored (as pickle files)
    """
    X = pickle.load(open(path + "/X.pkl", "rb"))
    y = pickle.load(open(path + "/y.pkl", "rb"))

    net = BayesianNet()
    model = Model(net)

    runner = ExperimentRunner(X, y, saving_path)
    runner.run_one_strategy("QueryInstanceBatchBALD", 6, None, 250, batchsize, 1/7, 2/6000, model,
                            "batchBALD_" + str(batchsize) + "batch", True, "batchBALD", query_strat(k,num_samples))