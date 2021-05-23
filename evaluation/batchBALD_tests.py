from evaluation.evaluation import ExperimentRunner
import numpy as np
from alipy.query_strategy import QueryInstanceBatchBALD
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchbnn as bnn

class BNN:

    def __init__(self):
        # create the BNN
        self.model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=784, out_features=100),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=100),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=100),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=10),
            nn.Softmax()
        )

        # create loss function and optimizer
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = 0.01

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def fit(self, X, y):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()

        pre = self.model(X)
        ce = self.ce_loss(pre, y)
        kl = self.kl_loss(self.model)
        cost = ce + self.kl_weight*kl
        
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

    def predict(self, X):
        pred = self.model(torch.from_numpy(X).float())
        return pred.argmax(dim=1).detach().numpy()

    def predict_proba(self, X):
        return np.array(self.model(torch.from_numpy(X).float()).detach().numpy())


def query_strat(label_ind, unlab_ind, batch_size, model_copy, query_strategy):
    return query_strategy.select(label_ind, unlab_ind, model_copy, batch_size, 10000)

def test_MNIST(batchsize, path, saving_path):
    """
    path: directory where X and y of the MNIST dataset are stored (as pickle files)
    """
    X = pickle.load(open(path + "/X.pkl", "rb"))
    y = pickle.load(open(path + "/y.pkl", "rb"))

    model = BNN()

    runner = ExperimentRunner(X, y, saving_path)
    runner.run_one_strategy("QueryInstanceBatchBALD", 6, None, 250, batchsize, 1/7, 2/6000, model,
                            "batchBALD_5batch", True, "batchBALD", query_strat)