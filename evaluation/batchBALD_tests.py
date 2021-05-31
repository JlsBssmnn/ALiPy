from evaluation.evaluation import ExperimentRunner
import numpy as np
from alipy.query_strategy import QueryInstanceBatchBALD
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm


class query_strat:
    def __init__(self,K,num_samples):
        self.K = K
        self.num_samples = num_samples
    
    def select(self, label_ind, unlab_ind, batch_size, model_copy, query_strategy, device=None):
        return query_strategy.select(label_ind, unlab_ind, model_copy, batch_size, self.num_samples, device)

def test_MNIST(batchsize, path, saving_path, k, num_samples):
    """
    batchsize: acquisition size
    path: path to the data folder created by pytorch
    """

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(path, train=True, download=True,
                    transform=transform)
    dataset2 = datasets.MNIST(path, train=False, download=True,
                    transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1)
    test_loader = torch.utils.data.DataLoader(dataset2)

    X = np.empty((70000,784))
    y = np.empty((70000,))
    for i, (data, target) in enumerate(train_loader):
        X[i] = (data.reshape(1,784).detach().numpy())
        y[i] = target.detach().numpy()
        
    for j, (data, target) in enumerate(test_loader):
        X[i+j] = (data.reshape(1,784).detach().numpy())
        y[i+j] = target.detach().numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BayesianNet().to(device)
    model = Model(net)

    runner = ExperimentRunner(X, y, saving_path)
    runner.run_one_strategy("QueryInstanceBatchBALD", 6, None, 250, batchsize, 1/7, 2/6000, model,
                            "batchBALD_" + str(batchsize) + "batch", True, "batchBALD", query_strat(k,num_samples),device,True)


class BayesianNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # self.conv1_drop = nn.Dropout(0.5)
        self.conv1_drop = MCDropout()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # self.conv2_drop = nn.Dropout(0.5)
        self.conv2_drop = MCDropout()
        self.fc1 = nn.Linear(1024, 128)
        # self.fc1_drop = nn.Dropout(0.5)
        self.fc1_drop = MCDropout()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, input):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input

    def fix_dropout(self):
        self.conv1_drop.train(False)
        self.conv2_drop.train(False)
        self.fc1_drop.train(False)

    def unfix_dropout(self):
        self.conv1_drop.train(True)
        self.conv2_drop.train(True)
        self.fc1_drop.train(True)

    def disable_dropout(self):
        self.conv1_drop.p = 0
        self.conv2_drop.p = 0
        self.fc1_drop.p = 0
    
    def enable_dropout(self):
        self.conv1_drop.p = 0.5
        self.conv2_drop.p = 0.5
        self.fc1_drop.p = 0.5

class Model:

    def __init__(self, net, batch_size=20):
        self.net = net
        self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss = nn.NLLLoss()
        self.batch_size = batch_size

    def fit(self, X, y, epochs=1, device=None):
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
        self.net.disable_dropout()
        if len(X) <= batch_size:
            pred = self.predict_proba_batch(X, device).argmax(axis=1)
            self.net.enable_dropout()
            return pred
        
        pred = self.predict_proba_batch(X[:batch_size], device).argmax(axis=1)
        for i in range(batch_size, len(X), batch_size):
            pred = np.concatenate((pred, self.predict_proba_batch(X[i:i+batch_size], device).argmax(axis=1)))
        self.net.enable_dropout()
        return pred

    def predict_proba(self, X, device=None, K=10):
        assert len(X) > 0
        self.net.fix_dropout()

        pred = self.predict_proba_datapoint(X[:1], device, K)[None,:,:]
        for i in tqdm(range(1,len(X)), desc="Calc Model output", leave=False):
            output = self.predict_proba_datapoint(X[i:i+1], device, K)[None,:,:]
            pred = np.concatenate((pred, output), axis=0)

        self.net.unfix_dropout()

        return pred

    def predict_proba_datapoint(self, X, device=None, K=10):
        if len(X) != 1:
            raise ValueError("predict_proba of this model only accepts arrays of length 1")

        X = X.reshape(1,1,28,28)
        X = np.concatenate((X,)*K, axis=0)
        output = self.net(torch.from_numpy(X).float().to(device))
        return np.exp(output.detach().cpu().numpy())

    def predict_proba_batch(self, batch, device=None):
        return self.net(torch.from_numpy(batch).reshape(-1,1,28,28).float().to(device)).detach().cpu().numpy()


# from mc_dropout.py

DROPOUT_PROB = 0.5

def set_dropout_p(bayesian_net: nn.Module, p):
    def update_k(module: nn.Module):
        if isinstance(module, MCDropout):
            module.p = p

    bayesian_net.apply(update_k)


class BayesianModule(nn.Module):
    """A module that we can sample multiple times from given a single input batch.
    To be efficient, the module allows for a part of the forward pass to be deterministic.
    """

    k = None

    def __init__(self, num_classes: int):
        super().__init__()

        # TODO: use a different class for this!
        self.num_classes: int = num_classes

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, k: int):
        BayesianModule.k = k

        input_B = self.deterministic_forward_impl(input_B)
        mc_input_BK = BayesianModule.mc_tensor(input_B, k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = BayesianModule.unflatten_tensor(mc_output_BK, k)
        return mc_output_B_K

    def deterministic_forward_impl(self, input: torch.Tensor):
        return input

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return mc_input_BK

    def set_dropout_p(self, p):
        def update_k(module: nn.Module):
            if isinstance(module, MCDropout):
                module.p = p

        self.apply(update_k)

    @staticmethod
    def unflatten_tensor(input: torch.Tensor, k: int):
        input = input.view([-1, k] + list(input.shape[1:]))
        return input

    @staticmethod
    def flatten_tensor(mc_input: torch.Tensor):
        return mc_input.flatten(0, 1)

    @staticmethod
    def mc_tensor(input: torch.tensor, k: int):
        mc_shape = [input.shape[0], k] + list(input.shape[1:])
        return input.unsqueeze(1).expand(mc_shape).flatten(0, 1)


class MCDropout(nn.Module):
    __constants__ = ["p"]

    def __init__(self):
        super().__init__()
        self.k = None

        p = DROPOUT_PROB

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.mask = None

    def extra_repr(self):
        return "p={}".format(self.p)

    def reset_mask(self):
        self.mask = None

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.reset_mask()

    def _get_sample_mask_shape(self, sample_shape):
        return sample_shape

    def _create_mask(self, input, k):
        mask_shape = [1, k] + list(self._get_sample_mask_shape(input.shape[1:]))
        mask = torch.empty(mask_shape, dtype=torch.bool, device=input.device).bernoulli_(self.p)
        return mask

    def forward(self, input: torch.Tensor):
        if self.p == 0.0:
            return input

        k = input.shape[0]
        if self.training:
            # Create a new mask on each call and for each batch element.
            mask = self._create_mask(input, k)
        else:
            if self.mask is None:
                # print('recreating mask', self)
                # Recreate mask.
                self.mask = self._create_mask(input, k)

            mask = self.mask

        mc_input = BayesianModule.unflatten_tensor(input, k)
        mc_output = mc_input.masked_fill(mask, 0) / (1 - self.p)

        # Flatten MCDI, batch into one dimension again.
        return BayesianModule.flatten_tensor(mc_output)