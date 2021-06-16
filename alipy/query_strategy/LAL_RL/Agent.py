import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, candidate_size, bias_initialization=None):
        super().__init__()
        self.fc1 = nn.Linear(candidate_size, 10)
        self.fc2 = nn.Linear(13, 5)
        self.fc3 = nn.Linear(5, 1)
        
        if bias_initialization is not None:
            self.fc3.bias = torch.nn.Parameter(torch.tensor(bias_initialization, dtype=torch.float))
        
    def forward(self, t):
        state = t[:,:-3]
        action = t[:,-3:]
        t = torch.sigmoid(self.fc1(state))
        t = torch.cat((t,action), dim=1)
        t = torch.sigmoid(self.fc2(t))
        t = self.fc3(t)
        return t


class Agent:
    def __init__(self, n_state_estimation=30, learning_rate=1e-3, batch_size=32, bias_average=0,
               gamma=0.999, device=None):
        self.net = Net(n_state_estimation, bias_average).to(device)
        self.target_net = Net(n_state_estimation, bias_average).to(device)
        self.target_net.eval()
        self.device = device
        
        # copy weihts from training net to target net
        self.target_net.load_state_dict(self.net.state_dict())
        
        # create loss function and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.gamma = gamma
        
    def train(self, minibatch):
        max_prediction_batch = []
        
        for i, next_classifier_state in enumerate(minibatch.next_classifier_state):
            # Predict q-value function value for all available actions
            n_next_actions = np.shape(minibatch.next_action_state[i])[1]
            next_classifier_state = np.repeat([next_classifier_state], n_next_actions, axis=0)
            next_classifier_state = np.concatenate((next_classifier_state, 
                                                    minibatch.next_action_state[i].transpose()), axis=1)
            input_tensor = torch.tensor(next_classifier_state, dtype=torch.float, device=self.device)
            
            # Use target_estimator
            target_predictions = self.target_net(input_tensor)
            
            # Use estimator
            predictions = self.net(input_tensor)
            
            target_predictions = np.ravel(target_predictions.detach().cpu().numpy())
            predictions = np.ravel(predictions.detach().cpu().numpy())
            
            # Follow Double Q-learning idea of van Hasselt, Guez, and Silver 2016
            # Select the best action according to predictions of estimator
            best_action_by_estimator = np.random.choice(np.where(predictions == np.amax(predictions))[0])
            # As the estimate of q-value of the best action, 
            # take the prediction of target estimator for the selecting action
            max_target_prediction_i = target_predictions[best_action_by_estimator]
            max_prediction_batch.append(max_target_prediction_i)
            
        expected_state_action_values = minibatch.reward + self.gamma*np.array(max_prediction_batch)
        expected_state_action_values = torch.tensor(expected_state_action_values, dtype=torch.float, device=self.device)
        
        input_tensor = np.concatenate((minibatch.classifier_state, minibatch.action_state), axis=1)
        input_tensor = torch.from_numpy(input_tensor).to(self.device).float()
        net_output = self.net(input_tensor)
        net_output = net_output.flatten()
        
        td_errors = net_output - expected_state_action_values
        
        # actually train the network
        loss = self.loss(net_output, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return td_errors.detach().cpu().numpy()
    
    def get_action(self, classifier_state, action_state):
        input_tensor = np.concatenate((np.repeat(classifier_state[None,:], action_state.shape[1], axis=0), 
                                       action_state.transpose()), axis=1)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float, device=self.device)
        predictions = self.net(input_tensor)
        predictions = predictions.flatten()
        
        predictions = predictions.detach().cpu().numpy()
        max_action = np.random.choice(np.where(predictions == predictions.max())[0])
        
        return max_action
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
    
    def save_net(self, path):
        torch.save(self.net.state_dict(), path + ".pt")

    def save_target_net(self, path):
        torch.save(self.target_net.state_dict(), path + "_target_net.pt")