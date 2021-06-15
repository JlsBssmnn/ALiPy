from .envs import LalEnvTargetAccuracy
from .datasets import DatasetUCI
from .helpers import ReplayBuffer
from .Agent import Agent, Net
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy
import collections
from ..base import BaseIndexQuery

from tqdm import tqdm

import torch


class StrategyLearner:
    def __init__(self, path, possible_dataset_names, n_state_estimation=30, size=-1, subset=-1,
                 quality_method=metrics.accuracy_score, tolerance_level=0.98, model=None,
                 replay_buffer_size = 1e4, prioritized_replay_exponent = 3) :
        dataset = DatasetUCI(possible_dataset_names, n_state_estimation=n_state_estimation, subset=subset,
                             size=size,path=path)
        if model == None:
            model = LogisticRegression()
        self.env = LalEnvTargetAccuracy(dataset, model, quality_method=quality_method,
                                        tolerance_level=tolerance_level)
        self.n_state_estimation = n_state_estimation
        self.replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size, prior_exp=prioritized_replay_exponent)


    def train_query_strategy(self, saving_path, file_name, warm_start_episodes=128, nn_updates_per_warm_start=100,
                n_state_estimation=30, learning_rate=1e-3, batch_size=32, gamma=0.999, update_rate=100,
                training_iterations=1000, episodes_per_iteration=10, updates_per_iteration=60,
                epsilon_start=1, epsilon_end=0.1, epsilon_step=1000, device=None):
        self.saving_path = saving_path + "/" + file_name
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.training_iterations = training_iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.updates_per_iteration = updates_per_iteration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step = epsilon_step

        bias_average = self.run_warm_start_episodes(warm_start_episodes)
        self.agent = Agent(n_state_estimation, learning_rate, batch_size, bias_average,
               gamma, device)
        self.train_agent(nn_updates_per_warm_start)
        self.run_training_iterations()


    def run_warm_start_episodes(self, n_episodes):
        # Keep track of episode duration to compute average
        episode_durations = []
        for _ in tqdm(range(n_episodes), desc="Warmstart episodes"):
            # Reset the environment to start a new episode
            # classifier_state contains vector representation of state of the environment (depends on classifier)
            # next_action_state contains vector representations of all actions available to be taken at the next step
            classifier_state, next_action_state = self.env.reset()
            terminal = False
            episode_duration = 0
            # before we reach a terminal state, make steps
            while not terminal:
                # Choose a random action
                action = np.random.randint(0, self.env.n_actions)
                # taken_action_state is a vector corresponding to a taken action
                taken_action_state = next_action_state[:,action]
                next_classifier_state, next_action_state, reward, terminal = self.env.step(action)
                # Store the transition in the replay buffer
                self.replay_buffer.store_transition(classifier_state, 
                                            taken_action_state, 
                                            reward, next_classifier_state, 
                                            next_action_state, terminal)
                # Get ready for next step
                classifier_state = next_classifier_state
                episode_duration += 1 
            episode_durations.append(episode_duration)
        # compute the average episode duration of episodes generated during the warm start procedure
        av_episode_duration = np.mean(episode_durations)
        print('\nAverage episode duration = ', av_episode_duration)

        return -av_episode_duration/2


    def train_agent(self, n_of_updates):
        for _ in tqdm(range(n_of_updates), desc="Train q-net", leave=False):
            # Sample a batch from the replay buffer proportionally to the probability of sampling.
            minibatch = self.replay_buffer.sample_minibatch(self.batch_size)
            # Use batch to train an agent. Keep track of temporal difference errors during training.
            td_error = self.agent.train(minibatch)
            # Update probabilities of sampling each datapoint proportionally to the error.
            self.replay_buffer.update_td_errors(td_error, minibatch.indeces)


    def run_training_iterations(self):
        i_episode = 0

        for iteration in tqdm(range(self.training_iterations), desc="Train iterations"):
            # GENERATE NEW EPISODES
            # Compute epsilon value according to the schedule.
            epsilon = max(self.epsilon_end, self.epsilon_start-iteration*(self.epsilon_start-self.epsilon_end)/self.epsilon_step)
            # Simulate training episodes.
            for _ in tqdm(range(self.episodes_per_iteration), desc="Episodes", leave=False):
                # Reset the environment to start a new episode.
                classifier_state, next_action_state = self.env.reset()
                terminal = False
                # Keep track of stats of episode to analyse it in tensorboard.
                episode_reward = 0
                episode_duration = 0
                # Run an episode.
                while not terminal:
                    # Let an agent choose an action or with epsilon probability, take a random action.
                    if np.random.ranf() < epsilon: 
                        action = np.random.randint(0, self.env.n_actions)
                    else:
                        action = self.agent.get_action(classifier_state, next_action_state)
                    
                    # taken_action_state is a vector that corresponds to a taken action
                    taken_action_state = next_action_state[:,action]
                    # Make another step.
                    next_classifier_state, next_action_state, reward, terminal = self.env.step(action)
                    # Store a step in replay buffer
                    self.replay_buffer.store_transition(classifier_state, 
                                                taken_action_state, 
                                                reward, 
                                                next_classifier_state, 
                                                next_action_state, 
                                                terminal)
                    # Change a state of environment.
                    classifier_state = next_classifier_state
                    # Keep track of stats and add summaries to tensorboard.
                    episode_reward += reward
                    episode_duration += 1
                i_episode += 1
                    
            # NEURAL NETWORK UPDATES
            self.train_agent(self.updates_per_iteration)
            if iteration % self.update_rate == 0:
                self.agent.update_target_net()

        self.agent.save_net(self.saving_path)



class QueryInstanceLAL_RL(BaseIndexQuery):
    def __init__(self, X, y, model_path, n_state_estimation=None, pred_batch=128, device=None):
        super(QueryInstanceLAL_RL, self).__init__(X, y)
        state_dict = torch.load(model_path, map_location=device)
        self.pred_batch = pred_batch
        self.device = device
        if n_state_estimation == None:
            self.n_state_estimation = state_dict[list(state_dict.keys())[0]].size(1)
        else:
            self.n_state_estimation = n_state_estimation
        self.net = Net(self.n_state_estimation,0)
        self.net.load_state_dict(state_dict)
        self.net.to(device)
        self.net.eval()

    def select(self, label_index, unlabel_index, model=None, batch_size=1):
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        if len(unlabel_index) <= batch_size:
            return unlabel_index
        assert len(unlabel_index) > self.n_state_estimation
        unlabel_index = np.asarray(unlabel_index)

        # initialize the model and train it if necessary
        if model == None:
            model = LogisticRegression()
            model.fit(self.X[label_index], self.y[label_index])
        
        # set aside some unlabeled data for the state representation
        state_indices = np.random.choice(len(unlabel_index), size=self.n_state_estimation, replace=False)
        unlabel_index = unlabel_index[np.array([x for x in range(len(unlabel_index)) if x not in state_indices])]

        # create the state
        predictions = model.predict_proba(self.X[state_indices])[:,0]
        predictions = np.array(predictions)
        idx = np.argsort(predictions)
        state = predictions[idx]

        #create the actions
        a1 = model.predict_proba(self.X[unlabel_index])[:,0]

        # calculate distances
        data = self.X[np.concatenate((label_index,unlabel_index),axis=0)]
        distances = scipy.spatial.distance.pdist(data, metric='cosine')
        distances = scipy.spatial.distance.squareform(distances)
        indeces_known = np.arange(len(label_index))
        indeces_unknown = np.arange(len(label_index), len(label_index)+len(unlabel_index))
        a2 = np.mean(distances[indeces_unknown,:][:,indeces_unknown],axis=0)
        a3 = np.mean(distances[indeces_known,:][:,indeces_unknown],axis=0)

        actions = np.concatenate(([a1], [a2], [a3]), axis=0).transpose()

        # calculate the q-values according to the q-network
        # first transform the state and actions for the network
        state = np.repeat([state], actions.shape[0], axis=0)
        state_actions = np.concatenate((state,actions),axis=1)
        input_tensor = torch.tensor(state_actions, dtype=torch.float, device=self.device)

        # get the prediction from the network
        pred = self.net(input_tensor[:self.pred_batch])
        for i in range(self.pred_batch, input_tensor.size(0), self.pred_batch):
            pred = torch.cat((pred, self.net(input_tensor[i:i+self.pred_batch])))
        pred = pred.flatten()

        # sort the actions with respect to their q-value
        idx = pred.argsort(descending=True)
        idx = idx[:batch_size].detach().cpu().numpy()

        # return the correspoding indeces from the unlabeld index
        return unlabel_index[idx]