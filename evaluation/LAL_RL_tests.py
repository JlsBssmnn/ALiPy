from alipy import ToolBox
import pickle
from tqdm.auto import tqdm
from alipy.query_strategy import QueryInstanceLAL_RL, LAL_RL_StrategyLearner
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

# with default parameters for LogisticRegression we likely get a ConvergenceWarning
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os

     
def test_LAL_RL(save_path, save_name, path, strategy_name, rounds=10, test_ratio=0.2, init_lab=2, num_of_queries=100, model=None, **kwargs):
    """
    save_path: directory where the result will be saved
    save_name: name for the saved file
    path: path to the dataset that will be used for AL, must be a dict containing the X and y data
    strategy_name: alipy query strategy that will be used
    rounds: how many AL rounds will be performed
    test_ratio: the ratio of data that will be used for testing
    init_lab: int, the number of samples that will be labeled initially
    num_of_queries: how many queries will be performed
    kwargs: parameters for the LAL_RL strategy (model_path, n_state_estimation, pred_batch, device)
    """
    # open dataset
    data = pickle.load(open(path, "rb"))
    X,y = data['X'], data['y']
    y = y.ravel()

    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

    # label 'init_lab' samples at beginning
    ini_lab_ratio = init_lab/(len(y)*(1-test_ratio))

    alibox.split_AL(test_ratio=test_ratio, initial_label_rate=ini_lab_ratio, split_count=rounds)
        
    # Use the default Logistic Regression classifier or SVM
    if model == None:
        model = LogisticRegression()
    elif model.upper() == "SVM":
        model = svm.SVC()
        
    # initialize either an LAL_RL strategy or another ALiPy strategy
    if strategy_name == "QueryInstanceLAL_RL":
        strategy = QueryInstanceLAL_RL(X=X, y=y,
                                       model_path=kwargs.get('model_path'),
                                       n_state_estimation=kwargs.get('n_state_estimation'),
                                       pred_batch=kwargs.get('pred_batch', 128),
                                       device=kwargs.get('device'))
    else:
        strategy = alibox.get_query_strategy(strategy_name)

    # the array that will save the results: dim 0 contains the AL rounds,
    # dim 1 contains the accuracies after each query (the last element is the accuracy
    # that could have been achieved if all data would be labeled)
    quality_results = np.empty((rounds,num_of_queries + 2))


    for round in tqdm(range(rounds), desc="AL rounds"):
        j = 0
        # Get the data split
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        
        # calculate the accuracy for the case that all data ist labeled
        model_copy = clone(model)
        model_copy.fit(X=X[train_idx], y=y[train_idx])
        pred = model_copy.predict(X[test_idx])
        max_accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                y_pred=pred,
                                                performance_metric='accuracy_score')
        quality_results[round,-1] = max_accuracy

        # calculate the initial accuracy
        model = clone(model)
        model.fit(X=X[label_ind.index], y=y[label_ind.index])
        pred = model.predict(X[test_idx])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                y_pred=pred,
                                                performance_metric='accuracy_score')

        # we only care about how close we are to the maximum accuracy, not the actual accuracy
        accuracy /= max_accuracy
        quality_results[round,j] = accuracy
        
        for _ in range(num_of_queries):
            j += 1
            select_ind = strategy.select(label_ind, unlab_ind, model=model, batch_size=1)
        
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
        
            model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
            pred = model.predict(X[test_idx])
            accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                    y_pred=pred,
                                                    performance_metric='accuracy_score')
            accuracy /= max_accuracy
            quality_results[round, j] = accuracy
        
    # saving the results
    np.save(save_path + "/" + save_name + ".npy", quality_results)


# finding the right hyper parameters for LAL_RL
class Attr:
    # class that stores fixed attributes
    def __init__(self, dataset_path, possible_dataset_names, saving_path):
        self.dataset_path = dataset_path
        self.possible_dataset_names = possible_dataset_names
        self.saving_path = saving_path
        
    def __str__(self):
        s = ""
        for key in self.__dict__:
            s += key + "=" + str(getattr(self, key)) + ","
        return s[:-1]


class Model(BaseEstimator):
    def __init__(self, attr, n_state_estimation, subset, tolerance_level, model, replay_buffer_size, 
                 prioritized_replay_exponent, warm_start_episodes, nn_updates_per_warm_start, learning_rate, 
                 batch_size, gamma, update_rate, training_iterations, episodes_per_iteration, updates_per_iteration, 
                 epsilon_start, epsilon_end, epsilon_step):
        self.attr = attr
        self.n_state_estimation = n_state_estimation
        self.subset = subset
        self.tolerance_level = tolerance_level
        self.model = model
        self.replay_buffer_size = replay_buffer_size
        self.prioritized_replay_exponent = prioritized_replay_exponent
        self.warm_start_episodes = warm_start_episodes
        self.nn_updates_per_warm_start = nn_updates_per_warm_start
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_rate = update_rate
        self.training_iterations = training_iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.updates_per_iteration = updates_per_iteration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step = epsilon_step
        
        self.LAL_RL = LAL_RL_StrategyLearner(attr.dataset_path, attr.possible_dataset_names, n_state_estimation,
                                             subset=subset, tolerance_level=tolerance_level,
                                             model = None if model==None else svm.SVC(),
                                             replay_buffer_size=replay_buffer_size,
                                             prioritized_replay_exponent=prioritized_replay_exponent)
    
    def fit(self, X, y):
        file_name = "n=" + str(self.n_state_estimation) + ",r=" + str(self.replay_buffer_size) + ",g=" + str(self.gamma) + ".txt"
        if os.path.exists(os.path.join(self.attr.saving_path, file_name)):
            return
        
        f = open(os.path.join(self.attr.saving_path, file_name), "w")
        for key in self.__dict__:
            f.write(key + "=" + str(getattr(self, key)) + "\n")
        f.close()
        
        self.LAL_RL.train_query_strategy(self.attr.saving_path, file_name[:-4], self.warm_start_episodes, 
                self.nn_updates_per_warm_start, self.learning_rate, self.batch_size, self.gamma, self.update_rate,
                self.training_iterations, self.episodes_per_iteration, self.updates_per_iteration,
                self.epsilon_start, self.epsilon_end, self.epsilon_step)
        
        test_LAL_RL(self.attr.saving_path, file_name[:-4], os.path.join(self.attr.dataset_path, "adult.p"), 
                    "QueryInstanceLAL_RL", 5, 0.3, 4, model_path=os.path.join(self.attr.saving_path, file_name[:-4]+".pt"))
    
    def score(self, X, y):
        # we don't use the score function, this is just arbitrary output
        return self.n_state_estimation


def search_hyper_parameters(dataset_path, possible_dataset_names, saving_path, iterations):
    param = dict(
        n_state_estimation = range(25,50),
        subset = [-1,0,1],
        tolerance_level = uniform(loc=0.94, scale=0.05),
        model = [None, "svm"],
        replay_buffer_size = range(5000, 10000),
        prioritized_replay_exponent = [1,2,3],
        warm_start_episodes = range(32,300),
        nn_updates_per_warm_start = range(0,200),
        learning_rate = [1e-3, 1e-4],
        batch_size = range(8,129),
        gamma = uniform(loc=0.8, scale=0.199),
        update_rate = range(1,6),
        training_iterations = range(750,1001),
        episodes_per_iteration = range(5,20),
        updates_per_iteration = range(30,200),
        epsilon_start = uniform(loc=0.8, scale=0.2),
        epsilon_end = uniform(loc=0, scale=0.2),
        epsilon_step = range(750,1001)
    )

    attr = Attr(dataset_path, possible_dataset_names, saving_path)
    est = Model(attr, n_state_estimation=1, subset=-1, tolerance_level=0.98, model=None, replay_buffer_size=1e4, 
                    prioritized_replay_exponent=3, warm_start_episodes=128, nn_updates_per_warm_start=100, learning_rate=1e-3, 
                    batch_size=32, gamma=0.999, update_rate=100, training_iterations=1000, episodes_per_iteration=10,
                    updates_per_iteration=60, epsilon_start=1, epsilon_end=0.1, epsilon_step=1000)
    clf = RandomizedSearchCV(est, param, n_iter=iterations, refit=False, cv=2)

    clf.fit([[1,2], [3,4]], [1,0])