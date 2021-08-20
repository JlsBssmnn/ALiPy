import numpy as np
import pandas as pd
import os
import sklearn
import pickle
from sklearn import metrics
import torchvision
from alipy.query_strategy import LAL_RL_StrategyLearner, QueryInstanceLAL_RL
from tqdm.auto import tqdm
from datetime import datetime

from alipy.query_strategy.query_labels import QueryInstanceBatchBALD
from .evaluation import ExperimentRunner

def prepare_datasets(csv_directory, saving_directory):
    """
    Save the csv datasets and EMIST and CIFAR-10 as dictionaries that contain
    the samples and the labels in the given directory
    """
    # the csv datasets
    csv_datasets = ["australian", "DIABETES", "dwtc", "FERTILITY", "flag", "GERMAN", "glass",
        "HABERMAN", "HEART", "IONOSPHERE_ionosphere", "olivetti", "PLANNING", "zoo"]
    for name in csv_datasets:
        csv_dataframe = pd.read_csv(os.path.join(csv_directory, name + ".csv"), header=0, delimiter=",")
        csv_dataframe = csv_dataframe[[x for x in csv_dataframe if x!="LABEL"] + ["LABEL"]]
        X = csv_dataframe.to_numpy()[:, :-1]
        y = csv_dataframe.to_numpy()[:, -1]

        if type(y[0]) == str:
            # convert the labels to integers
            _, y = np.unique(y, return_inverse=True)
        X = sklearn.preprocessing.minmax_scale(X)

        dataset_dict = {'X': X, 'y': y}
        f = open(os.path.join(saving_directory, name + ".p"), "wb")
        pickle.dump(dataset_dict, f)
        f.close()
    print("Successfully saved the csv datasets as pickle files")
    
    # EMNIST and CIFAR-10
    for i in range(2):
        if i == 0:
            test = torchvision.datasets.EMNIST(root=os.path.join(saving_directory, "pytorch_datasets"), split="byclass", train=False,
                download=True, transform=None)
            X_test = test.data.numpy()
            y_test = test.targets.numpy()
        elif i == 1:
            test = torchvision.datasets.CIFAR10(root=os.path.join(saving_directory, "pytorch_datasets"), train=False,
                download=True, transform=None)
            X_test = test.data
            y_test = test.targets
            y_test = np.array(y_test)

        X_test = X_test.reshape(X_test.shape[0], -1)
        X_test = sklearn.preprocessing.minmax_scale(X_test)
        y_test = y_test.flatten()

        dataset_dict = {'X': X_test, 'y': y_test}
    
        if i == 0:
            f = open(os.path.join(saving_directory, "EMNIST.p"), "wb")
        elif i == 1:
            f = open(os.path.join(saving_directory, "CIFAR10.p"), "wb")
        pickle.dump(dataset_dict, f)
        f.close()   
    print("Successfully saved EMNIST and CIFAR-10")

def quality_function(y_true, y_pred):
    """
    This function computes the f1 score with fixed parameters, the result is equal to
    >>> sklearn.metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    """
    return metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)

def train_LAL_RL_strats(dataset_path, saving_path, datasets="all"):
    """
    This function will train LAL_RL strategies on all small datasets
    and saves these strategies to the given saving_path
    """
    # not using EMNIST and CIFAR-10 because it needs too much memory and time
    all_datasets = [x for x in os.listdir(dataset_path) if x.endswith(".p") and 
                        not (x.startswith("EMNIST") or x.startswith("CIFAR10"))]
    all_datasets = [x[:-2] for x in all_datasets]

    assert set(all_datasets).issuperset(set(datasets)) or datasets == "all"
    eval_datasets = all_datasets if datasets == "all" else datasets

    time_file = open(os.path.join(saving_path, "time_info.txt"), "x")
    start = datetime.now()
    time_file.write(start.strftime("Start of the experiment: %d.%m.%Y - %H:%M:%S\n"))
    p = tqdm(total = len(eval_datasets) + 1)

    for dataset in eval_datasets:
        p.set_description("learn LAL_RL for " + dataset)
        start_of_round = datetime.now()

        learner = LAL_RL_StrategyLearner(dataset_path,
            [x for x in all_datasets if x != dataset], size=100, quality_method=quality_function)
        learner.train_query_strategy(saving_path, "LAL_RL_"+dataset, verbose=3)

        end_of_round = datetime.now()
        diff = str(end_of_round - start_of_round)
        time_file.write(f"\tDuration for dataset {dataset} --{diff[:diff.rfind('.')]}--\n")
        p.update()
    else:
        p.set_description("learn LAL_RL for all datasets")
        start_of_round = datetime.now()

        learner = LAL_RL_StrategyLearner(dataset_path, all_datasets, size=100, quality_method=quality_function)
        learner.train_query_strategy(saving_path, "LAL_RL_all_datasets", verbose=3)

        end_of_round = datetime.now()
        diff = str(end_of_round - start_of_round)
        time_file.write(f"\tDuration for all datasets --{diff[:diff.rfind('.')]}--\n")
        p.update()
    
    end = datetime.now()
    diff = str(end - start)
    time_file.write(end.strftime("End of the experiment: %d.%m.%Y - %H:%M:%S\n"))
    time_file.write(f"Duration of experiment {diff[:diff.rfind('.')]}\n")
    time_file.close()
    p.close()

def test_LAL_RL(dataset_path, model_path, saving_path, datasets="all"):
    """
    Runs the LAL_RL strategies saved at the model_path on all datasets and saves the result
    """
    all_datasets = initialize(dataset_path, saving_path, datasets)
    
    p = tqdm(total = len(all_datasets))
    for dataset in all_datasets:
        p.set_description("LAL_RL test on " + dataset)
        data = pickle.load(open(os.path.join(dataset_path, dataset), "rb"))
        X, y = data['X'], data['y']

        if dataset in ["EMNIST.p", "CIFAR10.p"]:
            al_cycles = 1000
            model_name = "LAL_RL_all_datasets.pt"
        else:
            al_cycles = 50
            model_name = "LAL_RL_"+dataset[:-2]+".pt"
        
        # required because of the structure of evaluation.py
        query_strategy = LAL_RL_strategy(QueryInstanceLAL_RL(X, y, os.path.join(model_path, model_name)))

        runner = ExperimentRunner(X, y, os.path.join(saving_path, dataset[:-2]))
        runner.run_one_strategy("QueryInstanceRandom", 100, al_cycles, batch_size=5, test_ratio=0.5, initial_label_rate='min',
            model=sklearn.ensemble.RandomForestClassifier(), file_name="LAL_RL", custom_query_strat=query_strategy,
            performance_metric="f1_score", log_timing=True)
        p.update()
    p.close()


class LAL_RL_strategy:
    def __init__(self, lal_rl):
        self.lal_rl = lal_rl

    def select(self, label_ind, unlab_ind, batch_size, model_copy, query_strategy, device=None):
        return self.lal_rl.select(label_ind, unlab_ind, model_copy, batch_size)


def test_batchBALD_BRF(dataset_path, saving_path, dropout_rate, datasets="all"):
    """
    Tests batchBALD on the datasets with a bayesian random forest classifier
    """
    all_datasets = initialize(dataset_path, saving_path, datasets)
    
    p = tqdm(total = len(all_datasets))
    for dataset in all_datasets:
        p.set_description("batchBALD test on " + dataset)
        data = pickle.load(open(os.path.join(dataset_path, dataset), "rb"))
        X, y = data['X'], data['y']
        
        if dataset in ["EMNIST.p", "CIFAR10.p"]:
            al_cycles = 1000
        else:
            al_cycles = 50

        # required because of the structure of evaluation.py
        query_strategy = BatchBALD_Query_Strategy(QueryInstanceBatchBALD(X, y))

        runner = ExperimentRunner(X, y, os.path.join(saving_path, dataset[:-2]))
        runner.run_one_strategy("QueryInstanceRandom", 100, al_cycles, batch_size=5, test_ratio=0.5, initial_label_rate='min',
            model=batchBALD_Model(BayesianRandomForest(dropout_rate=dropout_rate)), file_name="batchBALD",
            custom_query_strat=query_strategy, performance_metric="f1_score", log_timing=True)
        p.update()
    p.close()


class BayesianRandomForest(sklearn.ensemble.RandomForestClassifier):
    """
    This is a RandomForestClassifier with dropout. This means that some trees are deleted
    if activate_dropout is called. If deactivate_dropout is classed they are added again.
    """
    def __init__(self, dropout_rate, n_estimators=100):
        super().__init__(n_estimators=n_estimators)
        self.dropout = False
        self.dropout_rate = dropout_rate
        self.n_dropouts = int(np.round(self.n_estimators * self.dropout_rate))
    
    def activate_dropout(self):
        if self.dropout:
            return
        self.deleted_trees = []
        for i in range(self.n_dropouts):
            self.deleted_trees.append(self.estimators_.pop(np.random.choice(len(self.estimators_))))
        self.dropout = True
        
    def deactivate_dropout(self):
        if not self.dropout:
            return
        for i in range(len(self.deleted_trees)):
            self.estimators_.append(self.deleted_trees.pop(0))
        self.dropout = False
            
    def change_dropout(self):
        if self.dropout:
            self.deactivate_dropout()
            self.activate_dropout()
        else:
            self.activate_dropout()
            

class BatchBALD_Query_Strategy:
    def __init__(self, batchBALD):
        self.batchBALD = batchBALD

    def select(self, label_ind, unlab_ind, batch_size, model_copy, query_strategy, device):
        return self.batchBALD.select(label_ind, unlab_ind, model_copy, batch_size, num_samples=10000)


class batchBALD_Model:
    def __init__(self, classifier):
        self.classifier = classifier
        self.num_classes = None

    def fit(self, X, y):
        if self.num_classes == None:
            self.num_classes = np.unique(y).size
        return self.classifier.fit(X,y)
        
    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        self.classifier.activate_dropout()
        pred = self.classifier.predict_proba(X).reshape(-1,1,self.num_classes)
        for _ in range(9):
            self.classifier.change_dropout()
            pred = np.concatenate((pred, self.classifier.predict_proba(X).reshape(-1,1,self.num_classes)), axis=1)
        self.classifier.deactivate_dropout()
        return pred

def test_unc_rand(dataset_path, saving_path, datasets="all"):
    all_datasets = initialize(dataset_path, saving_path, datasets)

    p = tqdm(total = len(all_datasets))
    for dataset in all_datasets:
        p.set_description("random test on " + dataset)
        data = pickle.load(open(os.path.join(dataset_path, dataset), "rb"))
        X, y = data['X'], data['y']
        
        if dataset in ["EMNIST.p", "CIFAR10.p"]:
            al_cycles = 1000
        else:
            al_cycles = 50

        runner = ExperimentRunner(X, y, os.path.join(saving_path, dataset[:-2]))
        runner.run_one_strategy("QueryInstanceRandom", 100, al_cycles, batch_size=5, test_ratio=0.5, initial_label_rate='min',
            model=sklearn.ensemble.RandomForestClassifier(), file_name="random",
            performance_metric="f1_score", log_timing=True)

        p.set_description("uncertainty test on " + dataset)

        runner = ExperimentRunner(X, y, os.path.join(saving_path, dataset[:-2]))
        runner.run_one_strategy("QueryInstanceUncertainty", 100, al_cycles, batch_size=5, test_ratio=0.5, initial_label_rate='min',
            model=sklearn.ensemble.RandomForestClassifier(), file_name="uncertainty",
            performance_metric="f1_score", log_timing=True)
        p.update()
    p.close()


def initialize(dataset_path, saving_path, datasets="all"):
    all_datasets = [x for x in os.listdir(dataset_path) if x.endswith(".p")]

    assert set(all_datasets).issuperset(set(datasets)) or datasets == "all"
    if datasets is not "all":
        all_datasets = datasets

    # don't prioritize CIFAR10 and EMINST
    if "CIFAR10.p" in all_datasets:
        all_datasets.remove("CIFAR10.p")
        all_datasets.append("CIFAR10.p")
    if "EMNIST.p" in all_datasets:
        all_datasets.remove("EMNIST.p")
        all_datasets.append("EMNIST.p")

    for dataset in all_datasets:
        if dataset[:-2] not in os.listdir(saving_path):
            os.mkdir(os.path.join(saving_path, dataset[:-2]))

    print("Begin AL runs on", all_datasets)

    return all_datasets


def test_unc_rand2(dataset_path, saving_path, datasets="all"):
    all_datasets = initialize(dataset_path, saving_path, datasets)

    p = tqdm(total = len(all_datasets))
    for dataset in all_datasets:
        p.set_description("random test on " + dataset)
        data = pickle.load(open(os.path.join(dataset_path, dataset), "rb"))
        X, y = data['X'], data['y']
        
        runner = ExperimentRunner(X, y, os.path.join(saving_path, dataset[:-2], "random"), dataset[:-2])
        runner.evaluation("QueryInstanceRandom")


        p.set_description("uncertainty test on " + dataset)

        runner = ExperimentRunner(X, y, os.path.join(saving_path, dataset[:-2], "uncertainty"), dataset[:-2])
        runner.evaluation("QueryInstanceUncertainty")
        p.update()
    p.close()
