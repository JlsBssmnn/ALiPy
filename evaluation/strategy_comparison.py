import numpy as np
import pandas as pd
import os
import sklearn
import pickle
import torchvision
from alipy.query_strategy import LAL_RL_StrategyLearner
from tqdm.auto import tqdm

def prepare_datasets(csv_directory, saving_directory):
    """
    Save the csv datasets and EMIST and CIFAR-10 as dictionaries that contain
    the samples and the labels in the given directory
    """
    # the csv datasets
    csv_datasets = ["australian", "DIABETES", "dwtc", "FERTILITY", "flag", "GERMAN", "glass",
        "HABERMAN", "HEART", "IONOSPHERE_ionosphere", "olivetti", "PLANNING", "zoo"]
    for name in csv_datasets:
        csv_dataframe = pd.read_csv(os.path.join(csv_directory, name + ".csv"), header="LABEL", delimiter=",")
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
            train = torchvision.datasets.EMNIST(root=os.path.join(saving_directory, "pytorch_datasets"), split="byclass", train=True,
                download=True, transform=None)
            test = torchvision.datasets.EMNIST(root=os.path.join(saving_directory, "pytorch_datasets"), split="byclass", train=True,
                download=True, transform=None)
        elif i == 1:
            train = torchvision.datasets.CIFAR10(root=os.path.join(saving_directory, "pytorch_datasets"), split="byclass", train=True,
                download=True, transform=None)
            test = torchvision.datasets.CIFAR10(root=os.path.join(saving_directory, "pytorch_datasets"), split="byclass", train=True,
                download=True, transform=None)
    
        X_train = train.data.numpy()
        y_train = train.targets.numpy()
        X_test = test.data.numpy()
        y_test = test.targets.numpy()

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        X = X.reshape(X.shape[0], -1)
        X = sklearn.preprocessing.minmax_scale(X)
        y = y.flatten()

        dataset_dict = {'X': X, 'y': y}
    
        if i == 0:
            f = open(os.path.join(saving_directory, "EMNIST.p"), "wb")
        elif i == 1:
            f = open(os.path.join(saving_directory, "CIFAR10.p"), "wb")
        pickle.dump(dataset_dict, f)
        f.close()   
    print("Successfully saved EMNIST and CIFAR-10")


def train_LAL_RL_strats(dataset_path, saving_path):
    """
    This function will train LAL_RL strategies on all datasets except one
    and saves these strategies to the given saving_path
    """
    all_datasets = [x for x in os.listdir(dataset_path) if os.path.isfile(x)]
    if len([x for x in all_datasets if not x.endswith(".p")]) > 0:
        raise ValueError("There are file in the given directory that don't end with .p")
    all_datasets = [x[:-2] for x in all_datasets]

    for dataset in tqdm(all_datasets, desc="learn LAL_RL's"):
        learner = LAL_RL_StrategyLearner(dataset_path, [x for x in all_datasets if x != dataset],
            size=100)
        learner.train_query_strategy(saving_path, "LAL_RL_"+dataset, verbose=1)