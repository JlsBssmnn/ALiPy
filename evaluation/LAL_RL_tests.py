import copy
from alipy import ToolBox
import pickle
from tqdm.auto import tqdm
from alipy.query_strategy import QueryInstanceLAL_RL
from sklearn import svm

import numpy as np

     
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
        model = alibox.get_default_model()
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
        model_copy = copy.deepcopy(model)
        model_copy.fit(X=X[train_idx], y=y[train_idx])
        pred = model_copy.predict(X[test_idx])
        max_accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                y_pred=pred,
                                                performance_metric='accuracy_score')
        quality_results[round,-1] = max_accuracy

        # calculate the initial accuracy
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