import copy
from alipy import ToolBox
import pickle

import numpy as np

     
def test_LAL_RL(save_path, save_name, path, strategy_name, rounds=10, test_ratio=0.2, num_of_queries=100):
    data = pickle.load(open(path, "rb"))
    X,y = data['X'], data['y']
    y = y.ravel()

    alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

    # Split data
    # test_ratio, ini_lab_ratio = 0.8,0.002

    # always label 2 samples at beginning
    ini_lab_ratio = 2/(1605*(1-test_ratio))

    alibox.split_AL(test_ratio=test_ratio, initial_label_rate=ini_lab_ratio, split_count=rounds)
        
    # Use the default Logistic Regression classifier
    model = alibox.get_default_model()
        
    # The cost budget is 50 times querying
    stopping_criterion = alibox.get_stopping_criterion('num_of_queries', num_of_queries)
        
    # Use pre-defined strategy
    strategy = alibox.get_query_strategy(strategy_name)
    results = []

    quality_results = np.empty((rounds,num_of_queries + 2))


    for round in range(rounds):
        j = 0
        # Get the data split of one fold experiment
        train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        # Get intermediate results saver for one fold experiment
        saver = alibox.get_stateio(round)
        
        # Set initial performance point
        # model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        model_copy = copy.deepcopy(model)
        model_copy.fit(X=X[train_idx], y=y[train_idx])
        pred = model_copy.predict(X[test_idx])
        max_accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                y_pred=pred,
                                                performance_metric='accuracy_score')
        quality_results[round,-1] = max_accuracy

        model.fit(X=X[label_ind.index], y=y[label_ind.index])
        pred = model.predict(X[test_idx])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                y_pred=pred,
                                                performance_metric='accuracy_score')
        accuracy /= max_accuracy
        quality_results[round,j] = accuracy
        saver.set_initial_point(accuracy)
        
        while not stopping_criterion.is_stop():
            j += 1
            # Select a subset of Uind according to the query strategy
            # Passing any sklearn models with proba_predict method are ok
            select_ind = strategy.select(label_ind, unlab_ind, model=model, batch_size=1)
        
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
        
            # Update model and calc performance according to the model you are using
            model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
            pred = model.predict(X[test_idx])
            accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                    y_pred=pred,
                                                    performance_metric='accuracy_score')
            accuracy /= max_accuracy
            # Save intermediate results to file
            st = alibox.State(select_index=select_ind, performance=accuracy)
            saver.add_state(st)
            saver.save()
            quality_results[round, j] = accuracy
        
            # Passing the current progress to stopping criterion object
            stopping_criterion.update_information(saver)
        # Reset the progress in stopping criterion object
        stopping_criterion.reset()
        results.append(copy.deepcopy(saver))
        
    np.save(save_path + "/" + save_name + ".npy", quality_results)