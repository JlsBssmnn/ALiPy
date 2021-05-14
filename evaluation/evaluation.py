import numpy as np
import copy
from os import listdir
from os.path import isfile, isdir, join
from alipy import ToolBox
from alipy.experiment import StateIO, ExperimentAnalyser

def run_experiment(X,y,strategies=["QueryInstanceUncertainty"],num_splits=5,num_of_queries=20,batch_size=1,
                    test_ratio=0.3, initial_label_rate=0.1, saving_path=None, show_results=True):
    n_strategies = len(strategies)
    if n_strategies < 1:
        raise KeyError("Number of stragies must be at least 1")

    alibox = ToolBox(X=X, y=y, saving_path=saving_path)
    alibox.split_AL(test_ratio=test_ratio, initial_label_rate=initial_label_rate, split_count=num_splits)

    results = []
    for i in range(n_strategies):
        results.append([])

    for round in range(num_splits):
        train, test_idx, label_ind, unlab_ind = alibox.get_split(round)
        saver = alibox.get_stateio(round)
        label_ind_list = np.empty(n_strategies, dtype=object)
        unlab_ind_list = np.empty(n_strategies, dtype=object)
        saver_list = np.empty(n_strategies, dtype=object)
        strategies_list = np.empty(n_strategies, dtype=object)
        select_ind_list = np.empty(n_strategies, dtype=object)

        model = alibox.get_default_model()
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred, performance_metric='accuracy_score')
        saver.set_initial_point(accuracy)

        for i in range(n_strategies):
            label_ind_list[i] = copy.deepcopy(label_ind)
            unlab_ind_list[i] = copy.deepcopy(unlab_ind)
            saver_list[i] = copy.deepcopy(saver)
            file_name = saver_list[i]._saving_file_name
            saver_list[i]._saving_file_name = file_name[:-4] + "_" + strategies[i][13:] + ".pkl"
            strategies_list[i] = alibox.get_query_strategy(strategy_name=strategies[i])


        stopping_criterion = alibox.get_stopping_criterion('num_of_queries', num_of_queries)

        while not stopping_criterion.is_stop():
            for i in range(n_strategies):
                select_ind = strategies_list[i].select(label_index=label_ind_list[i], unlabel_index=unlab_ind_list[i],
                                                                batch_size=batch_size)
                label_ind_list[i].update(select_ind)
                unlab_ind_list[i].difference_update(select_ind)

                model.fit(X=X[label_ind_list[i].index, :], y=y[label_ind_list[i].index])
                pred = model.predict(X[test_idx, :])
                accuracy = alibox.calc_performance_metric(y_true=y[test_idx],
                                                          y_pred=pred,
                                                          performance_metric='accuracy_score')

                # Save intermediate results to file
                st = alibox.State(select_index=select_ind, performance=accuracy)
                saver_list[i].add_state(st)
                saver_list[i].save()

            # Passing the current progress to stopping criterion object
            stopping_criterion.update_information(saver_list[0])
        
        stopping_criterion.reset()
        for i in range(n_strategies):
            results[i].append(copy.deepcopy(saver_list[i]))

    if show_results:
        analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
        for i in range(n_strategies):
            analyser.add_method(method_name=strategies[i][13:], method_results=results[i])
        print(analyser)
        analyser.plot_learning_curves(title='AL results', std_area=True)


def plot_results_from_directory(directory):
    if not isdir(directory):
        raise ValueError("directory parameter must be a directory")
    all_files = [f for f in listdir(directory) if isfile(join(directory, f)) and f.startswith("AL_round_")]
    
    results = []
    strategies = list(set([s[s.rfind("_")+1:-4] for s in all_files]))
    num_strategies = len(strategies)

    for i in range(num_strategies):
        current_strat_files = [s for s in all_files if s[s.rfind("_")+1:-4] == strategies[i]]
        results.append([])
        for stateIO_file_path in current_strat_files:
            results[-1].append(StateIO.load(join(directory, stateIO_file_path)))

    analyser = ExperimentAnalyser()
    for i in range(num_strategies):
        analyser.add_method(strategies[i], results[i])
    print(analyser)
    analyser.plot_learning_curves(title="Results", std_area=True, saving_path=None)
