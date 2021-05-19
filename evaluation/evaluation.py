import numpy as np
import copy
from os import listdir
from os.path import isfile, isdir, join, abspath
from alipy import ToolBox
from alipy.experiment import StateIO, ExperimentAnalyser
from datetime import datetime
import matplotlib.pyplot as plt
import time

class ExperimentRunner:
    def __init__(self,X,y,saving_path=None):
        self.X = X
        self.y = y
        self.alibox = ToolBox(X=X, y=y, saving_path=saving_path)

    def run_one_strategy(self, strategy,num_splits=5,num_of_queries=None,max_acquired_data=None,batch_size=1,
                        test_ratio=0.3, initial_label_rate=0.1, model=None, file_name=None):
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=initial_label_rate, split_count=num_splits)

        if (num_of_queries == None and max_acquired_data == None) or (num_of_queries != None and max_acquired_data != None):
            raise ValueError("Either num_of_queries or max_acquired_data must be None and the other must not be None")
        if num_of_queries == None:
            num_of_queries = int(np.ceil(max_acquired_data / batch_size))

        for round in range(num_splits):
            train, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            saver = self.alibox.get_stateio(round)

            if model == None:
                model = self.alibox.get_default_model()
            model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index])
            pred = model.predict(self.X[test_idx, :])
            accuracy = self.alibox.calc_performance_metric(y_true=self.y[test_idx], y_pred=pred, performance_metric='accuracy_score')
            saver.set_initial_point(accuracy)

            saver_file_name = saver._saving_file_name
            if file_name == None:
                saver._saving_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + saver_file_name[3:-4] + "_" + strategy[13:] + ".pkl"
            else:
                saver._saving_file_name =  file_name + "_" + saver_file_name[3:-4] + ".pkl"
            query_strategy = self.alibox.get_query_strategy(strategy_name=strategy)

            stopping_criterion = self.alibox.get_stopping_criterion('num_of_queries', num_of_queries)

            while not stopping_criterion.is_stop():
                select_ind = query_strategy.select(label_index=label_ind, unlabel_index=unlab_ind,
                                                                batch_size=batch_size)
                label_ind.update(select_ind)
                unlab_ind.difference_update(select_ind)

                model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index])
                pred = model.predict(self.X[test_idx, :])
                accuracy = self.alibox.calc_performance_metric(y_true=self.y[test_idx],
                                                        y_pred=pred,
                                                        performance_metric='accuracy_score')

                # Save intermediate results to file
                st = self.alibox.State(select_index=select_ind, performance=accuracy)
                saver.add_state(st)
                saver.save()

                # Passing the current progress to stopping criterion object
                stopping_criterion.update_information(saver)
            
            stopping_criterion.reset()


class ExperimentPlotter:
    def plot_by_given_prefixes(self, directory, prefixes, labels=None, x_axis='num_of_queries', batch_size=None):
        """
        Plots the results that are found in the given directory but only considers files that start
        with a prefix inside of the prefixes list. Files that start with the same prefix are
        grouped into one line. If a list of labels is provided the plot will use these as name for
        the lines instead of the prefixes. So it's required that 
        >>> len(prefixes) == len(labels)
        """
        if labels != None and len(prefixes) != len(labels):
            raise ValueError("it's required that len(prefixes) == len(labels)")
        if batch_size == None:
            batch_size = [1]*len(prefixes)
        results = self.get_results_by_given_prefixes(directory, prefixes)

        analyser = ExperimentAnalyser()
        if labels == None:
            for i in range(len(prefixes)):
                analyser.add_method(prefixes[i], results[i])
        else:
            for i in range(len(prefixes)):
                analyser.add_method(labels[i], results[i])

        if x_axis == 'num_of_queries':
            self.plot_by_queries(analyser)
        elif x_axis == 'labeled_data':
            self.plot_by_labeled_data(analyser, prefixes if labels == None else labels, batch_size)

    def get_results_by_given_prefixes(self, directory, prefixes):
        if not isdir(directory):
            raise ValueError("directory parameter must be a directory")

        directory = abspath(directory)
        all_files = [f for f in listdir(directory) if isfile(join(directory, f))]
        # all_files = filter(lambda s: len([pre for pre in prefixes if s.startswith(pre)]) > 0, all_files)
        file_dict = dict()
        for file_name in all_files:
            for pre in prefixes:
                if file_name.startswith(pre):
                    if pre in file_dict:
                        file_dict[pre].append(file_name)
                    else:
                        file_dict[pre] = [file_name]
    
        results = []
        num_of_lines = len(prefixes)

        for i in range(num_of_lines):
            current_files = list(file_dict[prefixes[i]])
            results.append([])
            for stateIO_file_path in current_files:
                results[-1].append(StateIO.load(join(directory, stateIO_file_path)))

        return results

    def plot_by_queries(self, analyser):
        print(analyser)
        analyser.plot_learning_curves(title="Results", std_area=True, saving_path=None)
    
    def plot_by_labeled_data(self, analyser, method_names, batch_sizes):
        values = []
        std_values = []
        for j in range(len(method_names)):
            method_data = analyser.get_extracted_data(method_names[j])
            values.append(np.mean(method_data, axis=0))
            std_values.append(np.std(method_data, axis=0))

        for j in range(len(method_names)):
            x_axis = np.arange(0, (len(values[j])-1)*batch_sizes[j] + 1, batch_sizes[j],int)
            plt.plot(x_axis, values[j], label=method_names[j])
            plt.fill_between(x_axis, values[j] + std_values[j], values[j] - std_values[j], alpha=0.3)
            
        plt.xlabel("Acquired dataset size")
        plt.ylabel("Accuracy")
        plt.title("Results")
        plt.legend(fancybox=True, framealpha=0.5)
        plt.show()


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
