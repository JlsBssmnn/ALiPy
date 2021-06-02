import numpy as np
import copy
from os import listdir
from os.path import isfile, isdir, join, abspath
from alipy import ToolBox
from alipy.experiment import StateIO, ExperimentAnalyser
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class ExperimentRunner:
    def __init__(self,X,y,saving_path=None):
        self.X = X
        self.y = y
        self.alibox = ToolBox(X=X, y=y, saving_path=saving_path)

    def run_one_strategy(self, strategy,num_splits=5,num_of_queries=None,max_acquired_data=None,batch_size=1,
                        test_ratio=0.3, initial_label_rate=0.1, model=None, file_name=None, reset_model=False,
                        fit_strategy=None, custom_query_strat=None, device=None, equal_inst_per_class=False):
        """
            strategy: Name of the AL strategy that will be applied
            num_splits: how many AL rounds will be performed, the result of each round will be saved in a seperate file
            num_of_queries: the maximum number of queries until the round will end (max_acquired_data must be None)
            max_acquired_data: how many datapoints will be queried until the round will end (num_of_queries must be None)
            batch_size: the batch size of one query
            test_ratio: the ratio of the dataset that will be used for testing
            initial_label_rate: the ratio of the non test dataset that will be labeled at the beginning
            model: the model that will be trained, used for accuracy measure and will be passed to the query strategy
            file_name: the name for the files that will safe the results of the experiment 
                        (the round number will be added to the name)
            reset_model: if True the model will be reset in every query step
            fit_strategy: the strategy that will be used to train the model
            custom_query_strat: an object with select method that will be called when determining the datapoint that will
                                be queried
            device: the pytorch device
            equal_inst_per_class: If True it will try to modify the labeled index at the beginning of each round to 
                                  contain the same number of samples for each class
        """
        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=initial_label_rate, split_count=num_splits)

        if equal_inst_per_class:
            self.equalize_inst_per_class()

        num_of_queries = self.calc_num_of_queries(num_of_queries, max_acquired_data, batch_size)
        self.model = model

        for round in range(num_splits):
            train, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            saver = self.alibox.get_stateio(round)

            model_copy = copy.deepcopy(self.model)
            if model_copy == None:
                model_copy = self.alibox.get_default_model()

            model_copy = self.fit_strategy(model_copy, label_ind, fit_strategy, device)

            if device == None:
                pred = model_copy.predict(self.X[test_idx, :])
            else:
                pred = model_copy.predict(self.X[test_idx, :], device=device)

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
                select_ind = self.run_one_query(custom_query_strat, query_strategy, label_ind, unlab_ind, 
                                                batch_size, model_copy, device)

                if reset_model:
                    model_copy = copy.deepcopy(self.model)
                # use the default fit strategy or the provided custom one
                model_copy = self.fit_strategy(model_copy, label_ind, fit_strategy, device)
                
                if device == None:
                    pred = model_copy.predict(self.X[test_idx, :])
                else:
                    pred = model_copy.predict(self.X[test_idx, :], device=device)
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

    def fit_strategy(self, model, label_ind, test_idx, fit_strategy=None, device=None):
        if fit_strategy == None:
            if device == None:
                model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index])
            else:
                model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index], device=device)
        elif fit_strategy == "batchBALD":
            model = self.batchBALD_fit_strategy(model, label_ind, test_idx, device=device)

        return model

    # described on page 6 of the paper
    def batchBALD_fit_strategy(self, model, label_ind, test_idx, device=None):
        max_epochs = 1000
        num_declines = 0
        last_accuracy = -1
        best_model = None

        for _ in tqdm(range(max_epochs), desc="Training Model", leave=False):
            model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index], device=device)
            pred = model.predict(self.X[test_idx, :], device=device)
            accuracy = self.alibox.calc_performance_metric(y_true=self.y[test_idx],
                                                    y_pred=pred,
                                                    performance_metric='accuracy_score')
            if accuracy < last_accuracy:
                num_declines += 1
            else:
                best_model = copy.deepcopy(model)
                num_declines = 0
            if num_declines >= 3:
                break
            last_accuracy = accuracy
        
        return best_model

    def run_one_query(self, custom_query_strat, query_strategy, label_ind, unlab_ind, batch_size, model_copy, device):
        if custom_query_strat == None:
            select_ind = query_strategy.select(label_index=label_ind, unlabel_index=unlab_ind,
                                                        batch_size=batch_size, model=model_copy)
        else:
            select_ind = custom_query_strat.select(label_ind, unlab_ind, batch_size, model_copy, query_strategy, device=device)

        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        return select_ind

    def equalize_inst_per_class(self):
        train,_,label,unlabel = self.alibox.get_split()
        length_of_label = len(label[0])
        num_classes = len(np.unique(self.y))

        if length_of_label % num_classes != 0:
            print("Cannot equalize initial label state, because lenght of labeled index is not divisible by number of classes")
            return

        instances_per_class = length_of_label // num_classes
        d = self.reset_dic(num_classes)

        for split_round, label_round in enumerate(label):
            labels = self.y[label_round]
            
            # check number of samples per class
            for l in labels:
                d[l] += 1
            
            # not all classes have the same number of instances
            if max(d.values()) > instances_per_class:
                for k,v in d.items():
                    if v <= instances_per_class:
                        continue
                    replacements = v - instances_per_class
                    i = 0
                    while replacements > 0:
                        if labels[i] == k:
                            # find a class that has not enough instances
                            replacement_label = list(dict(filter(lambda x: x[1] < instances_per_class, d.items())).keys())[0]
                            
                            # find an index of a sample that belongs to this class
                            index = -1
                            for ind in train[split_round]:
                                if ind not in label_round and self.y[ind] == replacement_label:
                                    index = ind
                                    break
                            assert index != -1, "didn't find an index in train_index to use for replacement"
                            
                            # replace the index and update the data structures
                            old_index = label_round[i]
                            label_round[i] = ind
                            d[k] -= 1
                            d[replacement_label] += 1
                            replacements -= 1

                            # update the unlabeled index
                            for j in range(len(unlabel[split_round])):
                                if unlabel[split_round][j] == ind:
                                    break
                            unlabel[split_round][j] = old_index
                                
                        i += 1
            d = self.reset_dic(num_classes)
        self.alibox.label_idx = label
        self.alibox.unlabel_idx = unlabel

    def reset_dic(self, classes):
        dic = dict()
        for i in range(classes):
            dic[i] = 0
        return dic

    def calc_num_of_queries(self, num_of_queries, max_acquired_data, batch_size):
        if (num_of_queries == None and max_acquired_data == None) or (num_of_queries != None and max_acquired_data != None):
            raise ValueError("Either num_of_queries or max_acquired_data must be None and the other must not be None")
        if num_of_queries == None:
            num_of_queries = int(np.ceil(max_acquired_data / batch_size))

        return num_of_queries
        

class ExperimentPlotter:
    def plot_by_given_prefixes(self, directory, prefixes, labels=None, x_axis='num_of_queries', batch_size=None):
        """
        Plots the results that are found in the given directory but only considers files that start
        with a prefix inside of the prefixes list. Files that start with the same prefix are
        grouped into one line. If a list of labels is provided the plot will use these as name for
        the lines instead of the prefixes. So it's required that 
        >>> len(prefixes) == len(labels)
        
        Parameters
        ----------
            directory: the directory that will be searched for files
            prefixes: a list of strings
            labels: labels for the plot lines
            x_axis: either 'num_of_queries' (default) or 'labeled_data' for the metric on the x_axis
            batch_size: only required for 'labeled_data', a list of the batch sizes corresponding to the files that match prefixes
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
