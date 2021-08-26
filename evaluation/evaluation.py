import os
import pickle
import numpy as np
import copy
from os import listdir
from os.path import isfile, isdir, join, abspath
import pandas as pd
from prettytable import PrettyTable

import sklearn
from sklearn import metrics
from alipy import ToolBox
from alipy.experiment.al_experiment import AlExperiment
from alipy.query_strategy import query_labels
from alipy.experiment import StateIO, ExperimentAnalyser
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from alipy.metrics import performance
from alipy.query_strategy.LAL_RL.Test_AL import check_performance, check_performance_for_figure

class ExperimentRunner:
    def __init__(self,X,y,saving_path=None, dataset_name=None):
        self.X = X
        self.y = y
        self.saving_path = saving_path
        self.dataset_name = dataset_name

    @staticmethod
    def initialize_with_csv(csv_path, saving_path, header=0, delimiter=',', label_columnn_name=None,
                            normalize=True):
        """
        This method returns an ExperimentRunner by extracting the X and y values out of the given csv file

        csv_path: the path to the csv file
        saving_path: the path where the results of the experiments will be stored
        header: row numbers which represent column names
        delimiter: the csv delimiter
        label_column_name: the name of the column that contains the labels, if None is provided the last column is chosen
        normalize: whether to min-max-scale the samples
        """
        csv_dataframe = pd.read_csv(csv_path, header=header, delimiter=delimiter)
        if label_columnn_name != None:
            # move the column with the labels to the end
            csv_dataframe = csv_dataframe[[x for x in csv_dataframe if x!=label_columnn_name] + [label_columnn_name]]
        X = csv_dataframe.to_numpy()[:, :-1]
        y = csv_dataframe.to_numpy()[:, -1]

        if type(y[0]) == str:
            # convert the labels to integers
            _, y = np.unique(y, return_inverse=True)
        if normalize:
            X = sklearn.preprocessing.minmax_scale(X)
        return ExperimentRunner(X=X,y=y,saving_path=saving_path)

    def run_one_strategy(self, strategy,num_splits=5,num_of_queries=None,max_acquired_data=None,batch_size=1,
                        test_ratio=0.3, initial_label_rate=0.1, model=None, file_name=None, reset_model=False,
                        fit_strategy=None, custom_query_strat=None, device=None, equal_inst_per_class=False,
                        performance_metric='accuracy_score', log_timing=False):
        """
            strategy: Name of the AL strategy that will be applied
            num_splits: how many AL rounds will be performed, the result of each round will be saved in a seperate file
            num_of_queries: the maximum number of queries until the round will end (max_acquired_data must be None)
            max_acquired_data: how many datapoints will be queried until the round will end (num_of_queries must be None)
            batch_size: the batch size of one query
            test_ratio: the ratio of the dataset that will be used for testing
            initial_label_rate: the ratio of the non test dataset that will be labeled at the beginning
                                if this is 'min' then the amount of labeled data will be equal to the number of classes
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
            log_timing: whether to create a file that contains timing information or not
        """
        if len([f for f in listdir(self.saving_path) if f == file_name]) > 0:
            raise ValueError("There is already either a file or directory, that has with the given file_name")
        os.mkdir(join(self.saving_path, file_name))
        self.alibox = ToolBox(X=self.X, y=self.y, saving_path=join(self.saving_path, file_name))

        if log_timing:
            self.time_file = open(join(self.alibox._saving_path, file_name + "_time_info.txt"), "x")
            start = datetime.now()
            self.time_file.write(start.strftime("Start of the experiment: %d.%m.%Y - %H:%M:%S\n"))

        # this way the initial point for AL will contain one labeled sample per class
        if initial_label_rate == 'min':
            initial_label_rate = len(np.unique(self.y))/(len(self.y)*(1-test_ratio))

        performance_metric_param = dict()
        if performance_metric == "f1_score" and len(np.unique(self.y)) > 2:
            performance_metric_param['average'] = "weighted"

        self.alibox.split_AL(test_ratio=test_ratio, initial_label_rate=initial_label_rate, split_count=num_splits)

        if equal_inst_per_class:
            self.equalize_inst_per_class()

        num_of_queries = self.calc_num_of_queries(num_of_queries, max_acquired_data, batch_size)
        self.model = model

        for round in tqdm(range(num_splits), desc="AL rounds", leave=False):
            start_of_round = datetime.now()
            train, test_idx, label_ind, unlab_ind = self.alibox.get_split(round)
            saver = self.alibox.get_stateio(round, verbose=False)

            model_copy = copy.deepcopy(self.model)
            if model_copy == None:
                model_copy = self.alibox.get_default_model()

            model_copy = self.fit_strategy(model_copy, label_ind, test_idx, fit_strategy, performance_metric,
                performance_metric_param, device)

            if device == None:
                pred = model_copy.predict(self.X[test_idx, :])
            else:
                pred = model_copy.predict(self.X[test_idx, :], device=device)

            accuracy = self.alibox.calc_performance_metric(y_true=self.y[test_idx], y_pred=pred,
                performance_metric=performance_metric, **performance_metric_param)
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
                model_copy = self.fit_strategy(model_copy, label_ind, test_idx, fit_strategy, performance_metric,
                    performance_metric_param, device)
                
                if device == None:
                    pred = model_copy.predict(self.X[test_idx, :])
                else:
                    pred = model_copy.predict(self.X[test_idx, :], device=device)
                accuracy = self.alibox.calc_performance_metric(y_true=self.y[test_idx],
                                                        y_pred=pred,
                                                        performance_metric=performance_metric,
                                                        **performance_metric_param)

                # Save intermediate results to file
                st = self.alibox.State(select_index=select_ind, performance=accuracy)
                saver.add_state(st)
                saver.save()

                # Passing the current progress to stopping criterion object
                stopping_criterion.update_information(saver)
            
            stopping_criterion.reset()
            if log_timing:
                end_of_round = datetime.now()
                diff = str(end_of_round - start_of_round)
                self.time_file.write(f"\tDuration of round {round} --{diff[:diff.rfind('.')]}--\n")
        if log_timing:
            end = datetime.now()
            diff = str(end - start)
            self.time_file.write(end.strftime("End of the experiment: %d.%m.%Y - %H:%M:%S\n"))
            self.time_file.write(f"Duration of experiment {diff[:diff.rfind('.')]}\n")
            self.time_file.close()

    def fit_strategy(self, model, label_ind, test_idx, fit_strategy=None, performance_metric='accuracy_score',
                     performance_metric_param=dict(), device=None):
        if fit_strategy == None:
            if device == None:
                model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index])
            else:
                model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index], device=device)
        elif fit_strategy == "batchBALD":
            model = self.batchBALD_fit_strategy(model, label_ind, test_idx, performance_metric,
                performance_metric_param, device)

        return model

    # described on page 6 of the paper
    def batchBALD_fit_strategy(self, model, label_ind, test_idx, performance_metric='accuracy_score',
                               performance_metric_param=dict(), device=None):
        max_epochs = 1000
        num_declines = 0
        last_accuracy = -1
        best_accuracy = -1
        best_model = None

        for _ in tqdm(range(max_epochs), desc="Training Model", leave=False):
            model.fit(X=self.X[label_ind.index, :], y=self.y[label_ind.index], device=device)
            pred = model.predict(self.X[test_idx, :], device=device)
            accuracy = self.alibox.calc_performance_metric(y_true=self.y[test_idx],
                                                    y_pred=pred,
                                                    performance_metric=performance_metric,
                                                    **performance_metric_param)
            if accuracy < last_accuracy:
                num_declines += 1
            elif accuracy > best_accuracy:
                best_model = copy.deepcopy(model)
                best_accuracy = accuracy
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
        
    def evaluation(self, query_strat, model=None, model_path=None):
        # write time information
        self.time_file = open(join(self.saving_path, "time_info.txt"), "x")
        start = datetime.now()
        self.time_file.write(start.strftime("Start of the experiment: %d.%m.%Y - %H:%M:%S\n"))

        test_ratio = 0.5
        initial_label_rate = len(np.unique(self.y))/(len(self.y)*(1-test_ratio))
        batch_size = 5
        if self.dataset_name in ["CIFAR10", "EMNIST"]:
            num_of_queries = 1000
        else:
            num_of_queries = 50

        train_idx, test_idx, label_idx, unlabel_idx = self.split()

        ex = AlExperiment(self.X,self.y,
            model=sklearn.ensemble.RandomForestClassifier() if model == None else model,
            stopping_criteria="num_of_queries",
            stopping_value=num_of_queries,
            batch_size=batch_size,
            train_idx=train_idx,
            test_idx=test_idx,
            label_idx=label_idx,
            unlabel_idx=unlabel_idx)
        if query_strat == "QueryInstanceBatchBALD":
            ex.set_query_strategy(getattr(query_labels, query_strat), model=ex._model, num_samples=10000, verbose=0)
        elif query_strat == "QueryInstanceLAL_RL":
            ex.set_query_strategy(getattr(query_labels, query_strat), model=ex._model, model_path=model_path)
        else:
            ex.set_query_strategy(getattr(query_labels, query_strat))

        # set f1 score as the metric
        ex._performance_metric_name = "f1_score"
        ex._performance_metric = f1_score
        ex._metrics = True

        ex.start_query(False, saving_path=self.saving_path, verbose=False)

        # write time information
        end = datetime.now()
        diff = str(end - start)
        self.time_file.write(end.strftime("End of the experiment: %d.%m.%Y - %H:%M:%S\n"))
        self.time_file.write(f"Duration of experiment {diff[:diff.rfind('.')]}\n")
        self.time_file.close()

    def split(self, splitcount=100):
        train_idx = []
        test_idx = []
        label_idx = []
        unlabel_idx = []

        for _ in range(splitcount):
            successful_test_split = False
            while not successful_test_split:
                indecies = np.arange(len(self.y))
                test_ind = np.random.choice(len(indecies), int(len(indecies)/2), False)
                train = [x for x in indecies if x not in test_ind]
                if len(np.unique(self.y[train])) == len(np.unique(self.y)):
                    successful_test_split = True

            unlabel_ind = copy.deepcopy(train)
            label_ind = []

            for label in np.unique(self.y):
                init_labeled_index = np.random.choice(np.intersect1d(np.where(self.y == label)[0], train), 1)[0]
                label_ind.append(init_labeled_index)
                unlabel_ind.remove(init_labeled_index)
            
            train_idx.append(copy.deepcopy(train))
            test_idx.append(copy.deepcopy(test_ind))
            label_idx.append(copy.deepcopy(label_ind))
            unlabel_idx.append(copy.deepcopy(unlabel_ind))
        return train_idx, test_idx, label_idx, unlabel_idx

class ExperimentPlotter:
    def plot_by_given_prefixes(self, directory, prefixes, labels=None, x_axis='num_of_queries', batch_size=None,
        plot_exact_values=False, fill="std"):
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
            return self.plot_by_labeled_data(analyser, prefixes if labels == None else labels, batch_size,
                plot_exact_values=plot_exact_values, fill=fill)

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
    
    def plot_by_labeled_data(self, analyser, method_names, batch_sizes, plot_exact_values=False, fill="std"):
        values = []
        std_values = []
        auc = dict()
        for j in range(len(method_names)):
            method_data = analyser.get_extracted_data(method_names[j])
            values.append(np.mean(method_data, axis=0))
            std_values.append(np.std(method_data, axis=0))

        for j in range(len(method_names)):
            x_axis = np.arange(0, (len(values[j])-1)*batch_sizes[j] + 1, batch_sizes[j],int)
            plt.plot(x_axis, values[j], label=method_names[j])
            if fill == "standard_error":
                std_values[j] /= np.sqrt(values[j].shape[0])
            plt.fill_between(x_axis, values[j] + std_values[j], values[j] - std_values[j], alpha=0.3)
            auc[method_names[j]] = metrics.auc(x_axis, values[j]) / metrics.auc(x_axis, np.ones(len(x_axis)))

            if plot_exact_values:
                print(method_names[j] + ":")
                t = PrettyTable()
                t.add_column("x_axis", x_axis)
                t.add_column("y_axis", values[j])
                print(t)
            
        print("The auc-scores:\n", auc)
        plt.xlabel("Acquired dataset size")
        plt.ylabel("Accuracy")
        plt.title("Results")
        plt.legend(fancybox=True, framealpha=0.5)
        plt.show()
        return values

    def plot_numpy_array(self, directory, file_names, num_queries=None, fill="std", plot_exact_values=False):
        if type(num_queries) == int:
            num_queries = [num_queries]*len(file_names)

        auc = dict()
        for i,name in enumerate(file_names):
            data, std = self.get_numpy_array_data(directory+"/"+name+".npy",
                        num_queries[i] if num_queries != None else None)
            x_axis = np.arange(num_queries[i]+1 if num_queries != None else len(data), dtype=int)
            plt.plot(x_axis, data, label=name)
            if fill == "std":
                plt.fill_between(x_axis, data+std, data-std, alpha=0.3)
            elif fill == "standard_error":
                std /= np.sqrt(data.shape[0])
                plt.fill_between(x_axis, data+std, data-std, alpha=0.3)
            auc[name] = metrics.auc(x_axis, data) / metrics.auc(x_axis, np.ones(len(x_axis)))

        if plot_exact_values:
            print(name + ":")
            t = PrettyTable()
            t.add_column("x_axis", x_axis)
            t.add_column("y_axis", data)
            print(t)

        print("The auc-scores:\n", auc)

        plt.xlabel("Number of queries")
        plt.ylabel("Target quality")
        plt.title("Results")
        plt.legend(fancybox=True, framealpha=0.5)
        plt.show()

    def get_numpy_array_data(self, path, num_queries=None):
        array = np.load(path)
        if num_queries == None:
            return np.mean(array, axis=0), np.std(array, axis=0)
        else:
            return np.mean(array[:,:num_queries+1], axis=0), np.std(array[:,:num_queries+1], axis=0)

    def plot_LAL_RL_scores(self, directory, file_name):
        all_scores = pickle.load(open(directory + "/" + file_name + ".p", "rb"))
        i = 0

        for strat, scores in all_scores.items():
            m_line = np.mean(scores, axis=0)
            var_line = np.var(scores, axis=0)
            if i == 0:
                plt.plot(m_line, label = strat, color="blue")
                plt.fill_between(range(np.size(m_line)), m_line - var_line, m_line + var_line, alpha=0.3, color="blue")
            elif i == 1:
                plt.plot(m_line, label = strat, color="black")
                plt.fill_between(range(np.size(m_line)), m_line - var_line, m_line + var_line, alpha=0.3, color="black")
            elif i == 2:
                plt.plot(m_line, label = "LAL-RL", color="red")
                plt.fill_between(range(np.size(m_line)), m_line - var_line, m_line + var_line, alpha=0.3, color="red")
            else:
                plt.plot(m_line, label = strat)
                plt.fill_between(range(np.size(m_line)), m_line - var_line, m_line + var_line, alpha=0.3)
            i += 1
        plt.xlabel("number of annotations")
        plt.ylabel("% of target quality")
        plt.title(file_name)
        plt.legend(fancybox=True, framealpha=0.5)
        plt.show()

    def alipy_states_to_numpy(self, directory, destination, name):
        """
        This method takes all files in the given directory that contain 'round' and end with .pkl
        and interprets them as ALiPy state objects. The content of these objects is read and
        aggregated to one learning curve which is saved as a numpy array at the given destination
        """
        files = os.listdir(directory)
        files = [x for x in files if "round" in x and x.endswith(".pkl")]
        all_states = [StateIO.load(join(directory, x)) for x in files]
        analyser = ExperimentAnalyser()
        analyser.add_method("whatever", all_states)

        method_data = analyser.get_extracted_data("whatever")
        method_data = np.array(method_data)

        np.save(join(destination, name), method_data)

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


def f1_score(y_true, y_pred):
    """
    This function computes the f1 score with fixed parameters, the result is equal to
    >>> sklearn.metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    """
    return metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
