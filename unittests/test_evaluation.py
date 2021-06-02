import copy
import unittest
from evaluation import evaluation, data_extraction
import numpy as np

class Test_ExperimentRunner(unittest.TestCase):

    def test_batchBALD_fit_strategy(self):
        runner = evaluation.ExperimentRunner(*self.create_X_y())
        model = Model()
        test_ind = list(range(10))
        model = runner.batchBALD_fit_strategy(model, LabelIndex(), test_ind)

        self.assertEqual(model.fitcount, 5)
        self.assertEqual(list(model.predict(3,4)), list(np.array([0,1,2,3,4,6,7,8,9,5])))

    def test_equalize_inst_per_class1(self):
        X, y = data_extraction.Iris().getData("../../datasets/Iris/iris.data")
        runner = evaluation.ExperimentRunner(X,y)
        runner.alibox.split_AL(test_ratio=1/3, initial_label_rate=9/100, split_count=20)
        
        runner.equalize_inst_per_class()

        train, _, label, unlabel = runner.alibox.get_split()

        d = runner.reset_dic(3)

        for i, label_round in enumerate(label):
            labels = y[label_round]
            for l in labels:
                d[l] += 1

            self.assertEqual(len(np.unique(list(d.values()))), 1)
            self.assertEqual(list(d.values())[0], 3)

            self.assertTrue(set(label_round).isdisjoint(set(unlabel[i])))
            self.assertEqual(set(label_round).union(set(unlabel[i])), set(train[i]))
            d = runner.reset_dic(3)

    def test_equalize_inst_per_class2(self):
        X, y = data_extraction.Iris().getData("../../datasets/Iris/iris.data")
        runner = evaluation.ExperimentRunner(X,y)
        runner.alibox.split_AL(test_ratio=1/3, initial_label_rate=18/100, split_count=20)
        
        runner.equalize_inst_per_class()

        train, _, label, unlabel = runner.alibox.get_split()

        d = runner.reset_dic(3)

        for i, label_round in enumerate(label):
            labels = y[label_round]
            for l in labels:
                d[l] += 1

            self.assertEqual(len(np.unique(list(d.values()))), 1)
            self.assertEqual(list(d.values())[0], 6)

            self.assertTrue(set(label_round).isdisjoint(set(unlabel[i])))
            self.assertEqual(set(label_round).union(set(unlabel[i])), set(train[i]))
            d = runner.reset_dic(3)

    def test_fit_strategy(self):
        model = Model()
        runner = evaluation.ExperimentRunner(*self.create_X_y())

        runner.fit_strategy(model, LabelIndex(), LabelIndex())
        self.assertEqual(model.fitcount, 1)
        runner.fit_strategy(model, LabelIndex(), LabelIndex())
        self.assertEqual(model.fitcount, 2)

        model = Model()
        model = runner.fit_strategy(model, LabelIndex(), list(range(10)), "batchBALD")
        self.assertEqual(model.fitcount, 5)
        self.assertEqual(list(model.predict(3,4)), list(np.array([0,1,2,3,4,6,7,8,9,5])))

    def test_calc_num_of_queries(self):
        runner = evaluation.ExperimentRunner(*self.create_X_y())
        self.assertRaises(ValueError, runner.calc_num_of_queries, None, None, 5)
        self.assertRaises(ValueError, runner.calc_num_of_queries, 10, 200, 5)

        self.assertEqual(runner.calc_num_of_queries(23,None,1), 23)
        self.assertEqual(runner.calc_num_of_queries(27,None,4), 27)
        self.assertEqual(runner.calc_num_of_queries(35,None,400), 35)

        self.assertEqual(runner.calc_num_of_queries(None,250,5), 50)
        self.assertEqual(runner.calc_num_of_queries(None,250,10), 25)
        self.assertEqual(runner.calc_num_of_queries(None,250,40), 7)
        self.assertEqual(runner.calc_num_of_queries(None,300,1000), 1)

    def test_run_one_query(self):
        label_ind = LabelIndex2([0,1,2,3])
        unlabel_ind = UnlabelIndex([4,5,6,7,8,9,10])
        runner = evaluation.ExperimentRunner(*self.create_X_y())
        model = Model()
        runner.model = model
        model_copy = copy.deepcopy(model)
        query_strat = QueryStrat()

        select_ind = runner.run_one_query(query_strat, None, label_ind, unlabel_ind, 2, model_copy, None)

        self.assertEqual(label_ind.indices, [0,1,2,3,9,10])
        self.assertEqual(unlabel_ind.indices, [4,5,6,7,8])
        self.assertEqual(select_ind, [9,10])

        select_ind = runner.run_one_query(query_strat, None, label_ind, unlabel_ind, 3, model_copy, None)

        self.assertEqual(label_ind.indices, [0,1,2,3,9,10,6,7,8])
        self.assertEqual(unlabel_ind.indices, [4,5])
        self.assertEqual(select_ind, [6,7,8])

        select_ind = runner.run_one_query(query_strat, None, label_ind, unlabel_ind, 5, model_copy, None)

        self.assertEqual(label_ind.indices, [0,1,2,3,9,10,6,7,8,4,5])
        self.assertEqual(unlabel_ind.indices, [])
        self.assertEqual(select_ind, [4,5])

    def create_X_y(self):
        X = np.random.rand(10,3)
        y = np.array(range(10))
        return (X,y)



class Model:

    def __init__(self):
        self.fitcount = 0
        self.pred = np.array([[1,2,3,4,5,6,7,8,9,0],
                              [0,2,3,4,5,6,7,8,9,1],
                              [1,2,3,4,5,6,7,8,9,0],
                              [0,1,3,4,5,6,7,8,9,2],
                              [0,1,2,4,5,6,7,8,9,3],
                              [0,1,2,3,4,6,7,8,9,5], # Point of highest accuracy
                              [0,1,2,3,5,6,7,8,9,4],
                              [0,1,2,4,5,6,7,8,9,3],
                              [0,1,3,4,5,6,7,8,9,2]])
    
    def fit(self, **kwargs):
        self.fitcount += 1

    def predict(self, data, device):
        return self.pred[self.fitcount]


class LabelIndex:
    def __init__(self) -> None:
        self.index = 3  # it's just important to have a field named index

class LabelIndex2:
    def __init__(self, indices) -> None:
        self.indices = indices

    def update(self, ind):
        self.indices += ind


class UnlabelIndex:
    def __init__(self, indices) -> None:
        self.indices = indices

    def difference_update(self, ind):
        self.indices = [i for i in self.indices if i not in ind]


class QueryStrat:
    def select(self, label_ind, unlab_ind, batch_size, model_copy, query_strategy, device):
        return unlab_ind.indices[-batch_size:]


if __name__ == "__main__":
    unittest.main()