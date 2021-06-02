import unittest
from evaluation.batchBALD_tests import BayesianNet, Model
import torch
import numpy as np


class Test_BayesianNet(unittest.TestCase):
    def test_fix_dropout(self):
        net = BayesianNet()
        data = torch.rand(1,1,28,28)
        data = torch.cat((data,)*10)

        output1 = net.forward(data)
        output2 = net.forward(data)

        for i in range(len(output1)):
            self.assertFalse(list(output1[i]) == list(output2[i]))
            self.assertFalse(list(output1[i]) == list(output1[(i+1)%len(output1)]))

        net.fix_dropout()
        
        output1 = net.forward(data)
        output2 = net.forward(data)

        for i in range(len(output1)):
            self.assertTrue(list(output1[i]) == list(output2[i]))
            self.assertFalse(list(output1[i]) == list(output1[(i+1)%len(output1)]))

        net.unfix_dropout()

        output1 = net.forward(data)
        output2 = net.forward(data)

        for i in range(len(output1)):
            self.assertFalse(list(output1[i]) == list(output2[i]))
            self.assertFalse(list(output1[i]) == list(output1[(i+1)%len(output1)]))

    def test_disable_dropout(self):
        net = BayesianNet()
        net.disable_dropout()
        data = torch.rand(1,1,28,28)
        data = torch.cat((data,)*10)

        output1 = net.forward(data)
        output2 = net.forward(data)

        for i in range(len(output1)):
            self.assertTrue(list(output1[i]) == list(output2[i]))
            self.assertTrue(list(output1[i]) == list(output1[(i+1)%len(output1)]))

        net.enable_dropout()
        output1 = net.forward(data)
        output2 = net.forward(data)

        for i in range(len(output1)):
            self.assertFalse(list(output1[i]) == list(output2[i]))
            self.assertFalse(list(output1[i]) == list(output1[(i+1)%len(output1)]))

    def test_predict_with_probabilities(self):
        net = BayesianNet()
        net.disable_dropout()
        data = torch.rand(10,1,28,28)

        output = torch.exp(net(data))
        for i in output.flatten():
            self.assertTrue(0 <= float(i) <= 1)

        for i in output:
            self.assertEqual(round(float(torch.sum(i)), 6), 1.0)


class Test_Model(unittest.TestCase):
    def test_model(self):
        features = 784
        samples = 100

        zero = np.zeros((int(samples/2), features))
        one = np.ones((int(samples/2), features))
        X = np.concatenate((zero,one), axis=0)
        y = np.concatenate((np.zeros(int(samples/2)), np.ones(int(samples/2))))

        data = np.concatenate((X,y[:,None]), axis=1)
        np.random.shuffle(data)

        X = data[:,:-1]
        y = data[:,-1]

        model = Model(BayesianNet())
        model.fit(X[:int(samples/2)],y[:int(samples/2)],20)

        pred = model.predict(X[int(samples/2):])
        acc = np.sum(pred == y[int(samples/2):]) / (samples/2)

        self.assertTrue(0.99 < acc <= 1)

    def test_prodict_proba(self):
        model = Model(BayesianNet())

        # test for proper shape
        X = np.random.rand(1,784)
        pred = model.predict_proba(X)
        self.assertEqual(pred.shape, (1,10,10))

        X = np.random.rand(5,784)
        pred = model.predict_proba(X)
        self.assertEqual(pred.shape, (5,10,10))
        
        X = np.random.rand(1,784)
        pred = model.predict_proba(X,K=5)
        self.assertEqual(pred.shape, (1,5,10))

        X = np.random.rand(7,784)
        pred = model.predict_proba(X,K=3)
        self.assertEqual(pred.shape, (7,3,10))

        # test for consistent dropout mask
        X = np.random.rand(1,784)
        X = np.concatenate((X,)*7)
        pred = model.predict_proba(X)

        for i, ten in enumerate(pred[0]):
            for j in range(pred.shape[0]):
                self.assertEqual(list(ten), list(pred[j,i]))
            self.assertNotEqual(list(ten), list(pred[0,(i+1)%10]))

if __name__ == "__main__":
    unittest.main()
