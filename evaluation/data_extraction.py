import random
import numpy as np
from tensorflow.keras.datasets import mnist

class Iris:
    def getData(self):
        dataFile = open("././datasets/Iris/iris.data")
        X = np.ndarray(shape=(150,4))
        y = np.ndarray(shape=(150), dtype=int)
        i = 0

        lines = dataFile.read().splitlines()
        random.shuffle(lines)

        for line in lines:
            attr = line.split(',')
            for j in range(4):
                X[i][j] = float(attr[j])
            y[i] = self.getLabel(attr[-1])
            i += 1

        dataFile.close()
        return X,y

    def getLabel(self, labelString):
        if labelString == "Iris-setosa":
            return 0
        elif labelString == "Iris-versicolor":
            return 1
        elif labelString == "Iris-virginica":
            return 2
        else:
            return -1


class MNist:
    def getData(self):
        train, test = mnist.load_data()
        X = np.concatenate((train[0], test[0]))
        y = np.concatenate((train[1], test[1]))
        X = X.reshape((70000, 784))

        return X,y