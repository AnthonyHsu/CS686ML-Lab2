from classifier import classifier
import numpy as np
from svmMLiA import calcWs, smoPK

class svm_basic(classifier):
    def __init__(self):
        pass
    # refer to svmMLiA
    def loadDataSet(self, fileName):
        dataMat = []; labelMat = []
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat, labelMat
    # import svmMLiA
    def fit(self, dataMatIn, classLabels):
        b, alphas = smoPK(dataMatIn, classLabels, 0.6, 0.001, 40)
        ws = calcWs(alphas, dataMatIn, classLabels)

        return b, alphas, ws
    # refer to logistic_regression
    def predict(self, X, weights):
        hypotheses = []
        for x in X:
            prob = weights[0] + weights[1] * x[0] + weights[2] * x[1]
            if prob > 0:
                hypotheses.append(1)
            else:
                hypotheses.append(-1)
        return hypotheses
    # refer to test_logistic_vs_synthetic_data
    def plot_fit(self, fit_line, datamatrix, labelmatrix):
        import matplotlib.pyplot as plt
        import numpy as np

#         weights = fit_line.getA()
        dataarray = np.asarray(datamatrix)
        n = dataarray.shape[0]

        # Keep track of the two classes in different arrays so they can be plotted later...
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if int(labelmatrix[i]) == 1:
                xcord1.append(dataarray[i, 0])
                ycord1.append(dataarray[i, 1])
            else:
                xcord2.append(dataarray[i, 0])
                ycord2.append(dataarray[i, 1])
        fig = plt.figure()

        # Plot the data as points with different colours
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')

        if fit_line:
            weights = fit_line
            # Plot the best-fit line
            x = np.arange(-2, 7, 0.1)
            y = (-weights[0] - weights[1] * x) / weights[2]
            ax.plot(x, y)

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    def accuracy(self, labels, hypotheses):
        count = 0.0
        correct = 0.0

        for l, h in zip(labels, hypotheses):
            count += 1.0
            if l == h:
                correct += 1.0
        return correct / count

    def print_confusion_matrix(self, labels, hypotheses):
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        count = 1.0
        for l, h in zip(labels, hypotheses):
            count += 1.0
            if l == 1 and h == 1:
                tp += 1.0
            elif l == 1 and h == 0:
                fn += 1.0
            elif l == 0 and h == 0:
                tn += 1.0
            else:
                fp += 1
        print ('-----------------------------')
        print ('\tConfusion Matrix')
        print ('-----------------------------')
        print ('\t\tPredicted')
        print ('\tActual\tNO\tYES')
        print ('-----------------------------')
        print ('\tNO\t', tn, '\t', fp)
        print ('-----------------------------')
        print ('\tYES\t', fn, '\t', tp)
        print ('-----------------------------')
