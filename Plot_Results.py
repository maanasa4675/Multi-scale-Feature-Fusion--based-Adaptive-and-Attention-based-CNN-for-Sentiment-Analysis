import matplotlib
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, roc_curve
from itertools import cycle
import xlwt
from xlwt import Workbook


def plot_results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7]
    Algorithm = ['TERMS', 'OOA', 'CSO', 'DOX', 'RSO', 'PROPOSED']
    Classifier = ['TERMS', 'LSTM', 'DTCN', 'RNN', 'CNN', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Learnperc - Dataset', i + 1, 'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 2):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        Table.add_column(Classifier[5], value1[4, :])
        print('-------------------------------------------------- Learnperc - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval.shape[0], eval.shape[2]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[2]):
                for i in range(eval.shape[0]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
        fig = plt.figure()
        bx = fig.add_axes([0.11, 0.11, 0.8, 0.8])
        Dataset = ['Dataset1', 'Dataset2', 'Dataset3']
        bx.plot(Dataset, Graph[:, 0], color='r', linewidth=3, marker='x', markerfacecolor='b', markersize=16,
                label="TSO-MFF-AACNet")
        bx.plot(Dataset, Graph[:, 1], color='g', linewidth=3, marker='D', markerfacecolor='red', markersize=12,
                label="BWO-MFF-AACNet")
        bx.plot(Dataset, Graph[:, 2], color='b', linewidth=3, marker='x', markerfacecolor='green', markersize=16,
                label="CO-MFF-AACNet")
        bx.plot(Dataset, Graph[:, 3], color='c', linewidth=3, marker='D', markerfacecolor='cyan', markersize=12,
                label="RSO-MFF-AACNet")
        bx.plot(Dataset, Graph[:, 4], color='k', linewidth=3, marker='x', markerfacecolor='black', markersize=16,
                label="FORSO-MFF-AACNet")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14),
                   ncol=3, fancybox=True, shadow=True)
        # plt.xlabel('No Of Datasets')
        plt.ylabel(Terms[Graph_Terms[j]])
        # plt.ylim([80, 100])
        # plt.legend(loc=4)
        path1 = "./Results/_%s_line.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        # ax = plt.axes(projection="3d")
        ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
        X = np.arange(3)
        ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="LSTM")
        ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="RNN")
        ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="Resnet")
        ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="CNN")
        ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="FORSO-MFF-AACNet")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                   ncol=3, fancybox=True, shadow=True)
        plt.xticks(X + 0.20, ('Dataset1', 'Dataset2', 'Dataset3'))
        # plt.xlabel('No Of Datasets')
        plt.ylabel(Terms[Graph_Terms[j]])
        # plt.legend(loc=1)
        path1 = "./Results/_%s_bar.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path1)
        plt.show()


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'OOA', 'CSO', 'DOX', 'RSO', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(3):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='TSO-MFF-AACNet')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='BWO-MFF-AACNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='CO-MFF-AACNet')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='RSO-MFF-AACNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='FORSO-MFF-AACNet')
        plt.xlabel('No. of Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


import seaborn as sns


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    no_of_Dataset = 3
    for n in range(no_of_Dataset):
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n]), np.asarray(Predict[n]))
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title("Accuracy")
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.savefig(path)
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['LSTM', 'RNN', 'Resnet', 'CNN', 'FORSO-MFF-AACNet']
    for a in range(3):  # For 5 Datasets
        # Actual = np.load('Target.npy', allow_pickle=True).astype('int')
        Actual = np.load('Targets.npy', allow_pickle=True)[a]

        colors = cycle(["blue", "darkorange", "red", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    plot_results()
    plotConvResults()
    Plot_ROC_Curve()
    Plot_Confusion()
