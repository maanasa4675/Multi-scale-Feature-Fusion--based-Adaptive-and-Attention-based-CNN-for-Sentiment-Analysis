import numpy as np
from prettytable import PrettyTable



def plot_results():
    eval = np.load('Evaluate_all1.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7]
    Algorithm = ['TERMS', 'OOA', 'CSO', 'DOX', 'RSO', 'PROPOSED']
    Classifier = ['TERMS', 'Attention-CNN', 'BERT with CNN', 'Transformer with CNN', 'Word2Vector with CNN', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        # Table = PrettyTable()
        # Table.add_column(Algorithm[0], Terms)
        # for j in range(len(Algorithm) - 1):
        #     Table.add_column(Algorithm[j + 1], value1[j, :])
        # print('-------------------------------------------------- Learnperc - Dataset', i + 1, 'Algorithm Comparison',
        #       '--------------------------------------------------')
        # print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 2):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        Table.add_column(Classifier[5], value1[4, :])
        print('-------------------------------------------------- Dataset', i + 1, 'Ablation Experiment Comparison',
              '--------------------------------------------------')
        print(Table)

if __name__ == '__main__':
    plot_results()