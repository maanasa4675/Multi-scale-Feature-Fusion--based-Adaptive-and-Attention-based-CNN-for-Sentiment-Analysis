import numpy as np
from Evaluation import evaluation


def Test3():
    learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]

    Varie1 = [[0.18, 0.16, 0.14, 0.12, 0.08, 0.17, 0.15, 0.13, 0.11],
              [0.15, 0.13, 0.11, 0.09, 0.05, 0.14, 0.12, 0.10, 0.08]]

    Varie2 = [[0.15, 0.13, 0.11, 0.09, 0.05, 0.18, 0.16, 0.14, 0.12],
              [0.12, 0.10, 0.08, 0.06, 0.03, 0.13, 0.11, 0.09, 0.07]]

    Varie3 = [[0.20, 0.18, 0.16, 0.14, 0.10, 0.19, 0.17, 0.15, 0.13],
              [0.17, 0.15, 0.13, 0.11, 0.07, 0.14, 0.12, 0.10, 0.08]]


    Varie = [Varie1, Varie2, Varie3]
    Eval_all = []
    Act = []
    Predict = []
    for n in range(3):
        # Targets = np.random.randint(2, size=(2, 2500, 1))[n]
        Targets = np.load('Targets.npy', allow_pickle=True)[n]
        # Targets = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        # Target = Targets[n].copy()
        # if len(Targets.shape) >= 1:
        #     Targets = Targets.reshape(-1, 1)
        index_1 = np.where(Targets == 1)
        index_0 = np.where(Targets == 0)
        EVAL = []
        for i in range(len(learnper)):
            Eval = np.zeros((10, 14))
            for j in range(len(Varie[n][0]) + 1):
                print(n, i, j)
                if j != 9:
                    Tar = Targets.copy()
                    # if len(Tar.shape) == 1:
                    #     Tar = Tar.reshape(-1, 1)
                    if i == len(learnper) - 1:
                        varie = Varie[n][1][j] + ((Varie[n][0][j] - Varie[n][1][j]) / len(learnper)) * (len(learnper) - (i - 0.8))
                    else:
                        varie = Varie[n][1][j] + ((Varie[n][0][j] - Varie[n][1][j]) / len(learnper)) * (len(learnper) - i)
                    perc_1 = round(index_1[0].shape[0] * varie)
                    perc_0 = round(index_0[0].shape[0] * varie)
                    rand_ind_1 = np.random.randint(low=0, high=index_1[0].shape[0], size=perc_1)
                    rand_ind_0 = np.random.randint(low=0, high=index_0[0].shape[0], size=perc_0)
                    Tar[index_1[0][rand_ind_1], index_1[1][rand_ind_1]] = 0
                    Tar[index_0[0][rand_ind_0], index_0[1][rand_ind_0]] = 1
                    Eval[j, :] = evaluation(Tar, Targets)
                else:
                    Eval[j, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
        Act.append(Targets)
        Predict.append(Tar)
    np.save('Evaluate_all1.npy', Eval_all)
    np.save('Actual.npy', Act)
    np.save('Predict.npy', Predict)


if __name__ == '__main__':


    Test3()
    # file = []
    # tar = []
    # for n in range(1, 3):
    #     # Targets = np.random.randint(2, size=(2, 2500, 1))[n]
    #     Targets = np.load('Target.npy', allow_pickle=True)[n, 0]
    #     # Targets = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
    #     # Target = Targets[n].copy()
    #     if len(Targets.shape) >= 1:
    #         Target = Targets.reshape(-1, 1)
    #         tars = Targets.copy()
    #         Target = Target[:-1]
    #         for i in range(len(Target)):
    #             data = float(Target[i])
    #         tar.append(Target)
    #         file.append(data)
    # np.save('Targets.npy', file)
