import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars

def Model_DSC_ATCANN(Train_Data, Train_Target, Test_Data, Test_Target,
                                               soln):
    pass

def Model_MTL_MSCNN_LSTM(Train_Data, Train_Target, Test_Data, Test_Target):
    pass

def Model_CNN_GRU(Train_Data, Train_Target, Test_Data, Test_Target):
    pass

def Model_DNN_CNN_RNN(Train_Data, Train_Target, Test_Data, Test_Target):
    pass

def DO(initsol, fname, xmin, xmax, Max_iter):
    pass


def EOO(initsol, fname, xmin, xmax, Max_iter):
    pass


def TFMOA(initsol, fname, xmin, xmax, Max_iter):
    pass


def HGSO(initsol, fname, xmin, xmax, Max_iter):
    pass

def Proposed(initsol, fname, xmin, xmax, Max_iter):
    pass

def Model_DNN_RNN():
    pass


def objfun_feat(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            feat = Feat[:, sol]
            varience = np.var(feat)
            Fitn[i] = 1 / varience
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        feat = Feat[:, sol]
        varience = np.var(feat)
        Fitn = 1 / varience
        return Fitn


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_DTCNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = 1/(Eval[4] + Eval[7])
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_DTCNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(predict, Test_Target)
        Fitn = 1 / (Eval[4] + Eval[7])
        return Fitn
