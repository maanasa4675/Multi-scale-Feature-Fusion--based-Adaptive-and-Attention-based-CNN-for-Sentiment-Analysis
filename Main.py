import os
import gensim
import numpy as np
import pandas as pd
import xlrd
from sklearn.utils import shuffle
from numpy import matlib
from BWO import BWO
from CO import CO
from Model_CNN import Model_CNN
from Model_LSTM import Model_LSTM
from Model_RNN import Model_RNN
from Model_Resnet import Model_Resnet
from Normalize import normalize
from gensim.models import word2vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import random as rn
from transformers import T5ForConditionalGeneration,T5Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Global_Vars import Global_Vars
from TSO import TSO
from objfun_feat import objfun_cls
from Proposed import Proposed
from RSO import RSO
from Tfidf import TF_IDF
from transformers import BertTokenizer, BertModel
from Plot_Results import *
import torch

no_dataset = 3


# Removing Puctuations
def rem_punct(my_str):
    # define punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char + " "

    # display the unpunctuated string
    return no_punct


## Read Datasets and Preprocessing
an = 0
if an == 1:
    Data = []
    Pre_Data = []
    Target = []
    ps = PorterStemmer()
    subfolder = os.listdir("./Dataset")
    for sub_iter in range(len(subfolder)):  # For all folders
        sub_folders = subfolder[sub_iter]
        Each_Data = []
        Each_Tar = []
        Each_Pre = []
        for files in os.listdir("./Dataset" + "/" + sub_folders):  # For all sub_folders
            loc = "%s/%s/%s" % ('Dataset', sub_folders, files)
            if sub_folders == 'Dataset3':
                wb = xlrd.open_workbook(loc, encoding_override='latin1')
                dataframe = pd.read_excel(wb)  # Read Excel file
                new_dataframe = dataframe.dropna()
                tar = new_dataframe.values[:, 1]
                tar[tar == 'positive'] = 1
                tar[tar == 'negative'] = 0
                data = new_dataframe.values[:, 0]
            else:
                wb = xlrd.open_workbook(loc, encoding_override='latin1')
                dataframe = pd.read_excel(wb)  # Read Excel file
                if sub_folders == 'Dataset1':
                    targ = dataframe.values[:, 0]
                    targ[targ == 'positive'] = 1
                    targ[targ == 'negative'] = 0
                    targ[targ == 'neutral'] = 2
                    uni = np.unique(targ)
                    tar = np.zeros((len(targ), len(uni)))
                    for j in range(len(uni)):
                        index = np.where(targ == uni[j])
                        tar[index, j] = 1
                    data = dataframe.values[:, 1]
                elif sub_folders == 'Dataset2':
                    tar = dataframe.values[:, 1]
                    tar[tar == -1] = 0
                    data = dataframe.values[:, 0]
            Pre = []
            for i in range(len(data)):
                print(sub_folders, i, len(data))
                text_tokens = word_tokenize(data[i])  # convert it in to tokens
                stem = []
                for w in text_tokens:  # Stemming with Lower case conversion
                    stem_tokens = ps.stem(w)
                    stem.append(stem_tokens)
                words = [word for word in stem if
                         not word in stopwords.words()]  # tokens without stop words
                prep = rem_punct(words)  # Punctuation Removal
                Pre.append(prep)
            Each_Data.append(data)
            Each_Tar.append(tar)
            Each_Pre.append(np.asarray(Pre))
        Data.append(Each_Data)
        Target.append(Each_Tar)
        Pre_Data.append(Each_Pre)
    np.save('Data.npy', Data)
    np.save('Target.npy', Target)
    np.save('Pre_Data.npy', Pre_Data)

# Feature Extrection using BERT
an = 0
if an == 1:
    BERT = []
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Input text
    text = np.load('Pre_Data.npy', allow_pickle=True)
    for i in range(50, len(text)):
        file = text[0][0]
        for j in range(len(file)):
            tokens = tokenizer.tokenize(file[j])
            tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Convert tokens to token IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension

        # Perform forward pass through the BERT model
        outputs = model(token_ids_tensor)

        # Extract the last hidden state of the BERT model
        last_hidden_state = outputs.last_hidden_state
        BERT.append(last_hidden_state)
    np.save('BERT.npy', BERT)  # Save the BERT data

# Word2vec  and TF-IDF
an = 0
if an == 1:
    tweets = np.load('BERT.npy', allow_pickle=True)
    pca = PCA(n_components=1)
    Vector = []
    for d in range(len(tweets)):
        data1 = np.asarray(tweets[d])[:, 0]
        vect = np.zeros((len(data1), 101))
        for i in range(len(data1)):
            print(d, i)
            if not len(data1[i]):
                vect[i, :] = np.zeros((101))
            else:
                val = data1[i]
                model2 = gensim.models.Word2Vec(val, min_count=1,
                                                window=5, sg=1)  # Word2vector
                v = model2.wv.vectors
                p1 = pca.fit_transform(v.transpose())
                vect[i, 0:100] = p1[0:100].reshape(1, -1)
                vect[i, 100] = np.max(np.unique(TF_IDF(val)))
        Vector.append(vect)
    np.save('Vector.npy', Vector)

# Transformer
an = 0
if an == 1:
    paath = []





# optimization for Prediction
an = 0
if an == 1:
    for n in range(no_dataset):
        Data = np.load('Data' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Data = Data
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([5, 5, 50], Npop, 1)
        xmax = matlib.repmat([255, 50, 250], Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 25

        print("AOA...")
        [bestfit1, fitness1, bestsol1, time] = TSO(initsol, fname, xmin, xmax, Max_iter)  # AOA

        print("BWO...")
        [bestfit2, fitness2, bestsol2, time1] = BWO(initsol, fname, xmin, xmax, Max_iter)  # BWO

        print("TSO...")
        [bestfit3, fitness3, bestsol3, time2] = CO(initsol, fname, xmin, xmax, Max_iter)  # TSO

        print("EO...")
        [bestfit4, fitness4, bestsol4, time3] = RSO(initsol, fname, xmin, xmax, Max_iter)  # Model_EO

        print("Proposed...")
        [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed IEO

        BestSol = [bestsol1, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_CLS.npy', BestSol)

# Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_dataset):
        Feature = np.load('Features.npy', allow_pickle=True)  # loading step
        Target = np.load('Target.npy', allow_pickle=True)[:Feature.shape[0], :]  # loading step
        BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]  # loading step
        EVAL = []
        Learnper = [0.35, 0.45, 0.55, 0.65, 0.754, 0.85]
        for learn in range(len(Learnper)):
            learnperc = round(Feature.shape[0] * Learnper[learn])  # Split Training and Testing Datas
            Train_Data = Feature[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feature[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 8))
            for j in range(BestSol.shape[0]):
                print(learn, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :] = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)  # CNN With optimization
            Eval[5, :], pred = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)  # Model LSTM
            Eval[7, :], pred1 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model DTCN
            Eval[8, :], pred2 = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model RNN
            Eval[9, :], pred3 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)  # CNN Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Eval_all.npy', Eval_all)  # Save Eval all

plot_results()
plotConvResults()
Plot_ROC_Curve()
Plot_Confusion()
