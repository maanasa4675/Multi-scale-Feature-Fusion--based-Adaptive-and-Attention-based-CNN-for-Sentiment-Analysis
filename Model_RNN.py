import numpy as np
# https://www.tensorflow.org/guide/keras/rnn
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from Evaluation import evaluation


def Model_RNN(train_data, train_target, test_data, test_target):
    pred = np.zeros(test_target.shape)
    for i in range(train_target.shape[1]):
        out, model = RNN_train(train_data, train_target[:, i], test_data)  # RNN
        # out = np.reshape(out, pred.shape[0])
        out = np.round(out)
        pred[:] = out

    Eval = evaluation(pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return testPredict, model
